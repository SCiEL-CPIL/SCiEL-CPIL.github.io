import time
import numpy as np
from itertools import combinations

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # A is n x m = 4x4, B is m x p = 4x5, C is n x p = 4x5
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank), indexed as W[k*n+i]
    
    def make_schoolbook():
        """Create the schoolbook decomposition with rank = n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify_decomp(U, V, W, n_tests=5, seed=42):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rng = np.random.RandomState(seed)
        rank = U.shape[1]
        for _ in range(n_tests):
            A_t = rng.randint(-3, 4, (n, m)).astype(float)
            B_t = rng.randint(-3, 4, (m, p)).astype(float)
            C_expected = A_t @ B_t
            
            A_flat = A_t.flatten()
            B_flat = B_t.flatten()
            
            # Compute: sum_r (A_flat . U[:,r]) * (B_flat . V[:,r]) * W[:,r]
            # Result is a vector of length p*n in order k*n+i
            C_vec = np.zeros(p*n)
            for r in range(rank):
                prod = np.dot(A_flat, U[:, r]) * np.dot(B_flat, V[:, r])
                C_vec += prod * W[:, r]
            
            # Reshape C_vec to (p, n) then transpose to (n, p)
            C_comp = C_vec.reshape(p, n).T
            
            if not np.allclose(C_comp, C_expected, atol=1e-10):
                return False
        return True
    
    def try_reduce_rank(U, V, W, rng, max_attempts=2000):
        """
        Try to reduce rank by finding pairs of terms that can be merged.
        A pair (r1, r2) can be merged if there exists a linear combination
        that eliminates one term while preserving the tensor.
        
        Strategy: For each pair (r1, r2), check if W[:,r1] is a scalar multiple
        of W[:,r2] (or more generally, if the three vectors are linearly dependent
        in a way that allows elimination).
        
        More practically: try random linear combinations and check validity.
        """
        rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            if best_rank <= 1:
                break
            
            # Pick two random distinct terms
            r1, r2 = rng.choice(best_rank, 2, replace=False)
            
            # Try to eliminate r2 by replacing it with a linear combination
            # that makes its W vector zero (or makes the term redundant)
            # 
            # Approach: For a valid decomposition, if we have two terms r1 and r2,
            # we can try: new_r2 = alpha*r1 + r2 for some alpha, and check if
            # the new decomposition is still valid. If W[:,new_r2] becomes zero,
            # we can drop it.
            #
            # More generally, we look for alpha such that the new set of terms
            # still decomposes the tensor. This is hard to check directly.
            #
            # Alternative: use the fact that if W[:,r2] = c * W[:,r1] for some c,
            # then we can potentially merge. Let's check this condition.
            
            w1 = W[:, r1]
            w2 = W[:, r2]
            
            # Check if w2 is a scalar multiple of w1
            if np.linalg.norm(w1) > 1e-10:
                c = np.dot(w2, w1) / np.dot(w1, w1)
                if np.allclose(w2, c * w1, atol=1e-10):
                    # w2 = c * w1, so we can try to merge
                    # New term: replace r2 with (U[:,r2] - c*U[:,r1]) * (V[:,r2] - c*V[:,r1])
                    # Actually, this isn't quite right. Let me think...
                    # 
                    # If W[:,r2] = c*W[:,r1], then the contribution of term r2 is:
                    # (A.U_r2)(B.V_r2) * c * W_r1
                    # And term r1 contributes: (A.U_r1)(B.V_r1) * W_r1
                    # Together: [(A.U_r1)(B.V_r1) + c*(A.U_r2)(B.V_r2)] * W_r1
                    # This is NOT a single rank-1 term in general.
                    # 
                    # So this approach doesn't directly work. Let me try a different strategy.
                    pass
            
            # Alternative approach: random perturbation
            # Try replacing term r2 with a random linear combination of all other terms
            # and check if the result is still valid.
            # 
            # Actually, let me use a simpler but effective approach:
            # Try to find if any term is redundant (its removal doesn't change the result).
            
            # Check if term r2 is redundant: remove it and verify
            if best_rank > 1:
                U_test = np.delete(best_U, r2, axis=1)
                V_test = np.delete(best_V, r2, axis=1)
                W_test = np.delete(best_W, r2, axis=1)
                
                if verify_decomp(U_test, V_test, W_test, n_tests=3, seed=attempt % 100):
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    best_rank -= 1
        
        return best_U, best_V, best_W, best_rank
    
    def try_merge_terms(U, V, W, rng, max_attempts=5000):
        """
        Try to merge pairs of terms using the identity:
        If we have terms r1 and r2, we can sometimes replace them with a single term
        if certain conditions hold.
        
        A more systematic approach: use random 2x2 matrix multiplication tricks.
        For two terms with (U1,V1,W1) and (U2,V2,W2), if W1 and W2 are linearly
        independent, we can't merge. But if they're dependent, we might.
        
        Let me try a different structural approach: look for terms where
        U[:,r1] is proportional to U[:,r2] or V[:,r1] is proportional to V[:,r2].
        """
        rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            if best_rank <= 1:
                break
            
            r1, r2 = rng.choice(best_rank, 2, replace=False)
            
            # Check if U[:,r1] is proportional to U[:,r2]
            u1, u2 = best_U[:, r1], best_U[:, r2]
            v1, v2 = best_V[:, r1], best_V[:, r2]
            w1, w2 = best_W[:, r1], best_W[:, r2]
            
            # Case 1: u2 = a*u1
            if np.linalg.norm(u1) > 1e-10:
                a = np.dot(u2, u1) / np.dot(u1, u1)
                if np.allclose(u2, a*u1, atol=1e-10):
                    # u2 = a*u1, so term r2 = (A.u1)*a*(B.v2)*(w2)
                    # = a*(A.u1)*(B.v2)*(w2)
                    # Term r1 = (A.u1)*(B.v1)*(w1)
                    # If w2 = b*w1, then term r2 = a*b*(A.u1)*(B.v2)*(w1)
                    # Still not mergeable in general.
                    pass
            
            # Case 2: Try to find alpha such that new decomposition works
            # Replace r2 with: U_new = U_r2 + alpha*U_r1, V_new = V_r2, W_new = W_r2
            # and adjust r1: U_r1 stays, V_new1 = V_r1, W_new1 = W_r1 - alpha*(something)
            # This is getting complicated. Let me try a brute-force random approach.
            
            # Random linear combination test
            alpha = rng.uniform(-2, 2)
            U_test = best_U.copy()
            V_test = best_V.copy()
            W_test = best_W.copy()
            
            # Replace term r2 with alpha*r1 + r2
            U_test[:, r2] = alpha * U_test[:, r1] + U_test[:, r2]
            # Check validity
            if verify_decomp(U_test, V_test, W_test, n_tests=2, seed=attempt % 100):
                # Now check if we can drop any term
                for r in range(best_rank):
                    if r == r1 or r == r2:
                        continue
                    U_drop = np.delete(U_test, r, axis=1)
                    V_drop = np.delete(V_test, r, axis=1)
                    W_drop = np.delete(W_test, r, axis=1)
                    if verify_decomp(U_drop, V_drop, W_drop, n_tests=3, seed=(attempt+r) % 100):
                        best_U = U_drop
                        best_V = V_drop
                        best_W = W_drop
                        best_rank -= 1
                        break
        
        return best_U, best_V, best_W, best_rank
    
    # Start with schoolbook
    U, V, W = make_schoolbook()
    initial_rank = n * m * p  # 80
    
    # Verify initial
    assert verify_decomp(U, V, W), "Schoolbook verification failed"
    
    # Phase 1: Try to remove redundant terms
    rng = np.random.RandomState(12345)
    U, V, W, rank1 = try_reduce_rank(U, V, W, rng, max_attempts=5000)
    
    # Phase 2: Try merging terms
    rng2 = np.random.RandomState(67890)
    U, V, W, rank2 = try_merge_terms(U, V, W, rng2, max_attempts=10000)
    
    # Final verification
    if not verify_decomp(U, V, W, n_tests=10, seed=999):
        # Fallback to schoolbook
        U, V, W = make_schoolbook()
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
