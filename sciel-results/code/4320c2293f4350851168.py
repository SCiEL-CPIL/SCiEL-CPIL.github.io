import numpy as np
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 3, 4]
n, m, p = 3, 3, 4

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        S = np.einsum('ijt,jkt,kit->ijk', 
                      U.reshape(n,m,r), V.reshape(m,p,r), W.reshape(p,n,r))
        # Check if S equals the identity-like tensor
        # For matrix multiplication: S[i,j,k] should be 1 if this is a valid term
        # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
        # Wait, let me reconsider. The multiplication tensor is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this context)
        # No wait, that's not right either.
        
        # The matrix multiplication tensor M has:
        # M[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # So actually M is the identity in the middle index j? No...
        
        # Let me think again. For matrix multiplication A (n×m) * B (m×p) = C (n×p):
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
        # The tensor T such that T(A,B) = C has:
        # T[i,j,k] = 1 for all i,j,k (since every (i,j,k) triple contributes)
        # Wait no, that would make it the all-ones tensor, which has rank 1.
        
        # I think I'm confusing myself. Let me look at this differently.
        # The multiplication tensor is defined by:
        # <T, A ⊗ B ⊗ C> = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
        # But we want C = A*B, so C[i,k] = sum_j A[i,j]*B[j,k]
        # This means the tensor T has components T[i,j,k] such that:
        # sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] = sum_{i,k} C[i,k]^2 (if we're doing Frobenius)
        
        # Actually, the standard way: the multiplication tensor M_{n,m,p} has:
        # M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # And a rank-r decomposition is U (n*m × r), V (m*p × r), W (n*p × r) such that:
        # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
        # (or 0 for invalid, but here all are valid)
        
        # So we need: for all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
        target = np.ones((n, m, p))
        diff = S - target
        return np.max(np.abs(diff)) < 1e-9, S
    
    def schoolbook():
        """Return the schoolbook rank-36 decomposition."""
        U = np.zeros((n*m, 36), dtype=int)
        V = np.zeros((m*p, 36), dtype=int)
        W = np.zeros((p*n, 36), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def try_local_search(U, V, W, max_iters=50000):
        """Local search to try to reduce rank by merging columns."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        # Precompute S
        def compute_S():
            return np.einsum('ijt,jkt,kit->ijk', 
                            U.reshape(n,m,r), V.reshape(m,p,r), W.reshape(p,n,r))
        
        S = compute_S()
        
        # Try to find pairs of columns that can be merged
        # Two columns t1, t2 can be merged if there exists a linear combination
        # that preserves the decomposition
        # This is equivalent to finding if the two columns are "redundant"
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = r
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Randomly try to remove a column
            if best_rank <= 24:  # We're doing well
                break
                
            t = random.randint(0, best_rank - 1)
            
            # Try removing column t by checking if it's linearly dependent
            # in a way that preserves the tensor
            # Simple approach: try to express column t as combination of others
            # For now, just try zeroing out one column and see if we can fix it
            
            # Alternative: try swapping values in a column
            row_u = random.randint(0, n*m - 1)
            row_v = random.randint(0, m*p - 1)
            row_w = random.randint(0, p*n - 1)
            
            # Try changing one entry
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                old = U[row_u, t]
                U[row_u, t] = random.choice([-1, 0, 1])
                if U[row_u, t] != old:
                    S_new = compute_S()
                    if np.max(np.abs(S_new - np.ones((n,m,p)))) < 1e-9:
                        pass  # Still valid, keep it
                    else:
                        U[row_u, t] = old
            elif mat_choice == 1:
                old = V[row_v, t]
                V[row_v, t] = random.choice([-1, 0, 1])
                if V[row_v, t] != old:
                    S_new = compute_S()
                    if np.max(np.abs(S_new - np.ones((n,m,p)))) < 1e-9:
                        pass
                    else:
                        V[row_v, t] = old
            else:
                old = W[row_w, t]
                W[row_w, t] = random.choice([-1, 0, 1])
                if W[row_w, t] != old:
                    S_new = compute_S()
                    if np.max(np.abs(S_new - np.ones((n,m,p)))) < 1e-9:
                        pass
                    else:
                        W[row_w, t] = old
        
        return best_U, best_V, best_W, best_rank
    
    def structured_search(rank_target, max_restarts=10):
        """Try to find a decomposition of given rank using structured patterns."""
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            U = np.zeros((n*m, rank_target), dtype=int)
            V = np.zeros((m*p, rank_target), dtype=int)
            W = np.zeros((p*n, rank_target), dtype=int)
            
            # Initialize with random sparse pattern
            for t in range(rank_target):
                # Each column gets a sparse pattern
                u_sparse = random.randint(2, min(6, n*m))
                v_sparse = random.randint(2, min(6, m*p))
                w_sparse = random.randint(2, min(6, p*n))
                
                u_rows = random.sample(range(n*m), u_sparse)
                v_rows = random.sample(range(m*p), v_sparse)
                w_rows = random.sample(range(p*n), w_sparse)
                
                for row in u_rows:
                    U[row, t] = random.choice([-1, 1])
                for row in v_rows:
                    V[row, t] = random.choice([-1, 1])
                for row in w_rows:
                    W[row, t] = random.choice([-1, 1])
            
            # Local search
            U, V, W, final_rank = try_local_search(U, V, W, max_iters=10000)
            
            valid, S = verify(U, V, W)
            if valid:
                return U, V, W
            
        return None
    
    # First, get schoolbook as baseline
    U_school, V_school, W_school = schoolbook()
    best_rank = 36
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to improve from schoolbook
    U2, V2, W2, rank2 = try_local_search(U_school, V_school, W_school, max_iters=30000)
    valid2, _ = verify(U2, V2, W2)
    if valid2 and rank2 < best_rank:
        best_rank = rank2
        best_U, best_V, best_W = U2, V2, W2
    
    # Try structured search for lower ranks
    for r in [30, 28, 26, 24]:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        sol = structured_search(r, max_restarts=5)
        if sol is not None:
            U_sol, V_sol, W_sol = sol
            valid, _ = verify(U_sol, V_sol, W_sol)
            if valid:
                best_rank = r
                best_U, best_V, best_W = U_sol, V_sol, W_sol
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
