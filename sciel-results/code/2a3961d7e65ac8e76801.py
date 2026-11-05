"""Matrix multiplication 3x3x6 with improved decomposition."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook(n=3, m=3, p=6):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=3, m=3, p=6, tol=1e-10):
    """Check if the decomposition is valid: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    # Also verify that "unnecessary" products are zero
    # Actually, we just need the trilinear form to match the multiplication tensor
    # The multiplication tensor T[i,j,k] = 1 if this is a valid (i,j,k) product position
    # For 3x3 * 3x6: A is 3x3, B is 3x6, C = A*B is 3x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for valid (i,j,k) where i in [0,3), j in [0,3), k in [0,6)
    # And the decomposition must satisfy: for each rank factor c,
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    #   = 0 for all invalid (i,j,k)
    # But wait, in the standard formulation, ALL (i,j,k) in the full index set are valid
    # for the "multiplication" tensor. Let me reconsider.
    # 
    # The task says: "exactly decompose the matrix multiplication tensor"
    # For multiplying 3x3 by 3x6:
    #   (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i,j,k) with i in [0,3), j in [0,3), k in [0,6)
    # T[i,j,k] = 1 for all valid positions (all of them since j always in [0,3))
    # 
    # But the U,V,W matrices are indexed as:
    #   U[i*m + j, c] where i in [0,n), j in [0,m)
    #   V[j*p + k, c] where j in [0,m), k in [0,p)
    #   W[k*n + i, c] where k in [0,p), i in [0,n)
    # 
    # So the check is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # This is exactly what we're checking. But we also need it to be 0 for "cross terms"?
    # No - the tensor IS exactly the multiplication tensor, so all (i,j,k) triples have value 1.
    # 
    # Wait, that can't be right. If all entries are 1, then the rank would be 1.
    # 
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for n x m times m x p is a 3-tensor T of shape (n*m, m*p, p*n)
    # where T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k' else 0.
    # No wait, that's not standard either.
    # 
    # Standard: T_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # This means T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 iff j = j' AND i = i' AND k = k'
    # 
    # So the decomposition requires:
    # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c]
    # where T[a,b,c] = 1 iff (a mod m == b div p) and (a div m == c mod n) and (b mod p == c div n)
    # 
    # Let me verify with the schoolbook:
    # For (i,j,k): a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] should be 1 (since j==j, i==i, k==k)
    # And for any other combination, it should be 0.
    # 
    # In the schoolbook, each column c corresponds to exactly one (i,j,k) triple,
    # and U,V,W have 1s only at the corresponding positions. So the product is 1
    # only for that specific (a,b,c) combination. Good.
    
    # Now check that cross-terms are zero
    for a in range(n * m):
        for b in range(m * p):
            for c_idx in range(p * n):
                # Determine if T[a,b,c_idx] should be 1 or 0
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c_idx // n
                i2 = c_idx % n
                expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                
                actual = 0.0
                for cc in range(rank):
                    actual += Ua[a, cc] * Va[b, cc] * Wa[c_idx, cc]
                
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 6
    
    # Start with schoolbook rank 54
    best = _schoolbook(n, m, p)
    best_rank = 54
    
    # Try to find a better decomposition using random search / optimization
    # The key insight: for 3x3 x 3x6, we can potentially use a block structure
    
    # Strategy: Try random rank-r decompositions and check validity
    # This is expensive, so we'll use a smarter approach
    
    # Alternative: Use the fact that 3x3x6 = 3x3x3 + 3x3x3 (two blocks)
    # If we had a 3x3x3 decomposition of rank 23, we could get rank 46
    # But 3x3x6 is not simply two 3x3x3 blocks in a way that allows sharing
    
    # Let me try a direct approach: random initialization + gradient-free optimization
    # to minimize the residual
    
    # First, let's try a structured approach
    # The tensor for 3x3 * 3x6:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can write this as:
    # C = A * B where A is 3x3, B is 3x6
    # 
    # One approach: decompose B into two 3x3 matrices B1, B2
    # C = A * B1 + A * B2
    # If we use the same A factors for both, we save the A-side rank
    # But the B-side still needs separate factors
    
    # Actually, let's just try random rank r decompositions for r < 54
    # and see if any work. This is a needle in a haystack, but let's try small r.
    
    # Better: use the bilinear structure. 
    # For each rank factor c, we have:
    #   U[:, c] is a vector in R^(n*m) = R^9
    #   V[:, c] is a vector in R^(m*p) = R^18  
    #   W[:, c] is a vector in R^(p*n) = R^18
    # 
    # The constraint is: for all (a,b,c_idx), sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]
    # 
    # This is equivalent to: the tensor T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # 
    # Let's try a coordinate descent / greedy approach
    # Start with a random rank-r decomposition and try to reduce residual
    
    # Actually, let me try a known result: 
    # rank(3,3,6) is known to be at most 40 (from various constructions)
    # and at least some lower bound
    
    # Let me try random search for rank 40, 35, 30, etc.
    # But random search in high dimensions is unlikely to work
    
    # Better approach: Use the "lifting" technique
    # If we have a decomposition of 4x4x4 of rank 49, we can restrict to 3x3x6
    
    # Or use Strassen-style divide and conquer
    
    # For now, let's try a simple iterative improvement:
    # Start with schoolbook, then try to merge/remove factors
    
    # Actually, the most promising approach for a quick win:
    # Try to find a rank-46 decomposition by treating it as two 3x3x3 blocks
    # with shared structure
    
    # Let me try: express the 3x3x6 tensor as a combination
    # T[i,j,k] = delta condition
    # 
    # Split k into k1 (0,1,2) and k2 (3,4,5)
    # T = T1 + T2 where T1 covers k in {0,1,2} and T2 covers k in {3,4,5}
    # T1[i,j,k] = 1 iff j==j' and i==i' and k==k' for k in {0,1,2}
    # T2[i,j,k] = 1 iff j==j' and i==i' and k==k' for k in {3,4,5}
    # 
    # T1 is essentially a 3x3x3 tensor (with k reindexed to 0,1,2)
    # T2 is also a 3x3x3 tensor (with k reindexed to 0,1,2)
    # 
    # If we use the same U factors for both, and separate V,W factors:
    # T = sum_c U[:,c] ⊗ V1[:,c] ⊗ W1[:,c] + sum_c U[:,c] ⊗ V2[:,c] ⊗ W2[:,c]
    # 
    # This doesn't directly help unless we can share more.
    
    # Let me just try to do a local search from the schoolbook decomposition
    # to see if we can reduce the rank
    
    # Actually, let me try a completely different approach:
    # Use random vectors and solve for the third set of factors
    
    # Given U and V, the constraint becomes:
    # For all (a,b,c_idx): sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]
    # This is a linear system in W!
    # 
    # Specifically, for fixed U, V, the equation is:
    # sum_c (U[a,c]*V[b,c]) * W[c_idx,c] = T[a,b,c_idx]
    # 
    # Let M[a,b,c_idx,c] = U[a,c]*V[b,c]
    # Then sum_c M[a,b,c_idx,c] * W[c_idx,c] = T[a,b,c_idx]
    # 
    # For each c_idx, this gives: sum_c M[a,b,c_idx,c] * W[c_idx,c] = T[a,b,c_idx] for all (a,b)
    # That's (n*m)*(m*p) = 9*18 = 162 equations per c_idx, with p*n = 18 unknowns (W[c_idx, 0..17])
    # Wait no, W[c_idx, c] for c=0..rank-1, so rank unknowns.
    # And we have 162 equations.
    # So if rank < 162, the system is overdetermined.
    # 
    # But we can set up the full system:
    # Total equations: (n*m)*(m*p)*(p*n) = 9*18*18 = 2916
    # Total unknowns: rank * (p*n) = rank * 18 (for W) + rank * (n*m) = rank*9 (for U) + rank*(m*p) = rank*18 (for V)
    # = rank * 45
    # 
    # For rank = 54: 54*45 = 2430 unknowns, 2916 equations. Overdetermined.
    # For rank = 65: 65*45 = 2925 unknowns, 2916 equations. Underdetermined!
    # 
    # But the system is nonlinear (products of U,V,W), so it's not really a linear system
    # in the combined variables.
    
    # Let me try a different tactic: fix U and V randomly, solve for W via least squares,
    # then check if the residual is small. Iterate.
    
    np.random.seed(42)
    
    # Try various ranks
    for target_rank in [46, 45, 44, 43, 42, 40, 38, 36, 35, 34, 32, 30]:
        if time.time() - start_time > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - start_time > deadline - 30:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            
            # For each c_idx, solve: sum_c U[a,c]*V[b,c] * W[c_idx,c] = T[a,b,c_idx]
            # for all (a,b)
            # This is: A[c_idx] @ W[c_idx,:]^T = t[c_idx]
            # where A[c_idx][idx, c] = U[a,c]*V[b,c] with idx = a*(m*p)+b
            # and t[c_idx][idx] = T[a,b,c_idx]
            
            W = np.zeros((p * n, target_rank))
            residual = 0.0
            
            for c_idx in range(p * n):
                # Build the system for this c_idx
                # Equations: for all a in [0,9), b in [0,18)
                # sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]
                
                A_mat = np.zeros((n * m * m * p, target_rank))
                t_vec = np.zeros(n * m * m * p)
                
                eq_idx = 0
                for a in range(n * m):
                    for b in range(m * p):
                        # Compute T[a,b,c_idx]
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c_idx // n
                        i2 = c_idx % n
                        T_val = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                        
                        for c in range(target_rank):
                            A_mat[eq_idx, c] = U[a, c] * V[b, c]
                        t_vec[eq_idx] = T_val
                        eq_idx += 1
                
                # Solve A_mat @ w = t_vec for w = W[c_idx, :]
                # Least squares
                w, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, t_vec, rcond=None)
                W[c_idx, :] = w
                if len(residuals) > 0:
                    residual += residuals[0]
            
            # Check actual residual
            Ua = U
            Va = V
            Wa = W
            max_err = 0.0
            for a in range(n * m):
                for b in range(m * p):
                    for c_idx in range(p * n):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c_idx // n
                        i2 = c_idx % n
                        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                        actual = np.dot(Ua[a, :] * Va[b, :], Wa[c_idx, :])
                        max_err = max(max_err, abs(actual - expected))
            
            if max_err < 1e-8:
                # Convert to list of lists with rounded values
                U_list = [[round(x, 10) for x in Ua[row]] for row in range(n * m)]
                V_list = [[round(x, 10) for x in Va[row]] for row in range(m * p)]
                W_list = [[round(x, 10) for x in Wa[row]] for row in range(p * n)]
                # Clean up near-zero values
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for idx in range(len(row)):
                            if abs(row[idx]) < 1e-9:
                                row[idx] = 0
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better decomposition found, return schoolbook
    return best

# Test the schoolbook
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid = _verify(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}")
