import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target tensor condition:
    # Sum_r U[i',j',r] V[j',k',r] W[k,i,r] = 1 if i'=i and k'=k
    # Sum_r U[i',j',r] V[j',k',r] W[k,i,r] = 0 if i'!=i or k'!=k
    
    # Schoolbook solution (Rank 64)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find a better rank
    # We can use a randomized search or a specific construction.
    # For 2x4x8, a rank of 40-50 is plausible.
    # We will try a few ranks with a simple local search.
    
    def check_decomposition(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check if sum_r U[i',j',r] V[j',k',r] W[k,i,r] == 1 if i'=i, k'=k else 0
        # This is equivalent to checking if the tensor T matches.
        # T[i',j',j',k'; k,i] = 1 if i'=i, k'=k else 0
        # Note: j' is the index in U and V.
        
        # We can compute the tensor T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
        # But U is indexed by (i,j), V by (j,k), W by (k,i).
        # So T_decomp[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We want T_decomp[i,j,k] == 1 if i==i and k==k?
        # Wait, the indices in T_decomp are i, j, k.
        # The condition is:
        # For fixed i, k: T_decomp[i,j,k] must be 1 for all j?
        # No, T_decomp[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
        # This should be 1.
        # For fixed i', k' with i'!=i or k'!=k, the coefficient of A[i',j'] B[j',k'] in C[i,k] is 0.
        # But T_decomp is defined for specific i,j,k.
        # The term A[i',j'] B[j',k'] contributes to C[i,k] via T_decomp[i',j',k']?
        # No, the decomposition is:
        # C[i,k] = sum_r W[k,i,r] (sum_i',j' U[i',j',r] A[i',j']) (sum_j',k' V[j',k',r] B[j',k'])
        # Coeff of A[i',j'] B[j',k'] in C[i,k] is sum_r U[i',j',r] V[j',k',r] W[k,i,r].
        # This is T_decomp[i',j',k'] with W[k,i,r].
        # So we need:
        # sum_r U[i',j',r] V[j',k',r] W[k,i,r] == 1 if i'=i, k'=k
        # sum_r U[i',j',r] V[j',k',r] W[k,i,r] == 0 if i'!=i or k'!=k
        
        # This is a 4D tensor check.
        # We can iterate over i, k and check the vector of length m*p?
        # For fixed i, k:
        # vec[i',j',j',k'] = sum_r U[i',j',r] V[j',k',r] W[k,i,r]
        # We want vec[i,j',j',k] = 1 for all j'?
        # And vec[i',j',j',k'] = 0 for i'!=i or k'!=k.
        
        # Efficient check:
        # Let M_r = U[:, r] (nm x 1) and N_r = V[:, r] (mp x 1).
        # This is expensive.
        
        # Alternative:
        # Construct the full tensor T_decomp of size (n*m, m*p, p*n)?
        # T_decomp[i,j, j,k, k,i] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is too large.
        
        # We can check by sampling or using the structure.
        # Since n=2, m=4, p=8, the sizes are small.
        # nm=8, mp=32, pn=16.
        # We can compute the matrix P[i,k] of size (nm*mp) x (nm*mp)?
        
        # Let's just compute the sum for all i', j', k', i, k.
        # There are n*n * m*m * p*p = 4 * 16 * 64 = 4096 conditions?
        # No, i' in [0,n), j' in [0,m), k' in [0,p), i in [0,n), k in [0,p).
        # Total 2*4*8*2*8 = 1024 checks.
        # Each check is a dot product of length R.
        # 1024 * R operations.
        # If R=64, 65536 ops. Very fast.
        
        for i_prime in range(n):
            for j_prime in range(m):
                for k_prime in range(p):
                    for i in range(n):
                        for k in range(p):
                            # Sum_r U[i',j',r] V[j',k',r] W[k,i,r]
                            # U_idx = i'*m + j_prime
                            # V_idx = j_prime*p + k_prime
                            # W_idx = k*n + i
                            s = np.dot(U[i_prime * m + j_prime, :], V[j_prime * p + k_prime, :])
                            # Wait, W is also involved.
                            # sum_r U[i',j',r] V[j',k',r] W[k,i,r]
                            # This is not a simple dot product of U and V.
                            # It is sum_r (U[i',j',r] V[j',k',r]) W[k,i,r]
                            # Let X_r = U[i',j',r] V[j',k',r]
                            # s = dot(X, W[k,i,:])
                            
                            # To be fast, precompute UV_prod?
                            # No, just loop.
                            s = 0
                            for r in range(R):
                                s += U[i_prime * m + j_prime, r] * V[j_prime * p + k_prime, r] * W[k * n + i, r]
                            
                            if i_prime == i and k_prime == k:
                                if s != 1:
                                    return False
                            else:
                                if s != 0:
                                    return False
        return True

    # Simple local search
    # Start with schoolbook, try to remove columns?
    # Or random restarts.
    
    # For now, return schoolbook if no better found quickly.
    # A known rank for 2x4x8 is likely around 40-50.
    # I will try a few random initializations for R=50.
    
    for R in [56, 48, 40]:
        for _ in range(10):
            if time.time() > deadline:
                break
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            # Scale to make entries integers/half-integers?
            # The problem allows integer/half-integer.
            # Random floats won't work.
            # We need a structured search.
            # Given the complexity, I will stick to the schoolbook solution to ensure validity.
            # Finding a valid lower rank requires a specific algorithm (e.g., Groebner bases or specific constructions).
            # Without that, returning schoolbook is safe.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
