import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Strategy: Use known 3x3x3 rank-76 decomposition as building block
    # Then extend to 3x4x7 by adding terms for extra dimensions
    
    # The 3x3x3 tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # has a known rank-76 decomposition (improvement over schoolbook 27)
    
    # For 3x4x7, we decompose:
    # A is 3x4, B is 4x7, C is 3x7
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can write j = j0 (0,1,2) + j_extra (3)
    # k = k0 (0,1,2) + k_extra (4,5,6)
    
    # C[i,k] = sum_{j0=0}^{2} sum_{k0=0}^{2} A[i,j0] B[j0,k0] * delta(k0,k) * delta(j0,j)
    #         + sum_{j0=0}^{2} A[i,j0] B[j0,k_extra] for k_extra in {4,5,6}
    #         + sum_{k0=0}^{2} A[i,3] B[3,k0] * delta(k0,k)
    #         + A[i,3] B[3,k_extra] for k_extra in {4,5,6}
    
    # This is complex. Let me use a different approach:
    # Use the rank-76 3x3x3 decomposition for the core, then add 12 terms
    # for the extra dimensions (4-1=3 extra cols in A, 7-3=4 extra rows in B,
    # but we need to handle the cross terms)
    
    # Actually, let me just try to find a low-rank decomposition using
    # a structured initialization based on the Kronecker product idea.
    
    # Alternative: Use the fact that 3x4x7 can be seen as a sum of
    # rank-1 tensors. Let me try a deterministic construction.
    
    # For now, let's use a smarter random search with structure:
    # Initialize U and V with patterns that exploit the block structure.
    
    # Pre-compute equation indices
    num_eqs = n * n * p * p * m * m
    
    I = np.zeros(num_eqs, dtype=int)
    K = np.zeros(num_eqs, dtype=int)
    I2 = np.zeros(num_eqs, dtype=int)
    K2 = np.zeros(num_eqs, dtype=int)
    J = np.zeros(num_eqs, dtype=int)
    J2 = np.zeros(num_eqs, dtype=int)
    
    idx = 0
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for k2 in range(p):
                    for j in range(m):
                        for j2 in range(m):
                            I[idx] = i
                            K[idx] = k
                            I2[idx] = i2
                            K2[idx] = k2
                            J[idx] = j
                            J2[idx] = j2
                            idx += 1
                            
    b = np.zeros(num_eqs)
    mask = (I2 == I) & (K2 == K) & (J == J2)
    b[mask] = 1.0
    
    # Try ranks from a lower bound upward
    # Theoretical lower bound for 3x4x7 is at least max(3*4, 4*7, 3*7) = 28
    # But realistically, we expect something around 60-75
    
    # Use structured random search: initialize with block patterns
    for R in range(60, 84):
        if time.time() > deadline - 5.0:
            break
            
        max_trials = 30
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            # Structured initialization: use block-diagonal patterns
            # U has shape (12, R), V has shape (28, R)
            
            # Try different random seeds with structure
            np.random.seed(trial * 12345 + R)
            
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            
            # Build A matrix
            U_vals = U[I2*m + J, :]  # shape (num_eqs, R)
            V_vals = V[J2*p + K2, :]  # shape (num_eqs, R)
            Prod = U_vals * V_vals    # shape (num_eqs, R)
            
            A = np.zeros((num_eqs, (p*n)*R))
            
            r_idx = np.arange(R)
            base_cols = K * n + I
            Cols = base_cols[:, None] + r_idx[None, :] * (p*n)
            
            rows_flat = np.repeat(np.arange(num_eqs), R)
            cols_flat = Cols.flatten()
            vals_flat = Prod.flatten()
            
            A[rows_flat, cols_flat] = vals_flat
            
            try:
                w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
                
            w2 = w * 2
            w2_rounded = np.round(w2)
            if np.allclose(w2, w2_rounded, atol=1e-9):
                W_sol = w.reshape((p*n, R))
                
                if R < best_rank:
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W_sol
                    found = True
                    break
        
        if found:
            # Found a solution at rank R, continue to try lower ranks
            # But we're going upward, so break if we find a good one
            # Actually, we want minimal rank, so keep searching lower
            pass
    
    # If we found something better than schoolbook, return it
    # Otherwise return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
