import numpy as np
import scipy.linalg
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # Target: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # Scaled: sum_c U2[i*m+j, c] V2[j*p+k, c] W2[k*n+i, c] = 8
    # U2, V2, W2 are integers.
    
    # Pre-compute indices for the equations
    # Equation index e corresponds to (i, j, k)
    # e = i * m * p + j * p + k
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i, j, k))
                
    num_eqs = n * m * p
    # Unknowns for W2 are indexed by (k, i, c)
    # Unknown index u = k * n * r + i * r + c
    
    # We will try ranks from 62 down
    # Lower bound estimate: 
    # For 3x3x3, rank 23.
    # For 3x3x7, rank should be > 23.
    # Let's start from 62 and go down.
    
    best_result = None
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = {'U': U_school.tolist(), 'V': V_school.tolist(), 'W': W_school.tolist()}
    best_rank = rank_school
    
    # Try to improve
    for rank in range(rank_school - 1, 20, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for U2, V2
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random small integers for U2, V2
            # Range -2 to 2
            U2 = np.random.randint(-2, 3, size=(n*m, rank))
            V2 = np.random.randint(-2, 3, size=(m*p, rank))
            
            # Construct the linear system for W2
            # A * W2_flat = b
            # b is all 8s
            b = np.ones(num_eqs) * 8
            
            # A is num_eqs x (p*n*rank)
            # We don't need to build the full dense matrix if it's sparse, 
            # but for small sizes, dense is fine.
            # num_eqs = 63
            # num_unk = p * n * rank = 21 * rank
            # For rank=50, num_unk=1050.
            # 63 x 1050 matrix is small.
            
            A = np.zeros((num_eqs, p * n * rank))
            
            for e_idx, (i, j, k) in enumerate(eq_indices):
                # Equation e_idx: sum_c W2[k*n+i, c] * (U2[i*m+j, c] * V2[j*p+k, c]) = 8
                # Unknown index for W2[k*n+i, c] is (k*n + i) * rank + c
                row_base = k * n + i
                for c in range(rank):
                    coeff = U2[i*m+j, c] * V2[j*p+k, c]
                    if coeff != 0:
                        unk_idx = row_base * rank + c
                        A[e_idx, unk_idx] = coeff
            
            # Solve A * W2_flat = b
            # Use least squares
            try:
                W2_flat, residuals, rank_A, s = scipy.linalg.lstsq(A, b)
            except Exception:
                continue
                
            # Check if W2_flat is close to integers
            W2_rounded = np.round(W2_flat)
            error = np.max(np.abs(W2_flat - W2_rounded))
            
            if error < 1e-6:
                # Verify the solution exactly
                valid = True
                for e_idx, (i, j, k) in enumerate(eq_indices):
                    s_val = 0
                    row_base = k * n + i
                    for c in range(rank):
                        s_val += W2_rounded[row_base * rank + c] * U2[i*m+j, c] * V2[j*p+k, c]
                    if s_val != 8:
                        valid = False
                        break
                if valid:
                    # Found a valid decomposition with this rank
                    # Convert back to U, V, W (divide by 2)
                    U = U2 / 2.0
                    V = V2 / 2.0
                    W = W2_rounded / 2.0
                    
                    # Reshape W
                    W = W.reshape((p*n, rank))
                    
                    result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                    return result
                    
            # If not close to integer, we could try to optimize U2, V2
            # but for now, just try next random init
            
    return best_result
