import numpy as np
import scipy.linalg
import time
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    # U shape: (n*m, R) = (9, R)
    # V shape: (m*p, R) = (21, R)
    # W shape: (p*n, R) = (21, R)
    # Indexing:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # Scaled by 2: sum_r U2[i*m+j, r] V2[j*p+k, r] W2[k*n+i, r] = 2
    # U2, V2, W2 are integers.
    
    num_eqs = n * m * p
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i, j, k))
                
    # Schoolbook fallback
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
    
    # Kronecker product approach:
    # View A (3x3) as rows of 1x3 vectors, B (3x7) as columns of 3x1 vectors
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Use 3x3 matrix multiplication tensor (rank 9 schoolbook, or rank 7 Strassen)
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Kronecker structure:
    # Let T3 be the 3x3 matmul tensor with rank r3
    # T3[i,j,k] = sum_{r=0}^{r3-1} U3[i,j,r] * V3[j,k,r] * W3[i,k,r]
    #
    # For our problem, we can use:
    # U[i*m+j, r] = U3[i,j,r] * something
    # V[j*p+k, r] = V3[j,k,r] * something  
    # W[k*n+i, r] = W3[i,k,r] * something
    #
    # But we need to handle the p=7 dimension properly.
    # 
    # Alternative: Direct search with structured initialization
    # Try to find low-rank decompositions by exploiting the 3x3 block structure
    
    # First, try Strassen-based construction for rank 61
    # Strassen 2x2 matmul: 7 multiplications
    # For 3x3 * 3x7, we can partition into 2x2 blocks with overlap
    
    # Let's try a different approach: use the known rank-7 3x3 matmul (Strassen)
    # and extend to handle the 7 columns
    
    # Strassen's 7-term formula for 2x2 matmul:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 3x3 * 3x7, partition A into 2x2 blocks:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    # 
    # This is getting complex. Let's try a more direct search with better initialization.
    
    # Strategy: Use random search but with structured constraints
    # Try to find decompositions where U, V, W have sparse patterns
    
    # Better approach: Use integer programming or structured search
    # For rank R, we need 63 equations in 9*R + 21*R + 21*R = 51*R variables
    # For R=61: 3111 variables, 63 equations -> underdetermined
    
    # Let's try a more systematic search:
    # 1. Fix U and V to have small integer entries
    # 2. Solve for W
    # 3. Check if W has integer/half-integer entries
    
    # Try ranks from 61 down to 45
    for rank in range(61, 44, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 5000 if rank <= 55 else 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use structured random initialization
            # Make U and V sparse with small integer entries
            U2 = np.zeros((n*m, rank), dtype=int)
            V2 = np.zeros((m*p, rank), dtype=int)
            
            # Each row of U2 gets 1-2 non-zero entries
            for row in range(n*m):
                num_nonzero = np.random.choice([1, 2], p=[0.7, 0.3])
                cols = np.random.choice(rank, size=num_nonzero, replace=False)
                for c in cols:
                    U2[row, c] = np.random.choice([-2, -1, 1, 2])
            
            # Each row of V2 gets 1-3 non-zero entries
            for row in range(m*p):
                num_nonzero = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
                cols = np.random.choice(rank, size=num_nonzero, replace=False)
                for c in cols:
                    V2[row, c] = np.random.choice([-2, -1, 1, 2])
            
            # Build linear system for W2
            b = np.ones(num_eqs) * 2
            A = np.zeros((num_eqs, p * n * rank))
            
            for e_idx, (i, j, k) in enumerate(eq_indices):
                u_idx = i * m + j
                v_idx = j * p + k
                w_row_base = (k * n + i) * rank
                
                coeffs = U2[u_idx, :] * V2[v_idx, :]
                nz = np.nonzero(coeffs)[0]
                if len(nz) > 0:
                    A[e_idx, w_row_base + nz] = coeffs[nz]
            
            # Solve least squares
            try:
                W2_flat, residuals, rank_A, s = scipy.linalg.lstsq(A, b)
            except Exception:
                continue
            
            # Check if W2 is close to half-integers
            W2_scaled = W2_flat * 2
            W2_rounded = np.round(W2_scaled)
            error = np.max(np.abs(W2_scaled - W2_rounded))
            
            if error < 1e-5:
                # Verify exactly
                W2 = (W2_rounded / 2).reshape((p*n, rank))
                valid = True
                for e_idx, (i, j, k) in enumerate(eq_indices):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s_val = np.dot(U2[u_idx, :], V2[v_idx, :] * W2[w_idx, :])
                    if abs(s_val - 1.0) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    U = U2.astype(float)
                    V = V2.astype(float)
                    result = {'U': U.tolist(), 'V': V.tolist(), 'W': W2.tolist()}
                    return result
    
    return best_result
