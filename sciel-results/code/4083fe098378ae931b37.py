import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better decomposition for 3x3x3 part
    # Dimensions for 3x3x3: n=3, m=3, p=3
    n3, m3, p3 = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # We will search for rank r for 3x3x3
    # If found, total rank is r + 9 (for the 3x3x1 part)
    best_rank = rank_school
    best_decomp = fallback
    
    # Search ranks from 23 upwards
    for r in range(23, 30):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random initializations
        # We use a heuristic search
        # We fix U and V randomly, solve for W
        # If W is valid (halves/ints) and identity holds, we are done
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Generate random U, V with small integers
            # Values in {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, (n3 * m3, r))
            V_rand = np.random.randint(-1, 2, (m3 * p3, r))
            
            # Solve for W
            # W has shape (p3 * n3, r)
            W_sol = np.zeros((p3 * n3, r))
            valid = True
            
            for k in range(p3):
                for i in range(n3):
                    # Equation: sum_l U[i*m3+j, l] * V[j*p3+k, l] * W[k*n3+i, l] = 1 for all j
                    # Let w_l = W[k*n3+i, l]
                    # sum_l (U[i*m3+j, l] * V[j*p3+k, l]) * w_l = 1
                    # Matrix A of size m3 x r
                    A = np.zeros((m3, r))
                    for j in range(m3):
                        for l in range(r):
                            A[j, l] = U_rand[i * m3 + j, l] * V_rand[j * p3 + k, l]
                    
                    # Solve A * w = 1
                    # Use least squares or exact solve if possible
                    # Since we need exact halves/ints, we check consistency
                    try:
                        # If rank(A) < m3, check if 1 is in col space
                        # Use np.linalg.lstsq
                        w, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(m3), rcond=None)
                        
                        # Check if residual is close to 0
                        if np.linalg.norm(A @ w - np.ones(m3)) > 1e-9:
                            valid = False
                            break
                            
                        # Check if w entries are halves/ints
                        # w * 2 should be integer
                        w_check = w * 2
                        if not np.allclose(w_check, np.round(w_check), atol=1e-9):
                            valid = False
                            break
                            
                        W_sol[k * n3 + i, :] = w
                    except Exception:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Verify full identity
            # sum_l U[i*m3+j, l] * V[j*p3+k, l] * W[k*n3+i, l] == 1
            is_correct = True
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        s = 0
                        for l in range(r):
                            s += U_rand[i * m3 + j, l] * V_rand[j * p3 + k, l] * W_sol[k * n3 + i, l]
                        if abs(s - 1) > 1e-9:
                            is_correct = False
                            break
                    if not is_correct: break
                if not is_correct: break
                
            if is_correct:
                # Construct full 3x3x4 decomposition
                total_rank = r + 9
                U_full = np.zeros((n * m, total_rank))
                V_full = np.zeros((m * p, total_rank))
                W_full = np.zeros((p * n, total_rank))
                
                # Copy 3x3x3 part
                U_full[:9, :r] = U_rand
                V_full[:9, :r] = V_rand
                W_full[:9, :r] = W_sol # Note: W_sol indices are k*n3+i, which is k*3+i. In full W, indices are k*n+i = k*3+i. Matches.
                
                # Add 3x3x1 part (k=3)
                # Schoolbook for this part
                col_offset = r
                for i in range(n):
                    for j in range(m):
                        # U[i*m+j, col] = 1
                        # V[j*p+3, col] = 1
                        # W[3*n+i, col] = 1
                        U_full[i * m + j, col_offset] = 1
                        V_full[j * p + 3, col_offset] = 1
                        W_full[3 * n + i, col_offset] = 1
                        col_offset += 1
                        
                # Convert to lists of floats/ints
                # Ensure they are exactly half-integers
                U_list = [[float(x) for x in row] for row in U_full]
                V_list = [[float(x) for x in row] for row in V_full]
                W_list = [[float(x) for x in row] for row in W_full]
                
                return {"U": U_list, "V": V_list, "W": W_list}
            
    return fallback
