import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Fallback schoolbook decomposition
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
        
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    start_time = time.time()
    timeout = 100  # seconds
    
    # Search for lower rank
    # Try ranks from 20 up to best_rank - 1
    for R in range(20, best_rank):
        if time.time() - start_time > timeout:
            break
            
        for attempt in range(100): # 100 random attempts per rank
            if time.time() - start_time > timeout:
                break
                
            # Random U and V with small integers
            U_rand = np.random.randint(-1, 2, (n * m, R))
            V_rand = np.random.randint(-1, 2, (m * p, R))
            
            # Construct A matrix
            # A[:, r] = flatten( outer(U_rand[:, r], V_rand[:, r]) )
            A = np.zeros((n * m * m * p, R))
            for r in range(R):
                M_r = np.outer(U_rand[:, r], V_rand[:, r])
                A[:, r] = M_r.flatten()
                
            # Check rank of A
            rank_A = np.linalg.matrix_rank(A)
            if rank_A < R:
                continue
                
            # Solve for W
            W_sol = np.zeros((p * n, R))
            valid = True
            for k_prime in range(p):
                for i_prime in range(n):
                    # Construct b
                    b = np.zeros(n * m * m * p)
                    for i in range(n):
                        if i != i_prime: continue
                        for j in range(m):
                            for j_prime in range(m):
                                if j != j_prime: continue
                                for k in range(p):
                                    if k != k_prime: continue
                                    idx = (i * m + j) * (m * p) + (j_prime * p + k)
                                    b[idx] = 1.0
                                    
                    # Solve A x = b
                    try:
                        x, residuals, rank_x, s = np.linalg.lstsq(A, b, rcond=None)
                        if residuals and residuals[0] > 1e-9:
                            valid = False
                            break
                        # Check if x is integer or half-integer
                        x_scaled = x * 2
                        if not np.allclose(x_scaled, np.round(x_scaled), atol=1e-9):
                            valid = False
                            break
                        W_sol[k_prime * n + i_prime] = x
                    except Exception:
                        valid = False
                        break
                        
            if valid:
                # Convert to list of lists
                U_list = [[int(val) for val in row] for row in U_rand]
                V_list = [[int(val) for val in row] for row in V_rand]
                W_list = [[int(val * 2) / 2 for val in row] for row in W_sol]
                
                # Verify exactness with a simple check
                # Sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i', r] == delta
                # We trust the residual check for now.
                
                return {"U": U_list, "V": V_list, "W": W_list}

    return {"U": best_U, "V": best_V, "W": best_W}
