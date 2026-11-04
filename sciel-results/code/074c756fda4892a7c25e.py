import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s margin
    
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find lower rank
    # R must be at least n*p? No, lower bound is not strictly n*p.
    # But R must be at least max(n*m, m*p, p*n)? No.
    # R must be at least 1.
    # Let's try from 15 to 35.
    
    for R in range(15, rank_school):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
                
            # Random init with small integers
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            # Iterative refinement
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Solve for W
                # For each i, k:
                # A (3 x R) * W[k*n+i, :] (R) = 1
                # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                W_new = np.zeros_like(W)
                for i in range(n):
                    for k in range(p):
                        A = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        # Solve A * x = 1
                        # If A is singular or underdetermined, use lstsq
                        try:
                            x, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W_new[k * n + i, :] = x
                        except:
                            pass
                
                # Round W
                W = np.round(W_new * 2) / 2
                
                # Solve for V
                # For each j, k:
                # Equations for each i?
                # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                # For fixed j, k, unknowns are V[j*p+k, :]
                # Equations: for each i, Sum_r (U[i*m+j, r] * W[k*n+i, r]) * V[j*p+k, r] = 1
                # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # B (n x R) * V[j*p+k, :] (R) = 1
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, R))
                        for i in range(n):
                            for r in range(R):
                                B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        try:
                            x, residuals, rank_b, s = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V_new[j * p + k, :] = x
                        except:
                            pass
                V = np.round(V_new * 2) / 2
                
                # Solve for U
                # For each i, j:
                # Unknowns U[i*m+j, :]
                # Equations: for each k, Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                # Let C[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # C (p x R) * U[i*m+j, :] (R) = 1
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        C = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                C[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        try:
                            x, residuals, rank_c, s = np.linalg.lstsq(C, np.ones(p), rcond=None)
                            U_new[i * m + j, :] = x
                        except:
                            pass
                U = np.round(U_new * 2) / 2
                
                # Check exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_val = 0.0
                            for r in range(R):
                                s_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            if abs(s_val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
