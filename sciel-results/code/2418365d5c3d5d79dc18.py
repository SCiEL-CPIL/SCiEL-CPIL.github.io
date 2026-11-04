import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s margin
    
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

    # Known lower bound is roughly n^2 or similar, but for 3x3x4, 20 is best known.
    # We will search from 20 upwards.
    
    for R in range(20, rank_school):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for trial in range(100):
            if time.time() > deadline:
                break
                
            # Random init with small integers/half-integers
            # Use a mix of -1, 0, 1, 0.5, -0.5, 1.5, -1.5
            # Or just random uniform in [-2, 2] and round later?
            # Better: Random integers in [-2, 2]
            U = np.random.randint(-2, 3, (n * m, R)).astype(float)
            V = np.random.randint(-2, 3, (m * p, R)).astype(float)
            W = np.random.randint(-2, 3, (p * n, R)).astype(float)
            
            # Iterative refinement (ALS) in floating point
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for W
                W_new = np.zeros_like(W)
                for i in range(n):
                    for k in range(p):
                        A = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        # Solve A * x = 1
                        try:
                            x, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W_new[k * n + i, :] = x
                        except:
                            pass
                W = W_new
                
                # Solve for V
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
                V = V_new
                
                # Solve for U
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
                U = U_new
                
                # Check if we are close to a half-integer solution
                # Calculate residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_val = 0.0
                            for r in range(R):
                                s_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            residual += (s_val - 1.0) ** 2
                
                if residual < 1e-10:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s_val = 0.0
                                for r in range(R):
                                    s_val += U_snap[i * m + j, r] * V_snap[j * p + k, r] * W_snap[k * n + i, r]
                                if abs(s_val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

    return best_result
