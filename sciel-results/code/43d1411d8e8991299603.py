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

    # Strategy: Search for decomposition of M_{3,3,2} with low rank R2.
    # Then construct M_{3,3,4} with rank 2*R2.
    # M_{3,3,2} dimensions: n=3, m=3, p=2.
    n2, m2, p2 = 3, 3, 2
    
    # Try ranks for M_{3,3,2} from 12 to 18
    found_decomp = None
    for R2 in range(12, 19):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations
        for trial in range(200):
            if time.time() > deadline - 5:
                break
                
            # Random init with small integers/half-integers
            U2 = np.random.randint(-2, 3, (n2 * m2, R2)).astype(float)
            V2 = np.random.randint(-2, 3, (m2 * p2, R2)).astype(float)
            W2 = np.random.randint(-2, 3, (p2 * n2, R2)).astype(float)
            
            # Iterative refinement (ALS) in floating point
            success = False
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W2
                W2_new = np.zeros_like(W2)
                for i in range(n2):
                    for k in range(p2):
                        A = np.zeros((m2, R2))
                        for j in range(m2):
                            for r in range(R2):
                                A[j, r] = U2[i * m2 + j, r] * V2[j * p2 + k, r]
                        try:
                            x, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(m2), rcond=None)
                            W2_new[k * n2 + i, :] = x
                        except:
                            pass
                W2 = W2_new
                
                # Solve for V2
                V2_new = np.zeros_like(V2)
                for j in range(m2):
                    for k in range(p2):
                        B = np.zeros((n2, R2))
                        for i in range(n2):
                            for r in range(R2):
                                B[i, r] = U2[i * m2 + j, r] * W2[k * n2 + i, r]
                        try:
                            x, residuals, rank_b, s = np.linalg.lstsq(B, np.ones(n2), rcond=None)
                            V2_new[j * p2 + k, :] = x
                        except:
                            pass
                V2 = V2_new
                
                # Solve for U2
                U2_new = np.zeros_like(U2)
                for i in range(n2):
                    for j in range(m2):
                        C = np.zeros((p2, R2))
                        for k in range(p2):
                            for r in range(R2):
                                C[k, r] = V2[j * p2 + k, r] * W2[k * n2 + i, r]
                        try:
                            x, residuals, rank_c, s = np.linalg.lstsq(C, np.ones(p2), rcond=None)
                            U2_new[i * m2 + j, :] = x
                        except:
                            pass
                U2 = U2_new
                
                # Check residual
                residual = 0.0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            s_val = 0.0
                            for r in range(R2):
                                s_val += U2[i * m2 + j, r] * V2[j * p2 + k, r] * W2[k * n2 + i, r]
                            residual += (s_val - 1.0) ** 2
                
                if residual < 1e-10:
                    # Snap to half-integers
                    U2_snap = np.round(U2 * 2) / 2
                    V2_snap = np.round(V2 * 2) / 2
                    W2_snap = np.round(W2 * 2) / 2
                    
                    # Verify exactness
                    valid = True
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                s_val = 0.0
                                for r in range(R2):
                                    s_val += U2_snap[i * m2 + j, r] * V2_snap[j * p2 + k, r] * W2_snap[k * n2 + i, r]
                                if abs(s_val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        found_decomp = (U2_snap, V2_snap, W2_snap, R2)
                        success = True
                        break
            
            if success:
                break
        
        if found_decomp:
            break

    if found_decomp:
        U2, V2, W2, R2 = found_decomp
        R_total = 2 * R2
        
        # Construct U, V, W for 3x3x4
        U = np.zeros((n * m, R_total))
        V = np.zeros((m * p, R_total))
        W = np.zeros((p * n, R_total))
        
        # First half: columns 0, 1
        U[:, :R2] = U2
        for j in range(m):
            for k in range(2):
                V[j * p + k, :R2] = V2[j * p2 + k]
        for k in range(2):
            for i in range(n):
                W[k * n + i, :R2] = W2[k * n2 + i]
        
        # Second half: columns 2, 3
        U[:, R2:] = U2
        for j in range(m):
            for k in range(2, 4):
                V[j * p + k, R2:] = V2[j * p2 + (k - 2)]
        for k in range(2, 4):
            for i in range(n):
                W[k * n + i, R2:] = W2[(k - 2) * n2 + i]
        
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
