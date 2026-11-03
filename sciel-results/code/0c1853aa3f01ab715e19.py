import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 140
    
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # For a given rank r, try to find U, V, W
    # Fix U and V randomly, solve for W via linear system
    for r in range(13, 25):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(100):
            if time.time() > deadline:
                break
            
            np.random.seed(trial)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            
            # Build linear system for W
            # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is a linear system in W with n*p = 8 unknowns
            A = np.zeros((n*p, n*p))
            b = np.zeros(n*p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = k * n + i
                        for c in range(r):
                            A[row, k * n + i] += U[i*m+j, c] * V[j*p+k, c]
                        b[row] = 1.0
            
            # Wait, that's not right. Let me reconsider.
            # W has shape (p*n, r) = (8, r)
            # W[k*n+i, c] for c in range(r)
            # So for fixed (i,j,k), the equation is:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is linear in the vector W[k*n+i, :] which has r components.
            # But different (i,j,k) may map to the same row k*n+i.
            
            # Let me group by the W row index
            # For a given row w_idx = k*n+i, the equation involves W[w_idx, :]
            # Multiple (i,j,k) can give the same w_idx
            
            # Actually, each (i,k) pair uniquely determines w_idx = k*n+i
            # For fixed (i,k), j varies from 0 to m-1
            # So for each (i,k), we have m equations involving W[k*n+i, :]
            
            # Let me restructure: for each (i,k), we have m equations:
            # for j in range(m): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is a system of m equations in r unknowns (W[k*n+i, :])
            
            # If m <= r, we can solve. If m > r, it's overdetermined.
            # Here m=3, so if r >= 3, we can potentially solve.
            
            W = np.zeros((p*n, r))
            feasible = True
            for i in range(n):
                for k in range(p):
                    # Build system for W[k*n+i, :]
                    # m equations in r unknowns
                    A_sub = np.zeros((m, r))
                    b_sub = np.zeros(m)
                    for j in range(m):
                        for c in range(r):
                            A_sub[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        b_sub[j] = 1.0
                    
                    # Solve least squares
                    if r >= m:
                        W_vec, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                        # Check if solution is exact
                        if np.max(np.abs(A_sub @ W_vec - b_sub)) > 1e-9:
                            feasible = False
                            break
                        W[k*n+i, :] = W_vec
                    else:
                        # Underdetermined, use minimum norm solution
                        W_vec, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                        if np.max(np.abs(A_sub @ W_vec - b_sub)) > 1e-9:
                            feasible = False
                            break
                        W[k*n+i, :] = W_vec
                    
                    if not feasible:
                        break
                if not feasible:
                    break
            
            if feasible:
                if verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

