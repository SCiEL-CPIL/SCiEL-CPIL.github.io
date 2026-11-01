import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback solution (Rank 53)
    # 7 blocks of M_2,2,2 (Rank 7) + 1 block of M_2,2,1 (Rank 4)
    U_fallback = [[0] * 53 for _ in range(n * m)]
    V_fallback = [[0] * 53 for _ in range(m * p)]
    W_fallback = [[0] * 53 for _ in range(p * n)]
    
    # Strassen M_2,2,2
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook M_2,2,1
    U_1 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_1 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_1 = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U_fallback[i * m + j][c] = U_s[i * 2 + j][r]
            for j in range(2):
                V_fallback[j * p + k0][c] = V_s[j * 2 + 0][r]
                V_fallback[j * p + k1][c] = V_s[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W_fallback[kk * n + i][c] = W_s[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U_fallback[i * m + j][c] = U_1[i * 2 + j][r]
        for j in range(2):
            V_fallback[j * p + k_last][c] = V_1[j][r]
        for i in range(2):
            W_fallback[k_last * n + i][c] = W_1[i][r]

    # Try to find Rank 10 solution for M_2,2,3
    # If found, use 5 blocks of M_2,2,3 (Rank 50)
    
    def find_m223_rank10():
        # Tensor T for M_2,2,3
        # i in 0..1, j in 0..1, k in 0..2
        # T[i,j,k] = 1
        # U: 4x10, V: 6x10, W: 6x10
        
        def residuals(params):
            U = params[0:40].reshape(4, 10)
            V = params[40:100].reshape(6, 10)
            W = params[100:160].reshape(6, 10)
            
            res = []
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        # Sum over r
                        s = 0
                        for r in range(10):
                            s += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                        res.append(s - 1.0)
            return res

        best_sol = None
        best_err = float('inf')
        
        # Random restarts
        for _ in range(20):
            if time.time() > deadline - 20:
                break
            # Initialize
            # Try small integers
            init = np.random.randint(-2, 3, size=160)
            # Ensure some structure?
            # Maybe fix first few columns to identity to break symmetry
            # U[0,0]=1, U[1,1]=1, U[2,2]=1, U[3,3]=1 ?
            # This might be too restrictive.
            
            try:
                sol = least_squares(residuals, init, method='lm')
                if sol.success and sol.cost < 1e-8:
                    # Check if close to half-integers
                    U_sol = sol.x[0:40].reshape(4, 10)
                    V_sol = sol.x[40:100].reshape(6, 10)
                    W_sol = sol.x[100:160].reshape(6, 10)
                    
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify snapped solution
                    max_err = 0
                    for i in range(2):
                        for j in range(2):
                            for k in range(3):
                                s = 0
                                for r in range(10):
                                    s += U_snap[i*2+j, r] * V_snap[j*3+k, r] * W_snap[k*2+i, r]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-6:
                        return U_snap, V_snap, W_snap
            except Exception:
                pass
                
        return None

    U_3, V_3, W_3 = find_m223_rank10()
    
    if U_3 is not None:
        # Construct M_2,2,15 using 5 blocks of M_2,2,3
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(5):
            k0 = 3 * block
            for r in range(10):
                c = block * 10 + r
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][c] = U_3[i * 2 + j][r]
                for j in range(2):
                    for k in range(3):
                        kk = k0 + k
                        V[j * p + kk][c] = V_3[j * 3 + k][r]
                for k in range(3):
                    kk = k0 + k
                    for i in range(2):
                        W[kk * n + i][c] = W_3[k * 2 + i][r]
                        
        return {"U": U, "V": V, "W": W}

    # Fallback
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}

