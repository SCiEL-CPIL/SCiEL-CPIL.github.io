"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    n, m, p = 2, 2, 13
    
    # Target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_target[i * m + k, k * p + j, j * n + i] = 1.0

    def get_strassen_fallback():
        # Verified Strassen decomposition for 2x2 * 2x2:
        strassen_U = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1],
        ]
        strassen_V = [
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, -1, 0, 1, 0],
            [1, 0, -1, 0, 1, 0, 1],
        ]
        strassen_W = [
            [1, 0, 0, 1, -1, 0, 1],
            [1, -1, 1, 0, 0, 1, 0],
            [0, 1, -1, 1, 1, 0, 0],
            [1, 1, 0, -1, 0, -1, 0],
        ]
        
        # Verify Strassen decomposition numerically
        S_target = np.zeros((4, 4, 4), dtype=float)
        S_target[0, 0, 0] = 1
        S_target[0, 1, 2] = 1
        S_target[1, 2, 0] = 1
        S_target[1, 3, 2] = 1
        S_target[2, 0, 1] = 1
        S_target[2, 1, 3] = 1
        S_target[3, 2, 1] = 1
        S_target[3, 3, 3] = 1
        
        S_recon = np.zeros((4, 4, 4), dtype=float)
        for t in range(7):
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        S_recon[a, b, c] += strassen_U[a][t] * strassen_V[b][t] * strassen_W[c][t]
        
        if not np.allclose(S_recon, S_target, atol=1e-10):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U[i * m + k][column] = 1
                        V[k * p + j][column] = 1
                        W[j * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
        
        # Build full decomposition
        # 6 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
        # 1 schoolbook block for column 12
        total_rank = 6 * 7 + 4  # = 46
        
        U_full = [[0] * total_rank for _ in range(n * m)]  # (4, 46)
        V_full = [[0] * total_rank for _ in range(m * p)]  # (26, 46)
        W_full = [[0] * total_rank for _ in range(p * n)]  # (26, 46)
        
        col = 0
        # 6 Strassen blocks
        for block in range(6):
            j0 = 2 * block
            j1 = 2 * block + 1
            for t in range(7):
                # U: same for all blocks (operates on A entries)
                for a in range(4):
                    U_full[a][col] = strassen_U[a][t]
                # V: map local (k, j_local) to global (k, j)
                # local V index: k*2 + j_local where j_local 0->j0, 1->j1
                for k in range(2):
                    for jl in range(2):
                        j_global = j0 + jl
                        v_local = k * 2 + jl
                        v_global = k * p + j_global
                        V_full[v_global][col] = strassen_V[v_local][t]
                # W: map local (j_local, i) to global (j, i)
                for jl in range(2):
                    for i in range(2):
                        j_global = j0 + jl
                        w_local = jl * 2 + i
                        w_global = j_global * n + i
                        W_full[w_global][col] = strassen_W[w_local][t]
                col += 1
        
        # 1 schoolbook block for column 12
        j = 12
        for i in range(n):
            for k in range(m):
                U_full[i * m + k][col] = 1
                V_full[k * p + j][col] = 1
                W_full[j * n + i][col] = 1
                col += 1
        
        # Verify full decomposition
        T_recon = np.zeros((n * m, m * p, p * n), dtype=float)
        for t in range(total_rank):
            for a in range(n * m):
                ua = U_full[a][t]
                if ua == 0:
                    continue
                for b in range(m * p):
                    ub = ua * V_full[b][t]
                    if ub == 0:
                        continue
                    for c in range(p * n):
                        T_recon[a, b, c] += ub * W_full[c][t]
        
        if not np.allclose(T_recon, T_target, atol=1e-10):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U[i * m + k][column] = 1
                        V[k * p + j][column] = 1
                        W[j * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
        
        return {"U": U_full, "V": V_full, "W": W_full}

    # Attempt to find a lower rank decomposition via random search
    # We try ranks 45, 44, 43
    for R in [45, 44, 43]:
        if time.time() > deadline - 30: # Need time for verification and return
            break
            
        # Try multiple restarts
        for restart in range(5):
            if time.time() > deadline - 30:
                break
                
            random.seed(42 + restart + R)
            # Initialize with small random integers
            U = np.random.randint(-1, 2, (n*m, R)).astype(float)
            V = np.random.randint(-1, 2, (m*p, R)).astype(float)
            W = np.random.randint(-1, 2, (p*n, R)).astype(float)
            
            # Gradient Descent
            eta = 0.05
            for step in range(100):
                if time.time() > deadline - 30:
                    break
                    
                # Compute residual
                T_recon = np.einsum('it,jt,kt->ijk', U, V, W)
                Res = T_recon - T_target
                
                # Gradients
                grad_U = np.einsum('ijk,jt,kt->it', Res, V, W)
                grad_V = np.einsum('ijk,it,kt->jt', Res, U, W)
                grad_W = np.einsum('ijk,it,jt->kt', Res, U, V)
                
                # Update
                U -= eta * grad_U
                V -= eta * grad_V
                W -= eta * grad_W
                
                # Check error
                err = np.linalg.norm(Res)
                if err < 1e-4:
                    # Try rounding to half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify
                    T_check = np.einsum('it,jt,kt->ijk', U_int, V_int, W_int)
                    if np.allclose(T_check, T_target, atol=1e-9):
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                    # If rounding failed, we can try a few more steps with smaller eta
                    # but for now, break to try next restart
                    break

    # If no better solution found, return Strassen 46
    return get_strassen_fallback()
# EVOLVE-BLOCK-END
