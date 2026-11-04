import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    T = np.zeros((8, 16, 8), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1

    # Current known solution rank 28
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('r,s,t->rst', u, v, w)
            
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    best_rank = rank_known
    best_U, best_V, best_W = U_known, V_known, W_known
    
    start_time = time.time()
    deadline = start_time + 110

    def local_search_simulated_annealing(U_init, V_init, W_init, target_rank):
        R = target_rank
        U = [row[:] for row in U_init]
        V = [row[:] for row in V_init]
        W = [row[:] for row in W_init]
        
        # Zero out extra columns
        for c in range(target_rank, rank_known):
            for i in range(8): U[i][c] = 0
            for i in range(16): V[i][c] = 0
            for i in range(8): W[i][c] = 0
            
        # Precompute M_r = outer(V[:, r], W[:, r])
        M = [None] * R
        for r in range(R):
            M[r] = np.outer(np.array(V[:, r]), np.array(W[:, r]))
            
        # Compute initial residual
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(R):
            T_est += np.einsum('r,s,t->rst', np.array(U[:, r]), np.array(V[:, r]), np.array(W[:, r]))
        Residual = T - T_est
        current_cost = np.sum(Residual**2)
        
        best_cost = current_cost
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        
        temp = 1.0
        cooling = 0.9999
        min_temp = 0.01
        
        while time.time() < deadline and temp > min_temp:
            mat = random.choice(['U', 'V', 'W'])
            r = random.randint(0, R - 1)
            delta = random.choice([-1, 1])
            
            if mat == 'U':
                i = random.randint(0, 7)
                dot_val = np.dot(Residual[i].flatten(), M[r].flatten())
                norm_M2 = np.sum(M[r]**2)
                delta_cost = -2 * delta * dot_val + delta * delta * norm_M2
                new_cost = current_cost + delta_cost
                
                if new_cost < current_cost or (delta_cost > 0 and random.random() < np.exp(-delta_cost / temp)):
                    current_cost = new_cost
                    U[i][r] += delta
                    Residual[i] -= delta * M[r]
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_cost < 1e-9:
                            return best_U, best_V, best_W
            elif mat == 'V':
                j = random.randint(0, 15)
                N_r = np.outer(np.array(U[:, r]), np.array(W[:, r]))
                dot_val = np.dot(Residual[:, j, :].flatten(), N_r.flatten())
                norm_N2 = np.sum(N_r**2)
                delta_cost = -2 * delta * dot_val + delta * delta * norm_N2
                new_cost = current_cost + delta_cost
                
                if new_cost < current_cost or (delta_cost > 0 and random.random() < np.exp(-delta_cost / temp)):
                    current_cost = new_cost
                    V[j][r] += delta
                    Residual[:, j, :] -= delta * N_r
                    M[r][j, :] += delta * np.array(W[:, r])
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_cost < 1e-9:
                            return best_U, best_V, best_W
            else:
                k = random.randint(0, 7)
                N_r = np.outer(np.array(U[:, r]), np.array(V[:, r]))
                dot_val = np.dot(Residual[:, :, k].flatten(), N_r.flatten())
                norm_N2 = np.sum(N_r**2)
                delta_cost = -2 * delta * dot_val + delta * delta * norm_N2
                new_cost = current_cost + delta_cost
                
                if new_cost < current_cost or (delta_cost > 0 and random.random() < np.exp(-delta_cost / temp)):
                    current_cost = new_cost
                    W[k][r] += delta
                    Residual[:, :, k] -= delta * N_r
                    M[r][:, k] += delta * np.array(V[:, r])
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_cost < 1e-9:
                            return best_U, best_V, best_W
            
            temp *= cooling

        return best_U, best_V, best_W

    if time.time() < deadline - 10:
        for restart in range(20):
            if time.time() > deadline - 5: break
            U_res, V_res, W_res = local_search_simulated_annealing(U_known, V_known, W_known, 27)
            if check_solution(U_res, V_res, W_res, 27):
                best_rank = 27
                best_U, best_V, best_W = U_res, V_res, W_res
                break

    if best_rank == 28 and time.time() < deadline - 10:
        for restart in range(10):
            if time.time() > deadline - 5: break
            U_res, V_res, W_res = local_search_simulated_annealing(U_known, V_known, W_known, 26)
            if check_solution(U_res, V_res, W_res, 26):
                best_rank = 26
                best_U, best_V, best_W = U_res, V_res, W_res
                break

    return {"U": best_U, "V": best_V, "W": best_W}
