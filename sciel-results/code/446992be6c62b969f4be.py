"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    I_parts = [[0, 1], [2]]
    J_parts = [[0, 1], [2, 3], [4]]
    K_parts = [[0, 1], [2, 3], [4, 5], [6]]
    
    S_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    for i_part in I_parts:
        for j_part in J_parts:
            for k_part in K_parts:
                if time.time() > deadline:
                    break
                ni = len(i_part)
                mj = len(j_part)
                pk = len(k_part)
                
                if ni == 2 and mj == 2 and pk == 2:
                    i0, i1 = i_part
                    j0, j1 = j_part
                    k0, k1 = k_part
                    
                    for s in range(7):
                        u_vec = S_u[s]
                        v_vec = S_v[s]
                        w_vec = S_w[s]
                        
                        U[i0 * m + j0][col] += u_vec[0]
                        U[i0 * m + j1][col] += u_vec[1]
                        U[i1 * m + j0][col] += u_vec[2]
                        U[i1 * m + j1][col] += u_vec[3]
                        
                        V[j0 * p + k0][col] += v_vec[0]
                        V[j0 * p + k1][col] += v_vec[1]
                        V[j1 * p + k0][col] += v_vec[2]
                        V[j1 * p + k1][col] += v_vec[3]
                        
                        W[k0 * n + i0][col] += w_vec[0]
                        W[k0 * n + i1][col] += w_vec[1]
                        W[k1 * n + i0][col] += w_vec[2]
                        W[k1 * n + i1][col] += w_vec[3]
                        
                        col += 1
                else:
                    for i in i_part:
                        for j in j_part:
                            for k in k_part:
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
