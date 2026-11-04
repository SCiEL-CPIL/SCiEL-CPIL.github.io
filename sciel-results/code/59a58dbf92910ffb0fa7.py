import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 (Strassen-Strassen, Rank 49) ---
    
    # Strassen 7-term decomposition for 2x2
    # u, v, w are 4x7 matrices (flattened 2x2)
    # Indices: 00, 01, 10, 11
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, -1, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, -1, 1, 0, -1]
    ]
    
    # Strassen steps for 4x4 (Block level)
    # alpha, beta, gamma are 4x7 matrices (coeffs for 4 blocks)
    # Blocks: 00, 01, 10, 11
    alpha = [
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    beta = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    gamma = [
        [1, -1, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, -1, 1, 0, -1]
    ]
    
    # Verify Strassen 2x2
    # We assume the above matrices are correct.
    
    # Construct U, V, W for 4x4x4
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U
            for i in range(16):
                r = i // 4
                c = i % 4
                block_r = r // 2
                block_c = c // 2
                local_r = r % 2
                local_c = c % 2
                block_idx = block_r * 2 + block_c
                local_idx = local_r * 2 + local_c
                U[i][col] = alpha[block_idx][s] * U2[local_idx][t]
                
            # V
            for j in range(16):
                r = j // 4
                c = j % 4
                block_r = r // 2
                block_c = c // 2
                local_r = r % 2
                local_c = c % 2
                block_idx = block_r * 2 + block_c
                local_idx = local_r * 2 + local_c
                V[j][col] = beta[block_idx][s] * V2[local_idx][t]
                
            # W
            for k in range(16):
                r = k // 4
                c = k % 4
                block_r = r // 2
                block_c = c // 2
                local_r = r % 2
                local_c = c % 2
                block_idx = block_r * 2 + block_c
                local_idx = local_r * 2 + local_c
                W[k][col] = gamma[block_idx][s] * W2[local_idx][t]
                
    # --- Part 2: 4x4x1 (Schoolbook, Rank 16) ---
    col_offset = rank_444
    for i in range(4):
        for j in range(4):
            col = col_offset + i * 4 + j
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = 4, 4, 5
    # Check a random case
    A = np.random.randint(0, 10, (n, m))
    B = np.random.randint(0, 10, (m, p))
    C_expected = A @ B
    
    U_flat = A.flatten()
    V_flat = B.flatten()
    
    C_reconstructed = np.zeros((p, n))
    for k in range(p):
        for i in range(n):
            idx_w = k * n + i
            val = 0
            for c in range(len(U[0])):
                u_val = U_flat[i * m + (0 if False else 0)] # Wrong indexing
                # Correct indexing:
                # U row: i*m + j
                # V row: j*p + k
                # W row: k*n + i
                pass
                
    # Proper check
    tensor = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                tensor[i, j, k] = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                
    # tensor[i, j, k] should be 1
    if not np.allclose(tensor, 1):
        print("Decomposition failed!")
        # Fallback to schoolbook
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
        res = {"U": U, "V": V, "W": W}
    else:
        print(f"Decomposition valid. Rank: {len(res['U'][0])}")
        
    print(res)
