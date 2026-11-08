import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    strassen_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    M_blocks = [
        ([0, 3], [0, 3], 0),
        ([2, 3], [0], 2),
        ([0], [1, 3], 1),
        ([3], [2, 0], 2),
        ([0, 1], [3], 3),
        ([2, 0], [0, 1], 0),
        ([1, 3], [2, 3], 0),
    ]
    
    def get_block_indices(block_id, dim2):
        i_off = block_id // 2
        j_off = block_id % 2
        indices = []
        for di in range(2):
            for dj in range(2):
                indices.append((i_off * 2 + di) * dim2 + (j_off * 2 + dj))
        return indices
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_444_block(b_col_offset, c_col_offset):
        nonlocal col
        for a_blocks, b_blocks, c_block in M_blocks:
            for u_222, v_222, w_222 in strassen_222:
                U_local = [0] * 16
                V_local = [0] * 16
                W_local = [0] * 16
                
                for idx, ab in enumerate(a_blocks):
                    coeff = u_222[idx]
                    if coeff != 0:
                        for u_idx in get_block_indices(ab, 4):
                            U_local[u_idx] += coeff
                
                for idx, bb in enumerate(b_blocks):
                    coeff = v_222[idx]
                    if coeff != 0:
                        for v_idx in get_block_indices(bb, 4):
                            V_local[v_idx] += coeff
                
                c_idx = w_222.index(1) if 1 in w_222 else -1
                if c_idx >= 0:
                    for w_idx in get_block_indices(c_block, 4):
                        W_local[w_idx] += coeff
                
                for i in range(16):
                    if U_local[i] != 0:
                        U[i][col] = U_local[i]
                
                for i in range(16):
                    if V_local[i] != 0:
                        row = i // 4
                        col_l = i % 4
                        V[row * p + b_col_offset + col_l][col] = V_local[i]
                
                for i in range(16):
                    if W_local[i] != 0:
                        k = i // 4
                        i_l = i % 4
                        W[(c_col_offset + k) * n + i_l][col] = W_local[i]
                
                col += 1
    
    add_444_block(0, 0)
    add_444_block(4, 4)
    
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    
    np.random.seed(42)
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 9)
    C_expected = A @ B
    
    T = np.zeros((4, 4, 9))
    for r in range(U.shape[1]):
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    T[i, j, k] += U[i*4+j, r] * V[j*9+k, r] * W[k*4+i, r]
    
    C_computed = np.einsum('ij,jk,ijk->ik', A, B, T)
    print(f"Max error: {np.max(np.abs(C_computed - C_expected))}")
    print(f"Rank: {U.shape[1]}")
