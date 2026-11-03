"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition sizes
    n_blocks = [2, 1]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 2]
    
    # Total rank
    rank = 0
    for ni in n_blocks:
        for mj in m_blocks:
            for pk in p_blocks:
                if ni == 2 and mj == 2 and pk == 2:
                    rank += 7
                elif ni == 1 and mj == 2 and pk == 2:
                    rank += 4
                else:
                    rank += ni * mj * pk
                    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    rank_offset = 0
    n_start = 0
    for ni in n_blocks:
        m_start = 0
        for mj in m_blocks:
            p_start = 0
            for pk in p_blocks:
                if ni == 2 and mj == 2 and pk == 2:
                    tensors = get_strassen_tensors()
                    for r, (u_loc, v_loc, w_loc) in enumerate(tensors):
                        for i in range(ni):
                            for j in range(mj):
                                val = u_loc[i][j]
                                if val != 0:
                                    U[(n_start + i) * m + (m_start + j)][rank_offset + r] = val
                        for j in range(mj):
                            for k in range(pk):
                                val = v_loc[j][k]
                                if val != 0:
                                    V[(m_start + j) * p + (p_start + k)][rank_offset + r] = val
                        for k in range(pk):
                            for i in range(ni):
                                val = w_loc[k][i]
                                if val != 0:
                                    W[(p_start + k) * n + (n_start + i)][rank_offset + r] = val
                    rank_offset += 7
                elif ni == 1 and mj == 2 and pk == 2:
                    tensors = get_1x2x2_tensors()
                    for r, (u_loc, v_loc, w_loc) in enumerate(tensors):
                        for i in range(ni):
                            for j in range(mj):
                                val = u_loc[i][j]
                                if val != 0:
                                    U[(n_start + i) * m + (m_start + j)][rank_offset + r] = val
                        for j in range(mj):
                            for k in range(pk):
                                val = v_loc[j][k]
                                if val != 0:
                                    V[(m_start + j) * p + (p_start + k)][rank_offset + r] = val
                        for k in range(pk):
                            for i in range(ni):
                                val = w_loc[k][i]
                                if val != 0:
                                    W[(p_start + k) * n + (n_start + i)][rank_offset + r] = val
                    rank_offset += 4
                else:
                    rank_block = ni * mj * pk
                    for i in range(ni):
                        for j in range(mj):
                            for k in range(pk):
                                r = i * mj * pk + j * pk + k
                                U[(n_start + i) * m + (m_start + j)][rank_offset + r] = 1
                                V[(m_start + j) * p + (p_start + k)][rank_offset + r] = 1
                                W[(p_start + k) * n + (n_start + i)][rank_offset + r] = 1
                    rank_offset += rank_block
                p_start += pk
            m_start += mj
        n_start += ni
            
    return {"U": U, "V": V, "W": W}

def get_strassen_tensors():
    tensors = []
    
    u = [[1, 0], [0, 1]]
    v = [[1, 0], [0, 1]]
    w = [[1, 0], [0, 1]]
    tensors.append((u, v, w))
    
    u = [[0, 0], [1, 1]]
    v = [[1, 0], [0, 0]]
    w = [[0, 0], [1, 0]]
    tensors.append((u, v, w))
    
    u = [[1, 0], [0, 0]]
    v = [[0, 1], [0, -1]]
    w = [[0, 1], [0, 0]]
    tensors.append((u, v, w))
    
    u = [[0, 0], [0, 1]]
    v = [[-1, 0], [1, 0]]
    w = [[0, 0], [1, 0]]
    tensors.append((u, v, w))
    
    u = [[1, 1], [0, 0]]
    v = [[0, 0], [0, 1]]
    w = [[0, 1], [0, 0]]
    tensors.append((u, v, w))
    
    u = [[-1, 0], [1, 0]]
    v = [[1, 1], [0, 0]]
    w = [[0, 0], [0, 1]]
    tensors.append((u, v, w))
    
    u = [[0, 1], [-1, 0]]
    v = [[0, 0], [1, 1]]
    w = [[1, 0], [0, 0]]
    tensors.append((u, v, w))
    
    return tensors

def get_1x2x2_tensors():
    tensors = []
    
    u = [[1, 0]]
    v = [[1, 0], [0, 0]]
    w = [[1], [0]]
    tensors.append((u, v, w))
    
    u = [[0, 1]]
    v = [[0, 0], [1, 0]]
    w = [[1], [0]]
    tensors.append((u, v, w))
    
    u = [[1, 0]]
    v = [[0, 1], [0, 0]]
    w = [[0], [1]]
    tensors.append((u, v, w))
    
    u = [[0, 1]]
    v = [[0, 0], [0, 1]]
    w = [[0], [1]]
    tensors.append((u, v, w))
    
    return tensors
# EVOLVE-BLOCK-END
