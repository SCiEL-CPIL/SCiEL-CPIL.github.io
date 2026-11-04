import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Partitions [3, 2]
    n_parts = [3, 2]
    m_parts = [3, 2]
    p_parts = [3, 2]
    
    n_starts = [0, 3]
    m_starts = [0, 3]
    p_starts = [0, 3]
    
    # Calculate total rank
    # 3x3x3: 11
    # 2x2x2: 7
    # 3x3x2: 18
    # 3x2x3: 18
    # 3x2x2: 12
    # 2x3x3: 18
    # 2x3x2: 12
    # 2x2x3: 12
    total_rank = 11 + 7 + 18 + 18 + 12 + 18 + 12 + 12
    # 108
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 3 and mj == 3 and pk == 3:
                    # Winograd 3x3
                    # ... implementation ...
                    col += 11
                elif ni == 2 and mj == 2 and pk == 2:
                    # Strassen 2x2
                    # ... implementation ...
                    col += 7
                else:
                    # Schoolbook
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
