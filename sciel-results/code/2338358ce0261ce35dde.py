import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Partitions
    N_parts = [1, 2]
    M_parts = [1, 2, 2]
    P_parts = [2, 2, 2]
    
    # Calculate offsets
    N_starts = [0]
    for size in N_parts[:-1]:
        N_starts.append(N_starts[-1] + size)
        
    M_starts = [0]
    for size in M_parts[:-1]:
        M_starts.append(M_starts[-1] + size)
        
    P_starts = [0]
    for size in P_parts[:-1]:
        P_starts.append(P_starts[-1] + size)

    # Calculate total rank
    total_rank = 0
    for ni in N_parts:
        for mj in M_parts:
            for pk in P_parts:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Strassen vectors for 2x2x2
    # Local indices:
    # U: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V: 00, 01, 10, 11 -> 0, 1, 2, 3
    # W: 00, 01, 10, 11 -> 0, 1, 2, 3 (index = k*2 + i)
    # Note: W index for C_{i,j} is j*2 + i.
    # In Strassen formulas:
    # C_{00} is index 0
    # C_{01} is index 1
    # C_{10} is index 2
    # C_{11} is index 3
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 1, -1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    for i_idx, ni in enumerate(N_parts):
        n_start = N_starts[i_idx]
        for j_idx, mj in enumerate(M_parts):
            m_start = M_starts[j_idx]
            for k_idx, pk in enumerate(P_parts):
                p_start = P_starts[k_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Add Strassen decomposition
                    for l in range(7):
                        col = current_rank + l
                        # Add U
                        for local_i in range(2):
                            for local_j in range(2):
                                val = strassen_U[l][local_i * 2 + local_j]
                                if val != 0:
                                    U[n_start + local_i][col] += val
                        # Add V
                        for local_j in range(2):
                            for local_k in range(2):
                                val = strassen_V[l][local_j * 2 + local_k]
                                if val != 0:
                                    V[m_start + local_j][col] += val
                        # Add W
                        for local_k in range(2):
                            for local_i in range(2):
                                val = strassen_W[l][local_k * 2 + local_i]
                                if val != 0:
                                    W[p_start + local_k][col] += val
                    current_rank += 7
                else:
                    # Add Schoolbook decomposition
                    for local_i in range(ni):
                        for local_j in range(mj):
                            for local_k in range(pk):
                                col = current_rank
                                U[n_start + local_i][col] = 1
                                V[m_start + local_j][col] = 1
                                W[p_start + local_k][col] = 1
                                current_rank += 1
                                
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
