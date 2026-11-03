import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # U2, V2, W2 are 4x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1,j=1
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # j=0,k=0
        [0, 0, 1, 0, 0, 1, 0],  # j=0,k=1
        [0, 0, 0, 1, 0, 0, 1],  # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1,k=1
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0], # k=0,i=0
        [0, 1, 0, 1, 0, 0, 0], # k=0,i=1
        [0, 0, 1, 0, 1, 0, 0], # k=1,i=0
        [1, 0, 0, 0, 0, 1, 1]  # k=1,i=1
    ]
    
    # Construct 4x4 decomposition
    for i in range(n):
        for j in range(m):
            # Decompose i, j into block (i1, j1) and element (i0, j0)
            i1 = i // 2
            i0 = i % 2
            j1 = j // 2
            j0 = j % 2
            
            u_idx = i * m + j
            u_block_idx = i1 * 2 + j1
            u_elem_idx = i0 * 2 + j0
            
            for s1 in range(7):
                val1 = U2[u_block_idx][s1]
                if val1 == 0:
                    continue
                for s2 in range(7):
                    val2 = U2[u_elem_idx][s2]
                    if val2 == 0:
                        continue
                    r = s1 * 7 + s2
                    U[u_idx][r] = val1 * val2

    for j in range(m):
        for k in range(p):
            j1 = j // 2
            j0 = j % 2
            k1 = k // 2
            k0 = k % 2
            
            v_idx = j * p + k
            v_block_idx = j1 * 2 + k1
            v_elem_idx = j0 * 2 + k0
            
            for s1 in range(7):
                val1 = V2[v_block_idx][s1]
                if val1 == 0:
                    continue
                for s2 in range(7):
                    val2 = V2[v_elem_idx][s2]
                    if val2 == 0:
                        continue
                    r = s1 * 7 + s2
                    V[v_idx][r] = val1 * val2

    for k in range(p):
        for i in range(n):
            k1 = k // 2
            k0 = k % 2
            i1 = i // 2
            i0 = i % 2
            
            w_idx = k * n + i
            w_block_idx = k1 * 2 + i1
            w_elem_idx = k0 * 2 + i0
            
            for s1 in range(7):
                val1 = W2[w_block_idx][s1]
                if val1 == 0:
                    continue
                for s2 in range(7):
                    val2 = W2[w_elem_idx][s2]
                    if val2 == 0:
                        continue
                    r = s1 * 7 + s2
                    W[w_idx][r] = val1 * val2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
