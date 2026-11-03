import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Rank 99 decomposition using 6 Strassen 2x2x2 blocks
    # Blocks are defined by (i_indices, j_indices, k_indices)
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [4, 5]),
    ]
    
    # Determine which (i, j, k) triples are covered by blocks
    covered = set()
    for i_list, j_list, k_list in blocks:
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    covered.add((i, j, k))
    
    # Calculate total rank
    rank = 6 * 7  # 6 Strassen blocks * 7 terms each
    # Add schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    rank += 1
                    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    
    # Strassen decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B01-B11) -> C01 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 += M7
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1, t_start):
        # M1
        U[i0 * m + j0][t_start] += 1
        U[i1 * m + j1][t_start] += 1
        V[j0 * p + k0][t_start] += 1
        V[j1 * p + k1][t_start] += 1
        W[k0 * n + i0][t_start] += 1
        W[k1 * n + i1][t_start] += 1
        
        # M2
        U[i1 * m + j0][t_start+1] += 1
        U[i1 * m + j1][t_start+1] += 1
        V[j0 * p + k0][t_start+1] += 1
        W[k0 * n + i1][t_start+1] += 1
        
        # M3
        U[i0 * m + j0][t_start+2] += 1
        V[j0 * p + k1][t_start+2] += 1
        V[j1 * p + k1][t_start+2] -= 1
        W[k1 * n + i0][t_start+2] += 1
        
        # M4
        U[i1 * m + j1][t_start+3] += 1
        V[j1 * p + k0][t_start+3] += 1
        V[j0 * p + k0][t_start+3] -= 1
        W[k0 * n + i0][t_start+3] += 1
        W[k0 * n + i1][t_start+3] += 1
        
        # M5
        U[i0 * m + j0][t_start+4] += 1
        U[i0 * m + j1][t_start+4] += 1
        V[j1 * p + k1][t_start+4] += 1
        W[k0 * n + i0][t_start+4] -= 1
        W[k1 * n + i0][t_start+4] += 1
        
        # M6
        U[i1 * m + j0][t_start+5] += 1
        U[i0 * m + j0][t_start+5] -= 1
        V[j0 * p + k0][t_start+5] += 1
        V[j0 * p + k1][t_start+5] += 1
        W[k1 * n + i1][t_start+5] += 1
        
        # M7
        U[i0 * m + j1][t_start+6] += 1
        U[i1 * m + j1][t_start+6] -= 1
        V[j1 * p + k0][t_start+6] += 1
        V[j1 * p + k1][t_start+6] += 1
        W[k0 * n + i0][t_start+6] += 1
        W[k1 * n + i1][t_start+6] += 1

    for i_list, j_list, k_list in blocks:
        i0, i1 = i_list
        j0, j1 = j_list
        k0, k1 = k_list
        add_strassen_block(i0, i1, j0, j1, k0, k1, t)
        t += 7
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
                    t += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
