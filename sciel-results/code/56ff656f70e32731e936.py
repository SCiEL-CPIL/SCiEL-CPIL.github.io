import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Slice-and-dice construction:
    # Partition the middle dimension m=4 into two groups of size 2
    # For each group, solve a 3x2x7 sub-problem using schoolbook
    # Total rank = 2 * (3*2*7) = 84 (same as naive, but structured)
    
    # To actually reduce rank, we need non-trivial decompositions
    # For now, implement the structure with potential for optimization
    
    groups = [(0, 2), (2, 4)]
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    for j_start, j_end in groups:
        m_sub = j_end - j_start
        
        # Schoolbook for this sub-block: n * m_sub * p terms
        for i in range(n):
            for j in range(j_start, j_end):
                for k in range(p):
                    U_rows.append((i, j, k))
                    V_rows.append((j, k))
                    W_rows.append((k, i))
    
    rank = len(U_rows)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        i, j, k = U_rows[col]
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        
    return {"U": U, "V": V, "W": W}
