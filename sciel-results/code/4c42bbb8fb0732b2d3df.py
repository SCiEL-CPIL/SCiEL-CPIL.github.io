"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: split m=3 into 1+2, split p=10 into 5 groups of 2
    # Block 1: 2x1 x 1x10, schoolbook rank 20
    # Blocks 2-6: 2x2 x 2x2, Strassen rank 7 each, total 35
    # Total rank: 55
    rank = 20 + 5 * 7
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    # Block 1: j=0, all k
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Blocks 2-6: j in {1,2}, k in {2g, 2g+1}
    # Strassen 2x2x2 decomposition
    strassen_U = [
        [1, 0, 0, 1],  # A00+A11
        [0, 1, 0, 1],  # A10+A11
        [1, 0, 0, 0],  # A00
        [0, 0, 0, 1],  # A11
        [1, 0, 1, 0],  # A00+A01
        [0, 1, -1, 0], # A10-A00
        [0, 1, 0, -1], # A01-A11
    ]
    strassen_V = [
        [1, 0, 0, 1],  # B00+B11
        [1, 0, 0, 0],  # B00
        [0, 0, 1, -1], # B10-B11
        [1, -1, 0, 0], # B00-B10
        [0, 0, 0, 1],  # B11
        [1, 1, 0, 0],  # B00+B01
        [0, 1, 1, 0],  # B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1],  # C00+C11
        [0, 1, 0, 0],  # C10
        [0, 0, 1, 0],  # C01
        [0, 1, 0, 0],  # C10
        [0, 0, 1, 0],  # C01
        [0, 0, 0, 1],  # C11
        [1, 0, 0, 0],  # C00
    ]
    
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        for t in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    val = strassen_U[t][i_loc * 2 + j_loc]
                    if val != 0:
                        U[(i_loc) * m + (1 + j_loc)][col] = val
            for j_loc in range(2):
                for k_loc in range(2):
                    val = strassen_V[t][j_loc * 2 + k_loc]
                    if val != 0:
                        V[(1 + j_loc) * p + (k0 + k_loc)][col] = val
            for k_loc in range(2):
                for i_loc in range(2):
                    val = strassen_W[t][k_loc * 2 + i_loc]
                    if val != 0:
                        W[(k0 + k_loc) * n + i_loc][col] = val
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
