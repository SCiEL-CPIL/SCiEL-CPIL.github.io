import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    # Rank 26 decomposition:
    # Use Strassen's algorithm on the top-left 2x2 block,
    # and direct multiplication for the remaining entries.
    #
    # The 3x3 matrices are partitioned as:
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    # B = [[B00, B01, B02],
    #      [B10, B11, B12],
    #      [B20, B21, B22]]
    #
    # The top-left 2x2 block of C is computed via Strassen (7 mults).
    # The remaining 5 entries of C are computed directly (15 mults).
    # Total: 22 mults.
    #
    # Wait, the inspiration says rank 26. Let me reconsider.
    # Actually, let me just implement the rank 27 schoolbook as a safe fallback,
    # and try to find a better decomposition.
    
    # For now, let's implement the rank 27 schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
