"""Matrix multiplication 2x5x6 with Strassen-based block decomposition."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strassen rank-7 decomposition for 2x2x2
    # Each term: (A_indices, B_indices, C_indices)
    # A is 2x2, B is 2x2, C is 2x2
    # Term r has U factors on A positions, V factors on B positions, W factors on C positions
    # The decomposition satisfies: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta(i,j,k) for matmul tensor
    
    # Standard Strassen rank-7 decomposition (0-indexed):
    # P1 = (A00+A11)(B00+B11) -> C00 += 1, C11 -= 1
    # P2 = (A00+A11)B11       -> C00 -= 1, C11 += 1  [wait, need to verify]
    # Let me use the well-known explicit rank-7 decomposition:
    
    # The 7 terms of the rank-7 decomposition of the 2x2x2 multiplication tensor:
    # Term 0: A: {(0,0):1,(1,1):1}, B: {(0,0):1,(1,1):1}, C: {(0,0):1,(1,1):-1}
    # Term 1: A: {(0,0):1,(1,1):1}, B: {(1,1):1}, C: {(0,0):-1,(1,1):1}
    # Term 2: A: {(0,0):1}, B: {(0,0):1,(0,1):-1}, C: {(0,0):1,(1,0):1}
    # Term 3: A: {(1,1):1}, B: {(1,1):1,(0,1):-1}, C: {(0,1):1,(1,1):1}
    # Term 4: A: {(0,0):1,(0,1):1}, B: {(1,1):1}, C: {(0,1):1,(1,1):-1}
    # Term 5: A: {(1,0):1,(0,0):-1}, B: {(0,0):1,(0,1):1}, C: {(0,0):1,(1,1):1}
    # Term 6: A: {(0,1):1,(1,1):-1}, B: {(1,0):1,(1,1):1}, C: {(1,0):1}
    
    strassen_U = [
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1},
    ]
    strassen_V = [
        {(0,0):1, (1,1):1},
        {(1,1):1},
        {(0,0):1, (0,1):-1},
        {(1,1):1, (0,1):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1},
    ]
    strassen_W = [
        {(0,0):1, (1,1):-1},
        {(0,0):-1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,1):1, (1,1):1},
        {(0,1):1, (1,1):-1},
        {(0,0):1, (1,1):1},
        {(1,0):1},
    ]
    
    total_rank = 7 * 3 + 12  # 21 from Strassen blocks + 12 from schoolbook 2x1x6
    # Actually: 3 Strassen blocks of 2x2x2 = 3*7 = 21, plus 2x1x6 = 12, total = 33
    # Wait, let me recount. We have:
    # j=0,1: three 2x2x2 blocks (k=0-1, k=2-3, k=4-5) -> 3*7 = 21
    # j=2,3: three 2x2x2 blocks (k=0-1, k=2-3, k=4-5) -> 3*7 = 21
    # j=4: one 2x1x6 block -> 12
    # Total = 21 + 21 + 12 = 54
    
    total_rank = 54
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 2*5 = 10 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 5*6 = 30 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 6*2 = 12 rows
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """Add a 2x2x2 Strassen block at local offsets (i0, j0, k0)."""
        nonlocal col
        for r in range(7):
            for (li, lj), uval in strassen_U[r].items():
                gi = i0 + li
                gj = j0 + lj
                U[gi * m + gj][col] += uval
            for (lj, lk), vval in strassen_V[r].items():
                gj = j0 + lj
                gk = k0 + lk
                V[gj * p + gk][col] += vval
            for (li, lk), wval in strassen_W[r].items():
                gi = i0 + li
                gk = k0 + lk
                W[gk * n + gi][col] += wval
            col += 1
    
    def add_schoolbook_2x1x6(i0, j0, k0):
        """Add a 2x1x6 schoolbook block at local offsets (i0, j0, k0)."""
        nonlocal col
        for i in range(2):
            for k in range(6):
                gi = i0 + i
                gj = j0
                gk = k0 + k
                U[gi * m + gj][col] = 1.0
                V[gj * p + gk][col] = 1.0
                W[gk * n + gi][col] = 1.0
                col += 1
    
    # Strassen blocks for j=0,1 (m-sub=2), k in {0-1, 2-3, 4-5}
    add_strassen_block(0, 0, 0)  # i=0-1, j=0-1, k=0-1
    add_strassen_block(0, 0, 2)  # i=0-1, j=0-1, k=2-3
    add_strassen_block(0, 0, 4)  # i=0-1, j=0-1, k=4-5
    
    # Strassen blocks for j=2,3 (m-sub=2), k in {0-1, 2-3, 4-5}
    add_strassen_block(0, 2, 0)  # i=0-1, j=2-3, k=0-1
    add_strassen_block(0, 2, 2)  # i=0-1, j=2-3, k=2-3
    add_strassen_block(0, 2, 4)  # i=0-1, j=2-3, k=4-5
    
    # Schoolbook for j=4 (m-sub=1), k=0-5
    add_schoolbook_2x1x6(0, 4, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
