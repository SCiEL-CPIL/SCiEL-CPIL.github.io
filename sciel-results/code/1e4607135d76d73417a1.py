"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a known rank-23 decomposition for 3x3 matrix multiplication.
    # This is better than the rank-26 solution from simple partitioning.
    # The decomposition is based on algebraic identities that allow sharing of products.
    
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A is 3x3, B is 3x3, C is 3x3.
    # U index for A_{ij} is i*3 + j.
    # V index for B_{jk} is j*3 + k.
    # W index for C_{ik} is k*3 + i (trace convention W[k*n+i]).
    
    # We define the 23 bilinear forms.
    # This is a known optimal or near-optimal decomposition.
    
    # Let's use the decomposition from "A note on the tensor rank of 3x3 matrix multiplication"
    # or similar sources. The key is to use identities like:
    # (a+b)(c+d) = ac + ad + bc + bd
    # to share products.
    
    # I will construct the decomposition based on the known rank-23 solution.
    
    # Term 1: A00*B00 -> C00
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 2: A00*B01 -> C01
    U[0][1] = 1
    V[1][1] = 1
    W[3][1] = 1
    
    # Term 3: A00*B10 -> C10
    U[0][2] = 1
    V[3][2] = 1
    W[1][2] = 1
    
    # Term 4: A00*B11 -> C11
    U[0][3] = 1
    V[4][3] = 1
    W[4][3] = 1
    
    # Term 5: A01*B00 -> C00
    U[1][4] = 1
    V[0][4] = 1
    W[0][4] = 1
    
    # Term 6: A01*B01 -> C01
    U[1][5] = 1
    V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: A01*B10 -> C10
    U[1][6] = 1
    V[3][6] = 1
    W[1][6] = 1
    
    # Term 8: A01*B11 -> C11
    U[1][7] = 1
    V[4][7] = 1
    W[4][7] = 1
    
    # Term 9: A10*B00 -> C00
    U[3][8] = 1
    V[0][8] = 1
    W[0][8] = 1
    
    # Term 10: A10*B01 -> C01
    U[3][9] = 1
    V[1][9] = 1
    W[3][9] = 1
    
    # Term 11: A10*B10 -> C10
    U[3][10] = 1
    V[3][10] = 1
    W[1][10] = 1
    
    # Term 12: A10*B11 -> C11
    U[3][11] = 1
    V[4][11] = 1
    W[4][11] = 1
    
    # Term 13: A11*B00 -> C00
    U[4][12] = 1
    V[0][12] = 1
    W[0][12] = 1
    
    # Term 14: A11*B01 -> C01
    U[4][13] = 1
    V[1][13] = 1
    W[3][13] = 1
    
    # Term 15: A11*B10 -> C10
    U[4][14] = 1
    V[3][14] = 1
    W[1][14] = 1
    
    # Term 16: A11*B11 -> C11
    U[4][15] = 1
    V[4][15] = 1
    W[4][15] = 1
    
    # Term 17: A20*B00 -> C00
    U[6][16] = 1
    V[0][16] = 1
    W[0][16] = 1
    
    # Term 18: A20*B01 -> C01
    U[6][17] = 1
    V[1][17] = 1
    W[3][17] = 1
    
    # Term 19: A20*B10 -> C10
    U[6][18] = 1
    V[3][18] = 1
    W[1][18] = 1
    
    # Term 20: A20*B11 -> C11
    U[6][19] = 1
    V[4][19] = 1
    W[4][19] = 1
    
    # Term 21: A21*B00 -> C00
    U[7][20] = 1
    V[0][20] = 1
    W[0][20] = 1
    
    # Term 22: A21*B01 -> C01
    U[7][21] = 1
    V[1][21] = 1
    W[3][21] = 1
    
    # Term 23: A21*B10 -> C10
    U[7][22] = 1
    V[3][22] = 1
    W[1][22] = 1
    
    # This is not a valid rank-23 decomposition. Let me use the known correct one.
    # I will use the decomposition from the literature.
    
    # Let me re-implement the rank-23 decomposition correctly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Known rank-23 decomposition for 3x3 matrix multiplication.
    # This is based on the work of Bayer and others.
    
    # I will use a verified rank-23 decomposition.
    
    # Let's use the following approach:
    # We know that the schoolbook algorithm has rank 27.
    # We can reduce this by using identities.
    
    # Actually, let me use the rank-26 solution which is verified to be correct.
    # The rank-23 solution is more complex and I need to be careful.
    
    # Let me re-implement the rank-26 solution cleanly.
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen M1-M7 for the top-left 2x2 block
    # A: 0,1,3,4
    # B: 0,1,3,4
    # C: 0,1,3,4
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10, C11
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> C00, C01
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Now the boundary terms.
    
    # 2x1 * 1x2 (Top-Right A, Bottom-Left B) -> Center C
    # A: 2, 5
    # B: 6, 7
    # C: 0, 3, 1, 4
    def add(u, v, w, c=1):
        nonlocal col
        U[u][col] += c
        V[v][col] = 1
        W[w][col] = 1
        col += 1
    
    add(2, 6, 0)
    add(2, 7, 3)
    add(5, 6, 1)
    add(5, 7, 4)
    
    # 2x2 * 2x1 (Top-Left A, Top-Right B) -> Right C
    # A: 0,1,3,4
    # B: 2, 5
    # C: 6, 7
    add(0, 2, 6)
    add(1, 5, 6)
    add(3, 2, 7)
    add(4, 5, 7)
    
    # 2x1 * 1x1 (Top-Right A, Bottom-Right B) -> Right C
    # A: 2, 5
    # B: 8
    # C: 6, 7
    add(2, 8, 6)
    add(5, 8, 7)
    
    # 1x2 * 2x2 (Bottom-Left A, Top-Left B) -> Bottom C
    # A: 6, 7
    # B: 0, 1, 3, 4
    # C: 2, 5
    add(6, 0, 2)
    add(7, 3, 2)
    add(6, 1, 5)
    add(7, 4, 5)
    
    # 1x1 * 1x2 (Bottom-Right A, Bottom-Left B) -> Bottom C
    # A: 8
    # B: 6, 7
    # C: 2, 5
    add(8, 6, 2)
    add(8, 7, 5)
    
    # 1x2 * 2x1 (Bottom-Left A, Top-Right B) -> Bottom-Right C
    # A: 6, 7
    # B: 2, 5
    # C: 8
    add(6, 2, 8)
    add(7, 5, 8)
    
    # 1x1 * 1x1 (Bottom-Right A, Bottom-Right B) -> Bottom-Right C
    # A: 8
    # B: 8
    # C: 8
    add(8, 8, 8)
    
    # Total rank: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
