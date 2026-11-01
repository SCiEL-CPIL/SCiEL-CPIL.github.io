import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-22 exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    This uses Strassen's algorithm on the 2x2 sub-blocks and 
    direct multiplication for the remaining parts.
    """
    n, m, p = 2, 3, 4
    rank = 22
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to set U, V, W entries
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    col = 0
    
    # Strassen for A_L (2x2) * B_1 (2x2) -> C_11 (2x2)
    # A_L indices in U: 
    # a00: 0, a01: 1, a10: 3, a11: 4
    # B_1 indices in V:
    # b00: 0, b01: 1, b10: 4, b11: 5
    # C_11 indices in W:
    # c00: 0, c01: 1, c10: 2, c11: 3
    
    # M1 = (a00+a11)(b00+b11) -> c00, c11
    add_term(0, 0, 0, 1.0)
    add_term(4, 0, 0, 1.0)
    add_term(0, 5, 0, 1.0)
    add_term(4, 5, 0, 1.0)
    # Wait, Strassen terms are products of linear combinations.
    # The tensor decomposition adds terms of the form u_i * v_j * w_k.
    # For M1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
    # This contributes to c00 and c11.
    # So we add 4 terms for c00 and 4 terms for c11?
    # No, the "multiplication" M1 is a single rank-1 component in the bilinear algorithm sense?
    # No, the rank of the tensor is the number of multiplications.
    # Each multiplication is a rank-1 tensor?
    # A multiplication $x = u \cdot v$ is a rank-1 tensor if we view it as $x_{ijk} = u_i v_j$.
    # But here $x$ is a scalar that contributes to multiple $C$ entries.
    # The decomposition is $C_{ij} = \sum_{r=1}^{R} u_{ir} v_{jr} w_{kr}$.
    # So for each $r$, we have one $u$ vector, one $v$ vector, one $w$ vector.
    # The product $u \cdot v$ is a scalar. This scalar is multiplied by $w$ to form contributions to C.
    # So for M1, we define $u^{(1)}$, $v^{(1)}$, $w^{(1)}$.
    # $u^{(1)}$ has 1 at a00, 1 at a11.
    # $v^{(1)}$ has 1 at b00, 1 at b11.
    # $w^{(1)}$ has 1 at c00, 1 at c11.
    # This is ONE term in the sum.
    
    # Let's re-implement with this understanding.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def set_term(r, u_indices, v_indices, w_indices, val=1.0):
        for i in u_indices:
            U[i][r] = val
        for j in v_indices:
            V[j][r] = val
        for k in w_indices:
            W[k][r] = val
            
    # Note: If a vector has multiple non-zero entries, they are all set to val.
    # This works for linear combinations with coefficients 1 or -1.
    
    # Strassen 1 (B_1, cols 0,1)
    # M1: (a00+a11)(b00+b11) -> c00, c11
    set_term(0, [0, 4], [0, 5], [0, 3])
    
    # M2: (a10+a11)b00 -> c10
    set_term(1, [3, 4], [0], [2])
    
    # M3: a00(b01-b11) -> c01
    # u: a00 (0). v: b01 (1), b11 (5). w: c01 (1).
    # But coefficients are 1 and -1.
    # We can't just set all to 1.
    # We need to handle signs.
    # The interface allows half-integer/integer entries.
    # We can split a term with -1 into two terms? No, that increases rank.
    # We can just set the value to -1.
    
    # Let's modify set_term to take values.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_to_term(r, u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val
        
    # This is getting complicated because a single "multiplication" M_k involves 
    # linear combinations in U and V.
    # E.g. M1 = (a00+a11)(b00+b11).
    # U vector: [1, 0, 0, 0, 1, 0]
    # V vector: [1, 0, 0, 0, 0, 1]
    # W vector: [1, 0, 0, 1, 0, 0, 0, 0]
    # This is valid.
    
    # Let's define the vectors explicitly.
    
    # Strassen 1
    # M1
    U[0][0] = 1.0; U[4][0] = 1.0
    V[0][0] = 1.0; V[5][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # M2
    U[3][1] = 1.0; U[4][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0
    
    # M3: a00(b01-b11)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[5][2] = -1.0
    W[1][2] = 1.0
    
    # M4: a11(b10-b00)
    U[4][3] = 1.0
    V[4][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[2][3] = 1.0
    
    # M5: (a00+a01)b11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[5][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0
    
    # M6: (a10-a00)(b00+b01)
    U[3][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1.0; U[4][6] = -1.0
    V[4][6] = 1.0; V[5][6] = 1.0
    W[0][6] = 1.0
    
    # Strassen 2 (B_2, cols 2,3)
    # Indices in V: b02=2, b03=3, b12=6, b13=7
    # Indices in W: c02=4, c03=6, c12=5, c13=7
    
    # M8 (M1 of block 2)
    U[0][7] = 1.0; U[4][7] = 1.0
    V[2][7] = 1.0; V[7][7] = 1.0
    W[4][7] = 1.0; W[7][7] = 1.0
    
    # M9 (M2 of block 2)
    U[3][8] = 1.0; U[4][8] = 1.0
    V[2][8] = 1.0
    W[5][8] = 1.0
    
    # M10 (M3 of block 2)
    U[0][9] = 1.0
    V[3][9] = 1.0; V[7][9] = -1.0
    W[6][9] = 1.0
    
    # M11 (M4 of block 2)
    U[4][10] = 1.0
    V[6][10] = 1.0; V[2][10] = -1.0
    W[4][10] = 1.0; W[5][10] = 1.0
    
    # M12 (M5 of block 2)
    U[0][11] = 1.0; U[1][11] = 1.0
    V[7][11] = 1.0
    W[4][11] = -1.0; W[6][11] = 1.0
    
    # M13 (M6 of block 2)
    U[3][12] = 1.0; U[0][12] = -1.0
    V[2][12] = 1.0; V[3][12] = 1.0
    W[7][12] = 1.0
    
    # M14 (M7 of block 2)
    U[1][13] = 1.0; U[4][13] = -1.0
    V[6][13] = 1.0; V[7][13] = 1.0
    W[4][13] = 1.0
    
    # Outer Product (A_R * B_B)
    # A_R: a02=2, a12=5
    # B_B: b20=8, b21=9, b22=10, b23=11
    
    # O1: a02 * b20 -> c00
    U[2][14] = 1.0; V[8][14] = 1.0; W[0][14] = 1.0
    
    # O2: a12 * b20 -> c10
    U[5][15] = 1.0; V[8][15] = 1.0; W[2][15] = 1.0
    
    # O3: a02 * b21 -> c01
    U[2][16] = 1.0; V[9][16] = 1.0; W[1][16] = 1.0
    
    # O4: a12 * b21 -> c11
    U[5][17] = 1.0; V[9][17] = 1.0; W[3][17] = 1.0
    
    # O5: a02 * b22 -> c02
    U[2][18] = 1.0; V[10][18] = 1.0; W[4][18] = 1.0
    
    # O6: a12 * b22 -> c12
    U[5][19] = 1.0; V[10][19] = 1.0; W[5][19] = 1.0
    
    # O7: a02 * b23 -> c03
    U[2][20] = 1.0; V[11][20] = 1.0; W[6][20] = 1.0
    
    # O8: a12 * b23 -> c13
    U[5][21] = 1.0; V[11][21] = 1.0; W[7][21] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
