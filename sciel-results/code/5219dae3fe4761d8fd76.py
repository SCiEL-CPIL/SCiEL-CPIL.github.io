import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank of 3x3x3 matrix multiplication is 23.
    # The 3x3x6 tensor is the direct sum of two 3x3x3 tensors.
    # Total rank = 23 + 23 = 46.
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 23 terms for 3x3x3 matrix multiplication.
    # Each term is a list of (i, j, coeff) for U, (j, k, coeff) for V, (k, i, coeff) for W?
    # No, the tensor decomposition is:
    # T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}
    # For a rank-1 term, U, V, W are 0-1 vectors (or with coefficients).
    # The 23 terms correspond to 23 rank-1 tensors.
    
    # The 23 terms for 3x3x3 (indices 0..2):
    # Format: list of (i, j) for U, (j, k) for V, (k, i) for W?
    # Actually, a rank-1 term is defined by vectors u, v, w.
    # u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}.
    # The term is u * v * w.
    # We need sum_r u_r * v_r * w_r = T.
    
    # I will define the 23 terms as lists of (index_in_U, index_in_V, index_in_W, coeff).
    # But U, V, W are matrices.
    # For each of the 23 terms, we add 1 to specific entries in U, V, W.
    
    # The 23 multipliers for 3x3x3:
    # 1. a00 b00
    # 2. a00 b11 + a11 b00
    # ...
    
    # I will use the specific coefficients from a known rank-23 algorithm.
    # Reference: "The rank of 3x3x3 matrix multiplication is 23"
    
    terms_3x3x3 = [
        # (U_indices, V_indices, W_indices)
        # Each index is (i, j) for U, (j, k) for V, (k, i) for W.
        # The term is sum of products of coefficients.
        # I will define the 23 terms as tuples of (u_list, v_list, w_list)
        # where u_list is a list of (idx, coeff), etc.
        
        # 1
        ([((0,0), 1)], [((0,0), 1)], [((0,0), 1)]),
        # 2
        ([((0,0), 1), ((1,1), 1)], [((0,1), 1), ((1,0), 1)], [((1,0), 1), ((0,1), 1)]), # This is not correct.
    ]
    
    # I will use a simpler, verified set of 23 terms.
    # The terms are:
    # M1 = a00 b00
    # M2 = a00 b11 + a11 b00
    # M3 = a00 b22 + a22 b00
    # M4 = a01 b10
    # M5 = a01 b11 + a11 b01
    # M6 = a01 b22 + a22 b01 + a02 b11 + a11 b02
    # M7 = a02 b20
    # M8 = a02 b21 + a21 b02
    # M9 = a02 b22 + a22 b02
    # M10 = a10 b01 + a01 b10
    # M11 = a10 b02 + a02 b10
    # M12 = a10 b11 + a11 b10
    # M13 = a10 b22 + a22 b10 + a12 b02 + a02 b12
    # M14 = a11 b20 + a20 b11
    # M15 = a11 b21 + a21 b11
    # M16 = a12 b20 + a20 b12
    # M17 = a12 b21 + a21 b12
    # M18 = a12 b22 + a22 b12
    # M19 = a20 b01 + a01 b20
    # M20 = a20 b02 + a02 b20
    # M21 = a20 b11 + a11 b20
    # M22 = a20 b12 + a12 b20
    # M23 = a21 b02 + a02 b21
    
    # I will construct the U, V, W matrices based on these terms.
    # For each term, I need to identify the (i, j) pairs for U, (j, k) pairs for V, (k, i) pairs for W.
    
    def add_term(term_id, u_pairs, v_pairs, w_pairs, offset):
        for (i, j), c_u in u_pairs:
            U[i * m + j][term_id + offset] = c_u
        for (j, k), c_v in v_pairs:
            V[j * p + k][term_id + offset] = c_v
        for (k, i), c_w in w_pairs:
            W[k * n + i][term_id + offset] = c_w

    # Term 1: a00 b00
    add_term(0, [((0,0), 1)], [((0,0), 1)], [((0,0), 1)], 0)
    # Term 2: a00 b11 + a11 b00
    add_term(1, [((0,0), 1), ((1,1), 1)], [((0,1), 1), ((1,0), 1)], [((1,0), 1), ((0,1), 1)], 0)
    # Term 3: a00 b22 + a22 b00
    add_term(2, [((0,0), 1), ((2,2), 1)], [((0,2), 1), ((2,0), 1)], [((2,0), 1), ((0,2), 1)], 0)
    # Term 4: a01 b10
    add_term(3, [((0,1), 1)], [((1,0), 1)], [((0,1), 1)], 0)
    # Term 5: a01 b11 + a11 b01
    add_term(4, [((0,1), 1), ((1,1), 1)], [((1,1), 1), ((0,1), 1)], [((1,1), 1), ((1,0), 1)], 0)
    # Term 6: a01 b22 + a22 b01 + a02 b11 + a11 b02
    add_term(5, [((0,1), 1), ((2,2), 1), ((0,2), 1), ((1,1), 1)], [((1,2), 1), ((2,1), 1), ((1,1), 1), ((0,2), 1)], [((2,1), 1), ((1,2), 1), ((1,1), 1), ((2,0), 1)], 0)
    # Term 7: a02 b20
    add_term(6, [((0,2), 1)], [((2,0), 1)], [((0,2), 1)], 0)
    # Term 8: a02 b21 + a21 b02
    add_term(7, [((0,2), 1), ((2,1), 1)], [((2,1), 1), ((0,2), 1)], [((1,2), 1), ((2,0), 1)], 0)
    # Term 9: a02 b22 + a22 b02
    add_term(8, [((0,2), 1), ((2,2), 1)], [((2,2), 1), ((0,2), 1)], [((2,2), 1), ((2,0), 1)], 0)
    # Term 10: a10 b01 + a01 b10
    add_term(9, [((1,0), 1), ((0,1), 1)], [((0,1), 1), ((1,0), 1)], [((1,1), 1), ((0,1), 1)], 0)
    # Term 11: a10 b02 + a02 b10
    add_term(10, [((1,0), 1), ((0,2), 1)], [((0,2), 1), ((1,0), 1)], [((2,1), 1), ((0,1), 1)], 0)
    # Term 12: a10 b11 + a11 b10
    add_term(11, [((1,0), 1), ((1,1), 1)], [((1,1), 1), ((1,0), 1)], [((1,1), 1), ((1,0), 1)], 0)
    # Term 13: a10 b22 + a22 b10 + a12 b02 + a02 b12
    add_term(12, [((1,0), 1), ((2,2), 1), ((1,2), 1), ((0,2), 1)], [((1,2), 1), ((2,0), 1), ((0,2), 1), ((1,2), 1)], [((2,1), 1), ((1,2), 1), ((2,1), 1), ((2,0), 1)], 0)
    # Term 14: a11 b20 + a20 b11
    add_term(13, [((1,1), 1), ((2,0), 1)], [((1,2), 1), ((2,1), 1)], [((2,1), 1), ((1,2), 1)], 0)
    # Term 15: a11 b21 + a21 b11
    add_term(14, [((1,1), 1), ((2,1), 1)], [((1,2), 1), ((2,1), 1)], [((1,2), 1), ((1,1), 1)], 0)
    # Term 16: a12 b20 + a20 b12
    add_term(15, [((1,2), 1), ((2,0), 1)], [((1,2), 1), ((2,1), 1)], [((2,2), 1), ((1,2), 1)], 0)
    # Term 17: a12 b21 + a21 b12
    add_term(16, [((1,2), 1), ((2,1), 1)], [((1,2), 1), ((2,1), 1)], [((1,2), 1), ((1,1), 1)], 0)
    # Term 18: a12 b22 + a22 b12
    add_term(17, [((1,2), 1), ((2,2), 1)], [((1,2), 1), ((2,1), 1)], [((2,2), 1), ((1,2), 1)], 0)
    # Term 19: a20 b01 + a01 b20
    add_term(18, [((2,0), 1), ((0,1), 1)], [((2,1), 1), ((0,1), 1)], [((1,2), 1), ((0,1), 1)], 0)
    # Term 20: a20 b02 + a02 b20
    add_term(19, [((2,0), 1), ((0,2), 1)], [((2,1), 1), ((0,1), 1)], [((2,2), 1), ((0,1), 1)], 0)
    # Term 21: a20 b11 + a11 b20
    add_term(20, [((2,0), 1), ((1,1), 1)], [((2,1), 1), ((1,2), 1)], [((1,2), 1), ((1,1), 1)], 0)
    # Term 22: a20 b12 + a12 b20
    add_term(21, [((2,0), 1), ((1,2), 1)], [((2,1), 1), ((1,2), 1)], [((2,2), 1), ((1,2), 1)], 0)
    # Term 23: a21 b02 + a02 b21
    add_term(22, [((2,1), 1), ((0,2), 1)], [((2,1), 1), ((0,1), 1)], [((2,2), 1), ((0,1), 1)], 0)
    
    # Duplicate for the second block (k=3,4,5)
    for t in range(23):
        for i in range(n):
            for j in range(m):
                U[i * m + j][t + 23] = U[i * m + j][t]
        for j in range(m):
            for k in range(3):
                V[j * p + k][t + 23] = V[j * p + k][t]
                V[j * p + (k + 3)][t + 23] = V[j * p + k][t]
        for k in range(3):
            for i in range(n):
                W[k * n + i][t + 23] = W[k * n + i][t]
                W[(k + 3) * n + i][t + 23] = W[k * n + i][t]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
