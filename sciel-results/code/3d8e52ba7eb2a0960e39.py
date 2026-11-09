"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 26 decomposition using Strassen for 2x2 block and schoolbook for rest
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    # u_idx, v_idx, w_idx are flat indices for U, V, W
    # coeffs are the coefficients for that term
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs, col):
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][col] = coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][col] = coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][col] = coeff

    # Indices mapping:
    # A, B, C are 3x3
    # 0: (0,0) 1: (0,1) 2: (0,2)
    # 3: (1,0) 4: (1,1) 5: (1,2)
    # 6: (2,0) 7: (2,1) 8: (2,2)
    
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A11 indices: 0, 1, 3, 4
    # B11 indices: 0, 1, 3, 4
    # Output C11 indices: 0, 1, 3, 4
    
    # M1 = (a00+a11)(b00+b11)
    add_term([0, 4], [1, 1], [0, 4], [1, 1], [0, 1, 3, 4], [1, 1, 1, 1], 0)
    
    # M2 = (a10+a11)b00
    add_term([3, 4], [1, 1], [0], [1], [3, 4], [1, -1], 1)
    
    # M3 = a00(b01-b11)
    add_term([0], [1], [1, 4], [1, -1], [1, 4], [1, 1], 2)
    
    # M4 = a11(b10-b00)
    add_term([4], [1], [3, 0], [1, -1], [3, 4], [1, 1], 3)
    
    # M5 = (a00+a01)b11
    add_term([0, 1], [1, 1], [4], [1], [1, 4], [1, 1], 4)
    
    # M6 = (a10-a00)(b00+b01)
    add_term([3, 0], [1, -1], [0, 1], [1, 1], [4], [1], 5)
    
    # M7 = (a01-a11)(b10+b11)
    add_term([1, 4], [1, -1], [3, 4], [1, 1], [4], [1], 6)
    
    # 2. A12 * B21 -> C11
    # A12: 2, 5. B21: 6, 7. C11: 0, 1, 3, 4
    # a02*b20 -> C00 (0)
    add_term([2], [1], [6], [1], [0], [1], 7)
    # a02*b21 -> C01 (1)
    add_term([2], [1], [7], [1], [1], [1], 8)
    # a12*b20 -> C10 (3)
    add_term([5], [1], [6], [1], [3], [1], 9)
    # a12*b21 -> C11 (4)
    add_term([5], [1], [7], [1], [4], [1], 10)
    
    # 3. A11 * B12 -> C12
    # A11: 0,1,3,4. B12: 2,5. C12: 2,5
    # a00*b02 -> C02 (2)
    add_term([0], [1], [2], [1], [2], [1], 11)
    # a01*b12 -> C02 (2)
    add_term([1], [1], [5], [1], [2], [1], 12)
    # a10*b02 -> C12 (5)
    add_term([3], [1], [2], [1], [5], [1], 13)
    # a11*b12 -> C12 (5)
    add_term([4], [1], [5], [1], [5], [1], 14)
    
    # 4. A12 * B22 -> C12
    # A12: 2,5. B22: 8. C12: 2,5
    # a02*b22 -> C02 (2)
    add_term([2], [1], [8], [1], [2], [1], 15)
    # a12*b22 -> C12 (5)
    add_term([5], [1], [8], [1], [5], [1], 16)
    
    # 5. A21 * B11 -> C21
    # A21: 6,7. B11: 0,1,3,4. C21: 6,7
    # a20*b00 -> C20 (6)
    add_term([6], [1], [0], [1], [6], [1], 17)
    # a21*b10 -> C20 (6)
    add_term([7], [1], [3], [1], [6], [1], 18)
    # a20*b01 -> C21 (7)
    add_term([6], [1], [1], [1], [7], [1], 19)
    # a21*b11 -> C21 (7)
    add_term([7], [1], [4], [1], [7], [1], 20)
    
    # 6. A22 * B21 -> C21
    # A22: 8. B21: 6,7. C21: 6,7
    # a22*b20 -> C20 (6)
    add_term([8], [1], [6], [1], [6], [1], 21)
    # a22*b21 -> C21 (7)
    add_term([8], [1], [7], [1], [7], [1], 22)
    
    # 7. A21 * B12 -> C22
    # A21: 6,7. B12: 2,5. C22: 8
    # a20*b02 -> C22 (8)
    add_term([6], [1], [2], [1], [8], [1], 23)
    # a21*b12 -> C22 (8)
    add_term([7], [1], [5], [1], [8], [1], 24)
    
    # 8. A22 * B22 -> C22
    # A22: 8. B22: 8. C22: 8
    # a22*b22 -> C22 (8)
    add_term([8], [1], [8], [1], [8], [1], 25)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
