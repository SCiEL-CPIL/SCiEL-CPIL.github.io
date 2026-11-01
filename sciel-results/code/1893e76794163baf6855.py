"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen term
    # x_indices: 4 indices in U for the 2x2 matrix X
    # y_indices: 4 indices in V for the 2x2 matrix Y
    # z_indices: 4 indices in W for the 2x2 result Z
    # col: start column in U, V, W
    def add_strassen(x_idx, y_idx, z_idx, col):
        # M1 = (X11+X22)(Y11+Y22)
        # M2 = (X21+X22)Y11
        # M3 = X11(Y12-Y22)
        # M4 = X22(Y21-Y11)
        # M5 = (X11+X12)Y22
        # M6 = (X21-X11)(Y11+Y12)
        # M7 = (X12-X22)(Y21+Y22)
        
        # Z11 = M1 + M4 - M5 + M7
        # Z12 = M3 + M5
        # Z21 = M2 + M4
        # Z22 = M1 - M2 + M3 + M6
        
        # We need to define the 7 rank-1 terms.
        # Term 1: M1
        # U: X11, X22
        # V: Y11, Y22
        # W: Z11, Z22
        
        # This manual expansion is tedious. Let's use a pre-computed list of coefficients.
        # Strassen coefficients for U, V, W.
        # Each term r has u_r, v_r, w_r vectors.
        
        # x_idx = [x11, x12, x21, x22]
        # y_idx = [y11, y12, y21, y22]
        # z_idx = [z11, z12, z21, z22]
        
        # Term 1: (x11+x22) * (y11+y22)
        # U: x11:1, x22:1
        # V: y11:1, y22:1
        # W: z11:1, z22:1
        
        # Term 2: (x21+x22) * y11
        # U: x21:1, x22:1
        # V: y11:1
        # W: z21:1
        
        # Term 3: x11 * (y12-y22)
        # U: x11:1
        # V: y12:1, y22:-1
        # W: z12:1
        
        # Term 4: x22 * (y21-y11)
        # U: x22:1
        # V: y21:1, y11:-1
        # W: z21:1, z22:-1? No.
        # Z22 = M1 - M2 + M3 + M6. M4 is not in Z22.
        # Z21 = M2 + M4.
        # So M4 contributes to Z21 and Z22?
        # Let's re-verify Strassen.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # T1 (M1): U[x11,x22], V[y11,y22], W[z11,z22]
        # T2 (M2): U[x21,x22], V[y11], W[z21]
        # T3 (M3): U[x11], V[y12,-y22], W[z12,z22]
        # T4 (M4): U[x22], V[y21,-y11], W[z21]
        # T5 (M5): U[x11,x12], V[y22], W[z11,z12]
        # T6 (M6): U[x21,-x11], V[y11,y12], W[z22]
        # T7 (M7): U[x12,-x22], V[y21,y22], W[z11]
        
        # Let's refine the W contributions.
        # T1: W[z11]=1, W[z22]=1
        # T2: W[z21]=1
        # T3: W[z12]=1, W[z22]=1
        # T4: W[z21]=1
        # T5: W[z11]=-1, W[z12]=1
        # T6: W[z22]=1
        # T7: W[z11]=1
        
        # Check C11: T1(1) + T4(0) - T5(-1) + T7(1) = 1 + 1 + 1 = 3? No.
        # C11 = M1 + M4 - M5 + M7.
        # Coeff of M1 in C11 is 1.
        # Coeff of M4 in C11 is 1.
        # Coeff of M5 in C11 is -1.
        # Coeff of M7 in C11 is 1.
        
        # So W[z11] should be:
        # T1: 1
        # T4: 0
        # T5: -1
        # T7: 1
        # Sum: 1 - 1 + 1 = 1. Correct.
        
        # W[z12]:
        # C12 = M3 + M5.
        # T3: 1
        # T5: 1
        # Sum: 2? No, M3 and M5 are distinct terms.
        # In the decomposition, each term is a rank-1 tensor.
        # So T3 is one rank-1 term. T5 is another.
        # W[z12] for T3 is 1.
        # W[z12] for T5 is 1.
        
        # W[z21]:
        # C21 = M2 + M4.
        # T2: 1
        # T4: 1
        
        # W[z22]:
        # C22 = M1 - M2 + M3 + M6.
        # T1: 1
        # T2: -1
        # T3: 1
        # T6: 1
        
        # Let's code this.
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u_coeffs: dict {index: coeff}
            # v_coeffs: dict {index: coeff}
            # w_coeffs: dict {index: coeff}
            
            # T1: M1
            ({x_idx[0]: 1, x_idx[3]: 1}, {y_idx[0]: 1, y_idx[3]: 1}, {z_idx[0]: 1, z_idx[3]: 1}),
            # T2: M2
            ({x_idx[2]: 1, x_idx[3]: 1}, {y_idx[0]: 1}, {z_idx[2]: 1}),
            # T3: M3
            ({x_idx[0]: 1}, {y_idx[1]: 1, y_idx[3]: -1}, {z_idx[1]: 1, z_idx[3]: 1}),
            # T4: M4
            ({x_idx[3]: 1}, {y_idx[2]: 1, y_idx[0]: -1}, {z_idx[2]: 1}),
            # T5: M5
            ({x_idx[0]: 1, x_idx[1]: 1}, {y_idx[3]: 1}, {z_idx[0]: -1, z_idx[1]: 1}),
            # T6: M6
            ({x_idx[2]: 1, x_idx[0]: -1}, {y_idx[0]: 1, y_idx[1]: 1}, {z_idx[3]: 1}),
            # T7: M7
            ({x_idx[1]: 1, x_idx[3]: -1}, {y_idx[2]: 1, y_idx[3]: 1}, {z_idx[0]: 1})
        ]
        
        for t in terms:
            u_c, v_c, w_c = t
            for idx, c in u_c.items():
                U[idx][col] += c
            for idx, c in v_c.items():
                V[idx][col] += c
            for idx, c in w_c.items():
                W[idx][col] += c
            col += 1
            
    # Part 1: C[0..1, 0..1] = A[0..1, 0..1] B[0..1, 0..1] + A[0..1, 2..3] B[2..3, 0..1]
    # A11: U[0], U[1], U[4], U[5]
    # A12: U[2], U[3], U[6], U[7]
    # B11: V[0], V[1], V[4], V[5]
    # B21: V[8], V[9], V[12], V[13]
    # C11: W[0], W[1], W[2], W[3]
    
    # We need to combine A11 B11 and A12 B21.
    # This is not a single 2x2 multiplication.
    # It is a sum of two 2x2 multiplications.
    # We can just call add_strassen twice?
    # No, add_strassen computes XY.
    # We need to compute A11 B11 + A12 B21.
    # We can use the rank-14 decomposition of 2x4x2.
    # Which is just the sum of two rank-7 decompositions.
    # So we call add_strassen for A11 B11, then for A12 B21.
    # But they will occupy different columns.
    # Total 14 columns for this part.
    
    # A11 B11
    add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3], 0)
    # A12 B21
    add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3], 7)
    
    # Part 2: C[0..1, 2..3] = A[0..1, 0..1] B[0..1, 2..3] + A[0..1, 2..3] B[2..3, 2..3]
    # B12: V[2], V[3], V[6], V[7]
    # B22: V[10], V[11], V[14], V[15]
    # C12: W[4], W[5], W[6], W[7]
    
    # A11 B12
    add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7], 14)
    # A12 B22
    add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7], 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
