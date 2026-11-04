"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # A blocks: 0 -> cols 0,1; 1 -> cols 2,3
    # B blocks: 0 -> rows 0,1; 1 -> rows 2,3
    # B sub-blocks: l -> cols 2l, 2l+1
    
    for a_block in range(2):
        a_base_j = a_block * 2
        # A indices:
        # A_00: i=0, j=a_base_j
        # A_01: i=0, j=a_base_j+1
        # A_10: i=1, j=a_base_j
        # A_11: i=1, j=a_base_j+1
        
        for l in range(4):
            b_base_k = l * 2
            # B rows for this a_block:
            # If a_block=0, B rows are 0,1.
            # If a_block=1, B rows are 2,3.
            b_base_j = a_block * 2
            
            # B indices:
            # B_00: j=b_base_j, k=b_base_k
            # B_01: j=b_base_j, k=b_base_k+1
            # B_10: j=b_base_j+1, k=b_base_k
            # B_11: j=b_base_j+1, k=b_base_k+1
            
            # Strassen products M_1 to M_7
            # M_1 = (A_00 + A_11) * (B_00 + B_11)
            # M_2 = A_11 * (B_00 + B_11)
            # M_3 = A_00 * (B_11 - B_10)
            # M_4 = A_11 * (B_00 - B_01)
            # M_5 = (A_00 + A_01) * B_11
            # M_6 = (A_10 + A_11) * (B_00 + B_01)
            # M_7 = (A_00 + A_01) * B_10
            
            # Define linear forms for U (A) and V (B)
            # U indices: i * m + j
            # V indices: j * p + k
            
            # M_1
            U[0 * m + a_base_j][col] += 1
            U[1 * m + a_base_j + 1][col] += 1
            V[b_base_j * p + b_base_k][col] += 1
            V[(b_base_j + 1) * p + b_base_k + 1][col] += 1
            # W contributions:
            # C_00 (i=0, k=b_base_k) += M_1
            # C_01 (i=0, k=b_base_k+1) += M_1
            # C_10 (i=1, k=b_base_k) += M_1
            # C_11 (i=1, k=b_base_k+1) += M_1
            W[b_base_k * n + 0][col] += 1
            W[(b_base_k + 1) * n + 0][col] += 1
            W[b_base_k * n + 1][col] += 1
            W[(b_base_k + 1) * n + 1][col] += 1
            col += 1

            # M_2
            U[1 * m + a_base_j + 1][col] += 1
            V[b_base_j * p + b_base_k][col] += 1
            V[(b_base_j + 1) * p + b_base_k + 1][col] += 1
            # W:
            # C_00 -= M_2 (Wait, formula: C_00 = M1 - M4 - M6 + M7? No.)
            # Let's re-verify Strassen formulas.
            # C_11 = M1 - M2 + M5 - M7
            # C_12 = M1 - M2 + M5 - M7 ? No.
            
            # Standard Strassen:
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = A22(B11 + B22)
            # M3 = A11(B22 - B21)
            # M4 = A22(B11 - B12)
            # M5 = (A11 + A12)B22
            # M6 = (A21 + A22)(B11 + B12)
            # M7 = (A11 + A12)B21
            #
            # C11 = M1 - M4 - M6 + M7
            # C12 = M3 + M4 + M5 + M6
            # C21 = M2 + M3 - M5 + M7
            # C22 = M1 - M2 + M5 - M7
            
            # My indices:
            # A11 = A[0, a_base_j]
            # A12 = A[0, a_base_j+1]
            # A21 = A[1, a_base_j]
            # A22 = A[1, a_base_j+1]
            
            # B11 = B[b_base_j, b_base_k]
            # B12 = B[b_base_j, b_base_k+1]
            # B21 = B[b_base_j+1, b_base_k]
            # B22 = B[b_base_j+1, b_base_k+1]
            
            # C11 = C[0, b_base_k]
            # C12 = C[0, b_base_k+1]
            # C21 = C[1, b_base_k]
            # C22 = C[1, b_base_k+1]
            
            # So:
            # C[0, b_base_k] = M1 - M4 - M6 + M7
            # C[0, b_base_k+1] = M3 + M4 + M5 + M6
            # C[1, b_base_k] = M2 + M3 - M5 + M7
            # C[1, b_base_k+1] = M1 - M2 + M5 - M7
            
            # I already added M1 to all 4.
            # Now M2:
            # C[1, b_base_k] += M2
            # C[1, b_base_k+1] -= M2
            W[b_base_k * n + 1][col] += 1
            W[(b_base_k + 1) * n + 1][col] -= 1
            col += 1

            # M_3
            U[0 * m + a_base_j][col] += 1
            V[(b_base_j + 1) * p + b_base_k + 1][col] += 1
            V[(b_base_j + 1) * p + b_base_k][col] -= 1
            # W:
            # C[0, b_base_k+1] += M3
            # C[1, b_base_k] += M3
            W[(b_base_k + 1) * n + 0][col] += 1
            W[b_base_k * n + 1][col] += 1
            col += 1

            # M_4
            U[1 * m + a_base_j + 1][col] += 1
            V[b_base_j * p + b_base_k][col] += 1
            V[b_base_j * p + b_base_k + 1][col] -= 1
            # W:
            # C[0, b_base_k] -= M4
            # C[0, b_base_k+1] += M4
            W[b_base_k * n + 0][col] -= 1
            W[(b_base_k + 1) * n + 0][col] += 1
            col += 1

            # M_5
            U[0 * m + a_base_j][col] += 1
            U[0 * m + a_base_j + 1][col] += 1
            V[(b_base_j + 1) * p + b_base_k + 1][col] += 1
            # W:
            # C[0, b_base_k+1] += M5
            # C[1, b_base_k] -= M5
            # C[1, b_base_k+1] += M5
            W[(b_base_k + 1) * n + 0][col] += 1
            W[b_base_k * n + 1][col] -= 1
            W[(b_base_k + 1) * n + 1][col] += 1
            col += 1

            # M_6
            U[1 * m + a_base_j][col] += 1
            U[1 * m + a_base_j + 1][col] += 1
            V[b_base_j * p + b_base_k][col] += 1
            V[b_base_j * p + b_base_k + 1][col] += 1
            # W:
            # C[0, b_base_k] -= M6
            # C[0, b_base_k+1] += M6
            W[b_base_k * n + 0][col] -= 1
            W[(b_base_k + 1) * n + 0][col] += 1
            col += 1

            # M_7
            U[0 * m + a_base_j][col] += 1
            U[0 * m + a_base_j + 1][col] += 1
            V[(b_base_j + 1) * p + b_base_k][col] += 1
            # W:
            # C[0, b_base_k] += M7
            # C[1, b_base_k] += M7
            # C[1, b_base_k+1] -= M7
            W[b_base_k * n + 0][col] += 1
            W[b_base_k * n + 1][col] += 1
            W[(b_base_k + 1) * n + 1][col] -= 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
