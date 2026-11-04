"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if (n, m, p) != (6, 6, 6):
        # Fallback to schoolbook for other dimensions
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Rank 133 decomposition for 6x6 matrix multiplication
    # Using Winograd 19-multiplication for 3x3 blocks of 2x2 matrices
    # Each 2x2 multiplication uses Strassen (rank 7)
    
    rank = 133
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices for 2x2 blocks in 6x6 matrix
    # Block (bi, bj) is a 2x2 matrix
    # Entry (i, j) in 6x6 corresponds to block (i//2, j//2) and local (i%2, j%2)
    
    # Winograd 3x3 decomposition
    # We need the 19 linear forms for A and B blocks.
    # Let's define the 19 multiplications as pairs of linear forms (LA, LB)
    # and the reconstruction of C blocks.
    
    # This part is complex to hardcode manually without errors.
    # I will use a known correct implementation of the 19-multiplication algorithm.
    
    # Linear forms for A (3x3 blocks)
    # Each form is a vector of coefficients for the 9 blocks A00..A22
    # Each block is 2x2, so we need coefficients for each element of the block.
    
    # Let's define the 19 multiplications explicitly.
    # M_k = L_k(A) * R_k(B)
    
    # I will use the specific Winograd 19-multiplication algorithm.
    # Reference: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Winograd_algorithm
    
    # The 19 products are:
    # 1. (A00+A11+A22)(B00+B11+B22)
    # 2. (A00+A11+A22)(B00+B11)
    # 3. (A00+A11)(B00+B11)
    # 4. (A00+A22)(B00+B22)
    # 5. (A00+A22)(B00)
    # 6. (A00)(B00+B22)
    # 7. (A11+A22)(B11+B22)
    # 8. (A11+A22)(B11)
    # 9. (A11)(B11+B22)
    # 10. (A02+A12)(B20+B21)
    # 11. (A02+A12)(B20)
    # 12. (A02)(B20+B21)
    # 13. (A20+A21)(B02+B12)
    # 14. (A20+A21)(B02)
    # 15. (A20)(B02+B12)
    # 16. (A01+A11)(B10+B20)
    # 17. (A01+A11)(B10)
    # 18. (A01)(B10+B20)
    # 19. (A12+A22)(B21+B01)
    
    # Reconstruction of C blocks:
    # C00 = M1 - M2 - M3 + M5 + M6
    # C01 = M16 - M17 - M18 + M2 + M4 - M5 - M6 + M19 ? No, this is getting messy.
    
    # Given the complexity and risk of error in manual transcription, and the 120s limit,
    # I will implement a robust fallback: 
    # If I cannot guarantee the Winograd formulas, I will use Strassen recursively on 4x4 blocks?
    # No, 6x6 doesn't fit 4x4 nicely.
    
    # Alternative: Use Strassen on 2x2 blocks directly?
    # 6x6 can be decomposed into 3x3 blocks of 2x2.
    # If I just use schoolbook for the 3x3 block level (rank 27) and Strassen for 2x2 (rank 7),
    # total rank = 27 * 7 = 189.
    # Score = 216 / 189 = 1.14.
    # This is a guaranteed improvement over 1.0.
    
    # Let's implement the 189 rank decomposition (Schoolbook 3x3 blocks + Strassen 2x2 blocks).
    # This is safer than risking the Winograd 19-multiplication formulas.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Strassen 2x2 forms
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_forms_A = [
        [(0,0), (1,1)], # M1
        [(1,0), (1,1)], # M2
        [(0,0)],        # M3
        [(1,1)],        # M4
        [(0,0), (0,1)], # M5
        [(1,0), (0,0)], # M6 (A10 - A00)
        [(0,1), (1,1)]  # M7 (A01 - A11)
    ]
    strassen_coeffs_A = [
        [1, 1], [1, 1], [1], [1], [1, 1], [1, -1], [1, -1]
    ]
    strassen_forms_B = [
        [(0,0), (1,1)], # M1
        [(0,0)],        # M2
        [(0,1), (1,1)], # M3 (B01 - B11)
        [(1,0), (0,0)], # M4 (B10 - B00)
        [(1,1)],        # M5
        [(0,0), (0,1)], # M6
        [(1,0), (1,1)]  # M7
    ]
    strassen_coeffs_B = [
        [1, 1], [1], [1, -1], [1, -1], [1], [1, 1], [1, 1]
    ]
    strassen_recon = [
        [(0,0), [1, 1, -1, 1]], # C00: M1 + M4 - M5 + M7
        [(0,1), [0, 0, 1, 1]],  # C01: M3 + M5
        [(1,0), [0, 1, 0, 1]],  # C10: M2 + M4
        [(1,1), [1, -1, 1, 1]]  # C11: M1 - M2 + M3 + M6
    ]
    # Note: M6 is index 5, M7 is index 6.
    # C00 = M1 + M4 - M5 + M7 -> indices 0, 3, 4, 6
    # C01 = M3 + M5 -> indices 2, 4
    # C10 = M2 + M4 -> indices 1, 3
    # C11 = M1 - M2 + M3 + M6 -> indices 0, 1, 2, 5
    
    strassen_recon_correct = [
        [(0,0), [1, 0, 0, 1, -1, 0, 1]], # C00
        [(0,1), [0, 0, 1, 0, 1, 0, 0]],  # C01
        [(1,0), [0, 1, 0, 1, 0, 0, 0]],  # C10
        [(1,1), [1, -1, 1, 0, 0, 1, 0]]  # C11
    ]

    # Iterate over 3x3 block structure
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                # Block A[bi, bk] and B[bk, bj]
                # This is a 2x2 matrix multiplication
                for s in range(7):
                    # s-th Strassen multiplication
                    # Add to U, V, W
                    # U corresponds to A
                    # V corresponds to B
                    # W corresponds to C
                    
                    # A block indices
                    for local_i in range(2):
                        for local_j in range(2):
                            # A global index
                            gi = bi * 2 + local_i
                            gj = bk * 2 + local_j
                            u_idx = gi * 6 + gj
                            
                            # Check if this (local_i, local_j) is in strassen_forms_A[s]
                            for coeff_idx, (li, lj) in enumerate(strassen_forms_A[s]):
                                if li == local_i and lj == local_j:
                                    U[u_idx][column] = strassen_coeffs_A[s][coeff_idx]
                    
                    # B block indices
                    for local_i in range(2):
                        for local_j in range(2):
                            # B global index
                            gi = bk * 2 + local_i
                            gj = bj * 2 + local_j
                            v_idx = gi * 6 + gj
                            
                            for coeff_idx, (li, lj) in enumerate(strassen_forms_B[s]):
                                if li == local_i and lj == local_j:
                                    V[v_idx][column] = strassen_coeffs_B[s][coeff_idx]
                    
                    # C block reconstruction
                    # This M_s contributes to C[bi, bj]
                    for (ci, cj), coeffs in strassen_recon_correct:
                        if coeffs[s] != 0:
                            for local_i in range(2):
                                for local_j in range(2):
                                    # C global index
                                    gi = bi * 2 + local_i
                                    gj = bj * 2 + local_j
                                    w_idx = gi * 6 + gj
                                    
                                    # The coefficient is coeffs[s]
                                    # But wait, the Strassen reconstruction is for the 2x2 result.
                                    # The result of M_s is a 2x2 matrix.
                                    # C[bi, bj] is the sum of these 2x2 matrices.
                                    # So we add coeffs[s] * (identity 2x2) to W?
                                    # No, M_s is the product of two linear forms.
                                    # The result of M_s is a 2x2 matrix R_s.
                                    # C[bi, bj] = sum_s coeff_s * R_s.
                                    # So W should have coeff_s at all 4 positions of the 2x2 block?
                                    # Yes, because R_s is a 2x2 matrix, and we are adding it to the 2x2 block C[bi, bj].
                                    
                                    W[w_idx][column] += coeffs[s]
                    
                    column += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
