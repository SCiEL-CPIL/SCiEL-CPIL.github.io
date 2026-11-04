import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for 3x5x5 matrix multiplication.
    Uses a block decomposition strategy:
    - 3x3x3 sub-block uses a rank-26 decomposition (Strassen on 2x2 + schoolbook cross terms).
    - Remaining blocks use schoolbook decomposition.
    Total rank: 26 + 18 + 18 + 12 = 74.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # rank = n * m * p  # Schoolbook rank = 75
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    # u_idx: index in U (n*m)
    # v_idx: index in V (m*p)
    # w_idx: index in W (p*n)
    def add_term(u_idx, v_idx, w_idx):
        U.append([0] * (n * m))
        V.append([0] * (m * p))
        W.append([0] * (p * n))
        U[-1][u_idx] = 1
        V[-1][v_idx] = 1
        W[-1][w_idx] = 1

    # Helper to add a scaled rank-1 term
    def add_term_scaled(u_idx, v_idx, w_idx, val):
        U.append([0] * (n * m))
        V.append([0] * (m * p))
        W.append([0] * (p * n))
        U[-1][u_idx] = val
        V[-1][v_idx] = 1
        W[-1][w_idx] = 1

    # 1. 3x3x3 Block: A[0:3, 0:3] * B[0:3, 0:3]
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # Strassen on 2x2 part (i,j,k in 0..1)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 -= M1? No, standard Strassen formulas.
    # Let's just use the explicit rank-1 terms for the 26 rank decomposition.
    
    # Strassen terms for C[0:2, 0:2]
    # M1 = (A00+A11)(B00+B11)
    # This corresponds to:
    # (A00+A11) is a linear form. (B00+B11) is a linear form.
    # Product contributes to C00 and C11.
    # C00 += M1, C11 += M1? 
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor decomposition.
    # Each M is a product of a linear form in A and a linear form in B.
    # We add rank-1 terms for each M, weighted by the coefficients in C.
    
    # M1: A form: A00+A11, B form: B00+B11
    # Coeffs: C00: +1, C11: +1 (from M1 in C11? No, C11 = M1 - M2 + M3 + M6. So +1 for M1)
    # Wait, C11 formula: M1 - M2 + M3 + M6.
    # So M1 contributes +1 to C00 and +1 to C11.
    
    # We will add terms for each M.
    # Term for M1:
    # U: 1 at (0,0), 1 at (1,1)
    # V: 1 at (0,0), 1 at (1,1)
    # W: 1 at (0,0) [for C00], 1 at (1,1) [for C11]
    # This is not a single rank-1 term in the tensor sense if it contributes to multiple C entries with different weights?
    # Yes, it is. The tensor entry T[i,j,k] is the coefficient.
    # T[0,0,0] += 1 (from A00 B00)
    # T[0,0,1] += 1 (from A00 B11)
    # T[1,0,0] += 1 (from A11 B00)
    # T[1,0,1] += 1 (from A11 B11)
    # And similarly for C11 contributions.
    
    # Actually, it is easier to just iterate over all pairs (i,j) in A and (j,k) in B?
    # No, that's schoolbook.
    # The Strassen algorithm groups these.
    # I will construct the 26 terms explicitly.
    
    # List of (U_indices, V_indices, W_indices, Coeff)
    # U_indices: list of (i,j) with coeff
    # V_indices: list of (j,k) with coeff
    # W_indices: list of (k,i) with coeff? No, W is (k*n+i).
    # The term is sum_{i,j} u_{ij} A_{ij} * sum_{j,k} v_{jk} B_{jk} * sum_{k,i} w_{ki} C_{ki}?
    # No, the decomposition is C = sum_r <U_r, A> <V_r, B> W_r.
    # So for each rank-1 term r, we have:
    # U_r is a matrix (n x m), V_r is a matrix (m x p), W_r is a matrix (n x p) (stored as p*n vector).
    # C += <U_r, A> * <V_r, B> * W_r.
    
    # Let's define the 7 Strassen matrices U_r, V_r, W_r for the 2x2 block.
    # Then add the 19 schoolbook terms for the rest.
    
    # 2x2 Strassen
    # M1: U: [[1,0],[1,1]]? No.
    # U1 = [[1,0],[0,1]]? No.
    # A form: A00+A11. U1 = [[1,0],[0,1]].
    # B form: B00+B11. V1 = [[1,0],[0,1]].
    # W1: Contributes to C00 and C11.
    # C00 += 1 * M1. C11 += 1 * M1.
    # So W1 = [[1,0],[0,1]].
    
    # M2: A form: A10+A11. U2 = [[0,0],[1,1]].
    # B form: B00. V2 = [[1,0],[0,0]].
    # C10 += 1 * M2.
    # W2 = [[0,0],[1,0]].
    
    # M3: A form: A00. U3 = [[1,0],[0,0]].
    # B form: B10-B11. V3 = [[0,0],[1,-1]].
    # C01 += 1 * M3.
    # W3 = [[0,1],[0,0]].
    
    # M4: A form: A11. U4 = [[0,0],[0,1]].
    # B form: B01-B00. V4 = [[-1,1],[0,0]].
    # C00 += 1 * M4. C10 += 1 * M4.
    # W4 = [[1,0],[1,0]].
    
    # M5: A form: A00+A01. U5 = [[1,1],[0,0]].
    # B form: B11. V5 = [[0,0],[0,1]].
    # C00 -= 1 * M5. C01 += 1 * M5.
    # W5 = [[-1,1],[0,0]].
    
    # M6: A form: A10-A00. U6 = [[-1,0],[1,0]].
    # B form: B00+B01. V6 = [[1,1],[0,0]].
    # C11 += 1 * M6.
    # W6 = [[0,0],[0,1]].
    
    # M7: A form: A01-A11. U7 = [[0,1],[-1,1]].
    # B form: B10+B11. V7 = [[0,0],[1,1]].
    # C00 += 1 * M7.
    # W7 = [[1,0],[0,0]].
    
    # These 7 terms cover C[0:2, 0:2].
    # Now add the cross terms for C[0:2, 0:2] from A[0:2, 2] and B[2, 0:2].
    # C00 += A02 B20.
    # C01 += A02 B21.
    # C10 += A12 B20.
    # C11 += A12 B21.
    
    # Term 8: A02 B20 -> C00.
    # U8 = [[0,0,0],[0,0,0]]? No, 3x3.
    # U8[0,2] = 1.
    # V8[2,0] = 1.
    # W8[0,0] = 1.
    
    # Term 9: A02 B21 -> C01.
    # U9[0,2] = 1.
    # V9[2,1] = 1.
    # W9[0,1] = 1.
    
    # Term 10: A12 B20 -> C10.
    # U10[1,2] = 1.
    # V10[2,0] = 1.
    # W10[1,0] = 1.
    
    # Term 11: A12 B21 -> C11.
    # U11[1,2] = 1.
    # V11[2,1] = 1.
    # W11[1,1] = 1.
    
    # Now C[0:2, 2] (Column 2 of top 2 rows)
    # C02 = A00 B02 + A01 B12 + A02 B22
    # C12 = A10 B02 + A11 B12 + A12 B22
    
    # Term 12: A00 B02 -> C02.
    # U12[0,0]=1, V12[0,2]=1, W12[0,2]=1.
    
    # Term 13: A01 B12 -> C02.
    # U13[0,1]=1, V13[1,2]=1, W13[0,2]=1.
    
    # Term 14: A02 B22 -> C02.
    # U14[0,2]=1, V14[2,2]=1, W14[0,2]=1.
    
    # Term 15: A10 B02 -> C12.
    # U15[1,0]=1, V15[0,2]=1, W15[1,2]=1.
    
    # Term 16: A11 B12 -> C12.
    # U16[1,1]=1, V16[1,2]=1, W16[1,2]=1.
    
    # Term 17: A12 B22 -> C12.
    # U17[1,2]=1, V17[2,2]=1, W17[1,2]=1.
    
    # Now C[2, 0:2] (Row 2 of left 2 cols)
    # C20 = A20 B00 + A21 B10 + A22 B20
    # C21 = A20 B01 + A21 B11 + A22 B21
    
    # Term 18: A20 B00 -> C20.
    # U18[2,0]=1, V18[0,0]=1, W18[2,0]=1.
    
    # Term 19: A21 B10 -> C20.
    # U19[2,1]=1, V19[1,0]=1, W19[2,0]=1.
    
    # Term 20: A22 B20 -> C20.
    # U20[2,2]=1, V20[2,0]=1, W20[2,0]=1.
    
    # Term 21: A20 B01 -> C21.
    # U21[2,0]=1, V21[0,1]=1, W21[2,1]=1.
    
    # Term 22: A21 B11 -> C21.
    # U22[2,1]=1, V22[1,1]=1, W22[2,1]=1.
    
    # Term 23: A22 B21 -> C21.
    # U23[2,2]=1, V23[2,1]=1, W23[2,1]=1.
    
    # Now C[2, 2]
    # C22 = A20 B02 + A21 B12 + A22 B22
    
    # Term 24: A20 B02 -> C22.
    # U24[2,0]=1, V24[0,2]=1, W24[2,2]=1.
    
    # Term 25: A21 B12 -> C22.
    # U25[2,1]=1, V25[1,2]=1, W25[2,2]=1.
    
    # Term 26: A22 B22 -> C22.
    # U26[2,2]=1, V26[2,2]=1, W26[2,2]=1.
    
    # Total 26 terms for 3x3x3.
    
    # We will add these 26 terms to the global U, V, W.
    # Then add schoolbook for the other blocks.
    
    # Global dimensions: n=3, m=5, p=5.
    # U shape: 15 x rank
    # V shape: 25 x rank
    # W shape: 15 x rank
    
    # We will build the lists of terms (u_idx, v_idx, w_idx, coeff)
    # and then assemble the matrices.
    
    terms = []
    
    # Add 3x3x3 terms (mapped to global indices)
    # Global indices:
    # A: 3x5. i in 0..2, j in 0..4.
    # B: 5x5. j in 0..4, k in 0..4.
    # C: 3x5. i in 0..2, k in 0..4.
    # W index: k*n + i = k*3 + i.
    
    # The 3x3x3 block uses A[0:3, 0:3] and B[0:3, 0:3].
    # So local i,j,k map to global i,j,k directly.
    
    # Term 1 (M1)
    # U: (0,0), (1,1)
    # V: (0,0), (1,1)
    # W: (0,0), (1,1)
    # This is not a single rank-1 term in the global sense?
    # Yes it is.
    # U_vec: 1 at 0*5+0, 1 at 1*5+1.
    # V_vec: 1 at 0*5+0, 1 at 1*5+1.
    # W_vec: 1 at 0*3+0, 1 at 1*3+1.
    # But wait, M1 contributes to C00 and C11.
    # The formula is C += <U, A> <V, B> W.
    # If W has 1s at (0,0) and (1,1), then C00 and C11 are both updated by the same product.
    # This is correct for M1.
    
    # I will define the 26 terms as (list_of_U_indices, list_of_V_indices, list_of_W_indices)
    # and create a rank-1 term for each.
    
    # Note: Some Strassen terms have negative coefficients.
    # We can absorb the sign into U, V, or W.
    
    # Let's define the 26 terms explicitly.
    
    # Helper to add a term from local 3x3x3 indices
    def add_3x3_term(u_locals, v_locals, w_locals):
        # u_locals: list of (i, j, coeff)
        # v_locals: list of (j, k, coeff)
        # w_locals: list of (k, i, coeff) -> W index is k*3+i
        u_vec = [0] * 15
        v_vec = [0] * 25
        w_vec = [0] * 15
        
        for i, j, c in u_locals:
            u_vec[i * 5 + j] = c
        for j, k, c in v_locals:
            v_vec[j * 5 + k] = c
        for k, i, c in w_locals:
            w_vec[k * 3 + i] = c
            
        terms.append((u_vec, v_vec, w_vec))

    # 1. M1
    add_3x3_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    # 2. M2
    add_3x3_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)])
    # 3. M3
    add_3x3_term([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1)])
    # 4. M4
    add_3x3_term([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)])
    # 5. M5
    add_3x3_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
    # 6. M6
    add_3x3_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
    # 7. M7
    add_3x3_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    
    # 8. A02 B20 -> C00
    add_3x3_term([(0,2,1)], [(2,0,1)], [(0,0,1)])
    # 9. A02 B21 -> C01
    add_3x3_term([(0,2,1)], [(2,1,1)], [(0,1,1)])
    # 10. A12 B20 -> C10
    add_3x3_term([(1,2,1)], [(2,0,1)], [(1,0,1)])
    # 11. A12 B21 -> C11
    add_3x3_term([(1,2,1)], [(2,1,1)], [(1,1,1)])
    
    # 12. A00 B02 -> C02
    add_3x3_term([(0,0,1)], [(0,2,1)], [(0,2,1)])
    # 13. A01 B12 -> C02
    add_3x3_term([(0,1,1)], [(1,2,1)], [(0,2,1)])
    # 14. A02 B22 -> C02
    add_3x3_term([(0,2,1)], [(2,2,1)], [(0,2,1)])
    
    # 15. A10 B02 -> C12
    add_3x3_term([(1,0,1)], [(0,2,1)], [(1,2,1)])
    # 16. A11 B12 -> C12
    add_3x3_term([(1,1,1)], [(1,2,1)], [(1,2,1)])
    # 17. A12 B22 -> C12
    add_3x3_term([(1,2,1)], [(2,2,1)], [(1,2,1)])
    
    # 18. A20 B00 -> C20
    add_3x3_term([(2,0,1)], [(0,0,1)], [(2,0,1)])
    # 19. A21 B10 -> C20
    add_3x3_term([(2,1,1)], [(1,0,1)], [(2,0,1)])
    # 20. A22 B20 -> C20
    add_3x3_term([(2,2,1)], [(2,0,1)], [(2,0,1)])
    
    # 21. A20 B01 -> C21
    add_3x3_term([(2,0,1)], [(0,1,1)], [(2,1,1)])
    # 22. A21 B11 -> C21
    add_3x3_term([(2,1,1)], [(1,1,1)], [(2,1,1)])
    # 23. A22 B21 -> C21
    add_3x3_term([(2,2,1)], [(2,1,1)], [(2,1,1)])
    
    # 24. A20 B02 -> C22
    add_3x3_term([(2,0,1)], [(0,2,1)], [(2,2,1)])
    # 25. A21 B12 -> C22
    add_3x3_term([(2,1,1)], [(1,2,1)], [(2,2,1)])
    # 26. A22 B22 -> C22
    add_3x3_term([(2,2,1)], [(2,2,1)], [(2,2,1)])
    
    # Now add Schoolbook for the other blocks.
    # Block 2: A[0:3, 3:5] B[3:5, 0:3]
    # i in 0..2, j in 3..4, k in 0..2
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                u_vec = [0] * 15
                v_vec = [0] * 25
                w_vec = [0] * 15
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 3 + i] = 1
                terms.append((u_vec, v_vec, w_vec))
                
    # Block 3: A[0:3, 0:3] B[0:3, 3:5]
    # i in 0..2, j in 0..2, k in 3..4
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                u_vec = [0] * 15
                v_vec = [0] * 25
                w_vec = [0] * 15
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 3 + i] = 1
                terms.append((u_vec, v_vec, w_vec))
                
    # Block 4: A[0:3, 3:5] B[3:5, 3:5]
    # i in 0..2, j in 3..4, k in 3..4
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 5):
                u_vec = [0] * 15
                v_vec = [0] * 25
                w_vec = [0] * 15
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 3 + i] = 1
                terms.append((u_vec, v_vec, w_vec))
                
    # Assemble U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(15)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(15)]
    
    for r, (u_vec, v_vec, w_vec) in enumerate(terms):
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[idx][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
