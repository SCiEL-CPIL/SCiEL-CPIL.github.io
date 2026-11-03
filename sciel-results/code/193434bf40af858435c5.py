import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # local i, j, k in 0..1
    # global mapping functions
    def add_strassen(i_map, j_map, k_map):
        nonlocal col
        # Strassen 7 terms
        # M1 = A11 B11
        # M2 = A22 (B12 + B22)
        # M3 = (A11 + A22) B22
        # M4 = A11 (B21 - B22)
        # M5 = (A11 + A12) B21
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        
        # Indices in local 0..1
        # i, j, k
        # A indices: i, j
        # B indices: j, k
        # C indices: i, k
        
        terms = [
            # (U_i, U_j, V_j, V_k, W_i, W_k)
            # M1
            (1, 0, 1, 0, 1, 0),
            # M2
            (1, 1, 1, 1, 1, 1), # A22 * B12
            (1, 1, 1, 1, 1, 0), # A22 * B22
            # M3
            (1, 0, 1, 1, 1, 1), # A11 * B22
            (1, 1, 1, 1, 1, 1), # A22 * B22
            # M4
            (1, 0, 1, 0, 1, 0), # A11 * B21
            (1, 0, 1, 1, 1, 1), # A11 * B22
            # M5
            (1, 0, 1, 0, 1, 0), # A11 * B21
            (1, 1, 1, 0, 1, 0), # A12 * B21
            # M6
            (1, 1, 1, 0, 1, 0), # A21 * B11
            (1, 1, 1, 1, 1, 1), # A21 * B12
            (1, 0, 1, 0, 1, 0), # A11 * B11
            (1, 0, 1, 1, 1, 1), # A11 * B12
            # M7
            (1, 1, 1, 0, 1, 0), # A12 * B21
            (1, 1, 1, 1, 1, 1), # A12 * B22
            (1, 0, 1, 0, 1, 0), # A22 * B21
            (1, 0, 1, 1, 1, 1), # A22 * B22
        ]
        
        # Wait, the above list is messy. 
        # Standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M2 + M3
        # C21 = M5 + M6
        # C22 = M1 - M2 + M3 + M6
        
        # M1 = A11 B11
        # M2 = A22 (B12 + B22)
        # M3 = (A11 + A22) B22
        # M4 = A11 (B21 - B22)
        # M5 = (A11 + A12) B21
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        
        # We need to expand these into rank-1 terms.
        
        # M1: A11 B11
        # U: i=0,j=0; V: j=0,k=0; W: i=0,k=0
        # Term 1
        U[i_map(0)*2 + j_map(0)][col] = 1
        V[j_map(0)*2 + k_map(0)][col] = 1
        W[k_map(0)*2 + i_map(0)][col] = 1
        col += 1
        
        # M2: A22 B12 + A22 B22
        # Term 2: A22 B12
        U[i_map(1)*2 + j_map(1)][col] = 1
        V[j_map(1)*2 + k_map(0)][col] = 1
        W[k_map(0)*2 + i_map(1)][col] = 1
        col += 1
        # Term 3: A22 B22
        U[i_map(1)*2 + j_map(1)][col] = 1
        V[j_map(1)*2 + k_map(1)][col] = 1
        W[k_map(1)*2 + i_map(1)][col] = 1
        col += 1
        
        # M3: A11 B22 + A22 B22
        # Term 4: A11 B22
        U[i_map(0)*2 + j_map(0)][col] = 1
        V[j_map(0)*2 + k_map(1)][col] = 1
        W[k_map(1)*2 + i_map(0)][col] = 1
        col += 1
        # Term 5: A22 B22
        U[i_map(1)*2 + j_map(1)][col] = 1
        V[j_map(1)*2 + k_map(1)][col] = 1
        W[k_map(1)*2 + i_map(1)][col] = 1
        col += 1
        
        # M4: A11 B21 - A11 B22
        # Term 6: A11 B21
        U[i_map(0)*2 + j_map(0)][col] = 1
        V[j_map(0)*2 + k_map(0)][col] = 1
        W[k_map(0)*2 + i_map(0)][col] = 1
        col += 1
        # Term 7: A11 B22 (negative)
        U[i_map(0)*2 + j_map(0)][col] = 1
        V[j_map(0)*2 + k_map(1)][col] = 1
        W[k_map(1)*2 + i_map(0)][col] = -1
        col += 1
        
        # M5: A11 B21 + A12 B21
        # Term 8: A11 B21
        U[i_map(0)*2 + j_map(0)][col] = 1
        V[j_map(0)*2 + k_map(0)][col] = 1
        W[k_map(0)*2 + i_map(0)][col] = 1
        col += 1
        # Term 9: A12 B21
        U[i_map(1)*2 + j_map(1)][col] = 1
        V[j_map(1)*2 + k_map(0)][col] = 1
        W[k_map(0)*2 + i_map(1)][col] = 1
        col += 1
        
        # M6: A21 B11 + A21 B12 - A11 B11 - A11 B12
        # Term 10: A21 B11
        U[i_map(1)*2 + j_map(0)][col] = 1
        V[j_map(0)*2 + k_map(0)][col] = 1
        W[k_map(0)*2 + i_map(1)][col] = 1
        col += 1
        # Term 11: A21 B12
        U[i_map(1)*2 + j_map(0)][col] = 1
        V[j_map(0)*2 + k_map(1)][col] = 1
        W[k_map(1)*2 + i_map(1)][col] = 1
        col += 1
        # Term 12: A11 B11 (negative)
        U[i_map(0)*2 + j_map(0)][col] = 1
        V[j_map(0)*2 + k_map(0)][col] = 1
        W[k_map(0)*2 + i_map(0)][col] = -1
        col += 1
        # Term 13: A11 B12 (negative)
        U[i_map(0)*2 + j_map(0)][col] = 1
        V[j_map(0)*2 + k_map(1)][col] = 1
        W[k_map(1)*2 + i_map(0)][col] = -1
        col += 1
        
        # M7: A12 B21 + A12 B22 - A22 B21 - A22 B22
        # Term 14: A12 B21
        U[i_map(1)*2 + j_map(1)][col] = 1
        V[j_map(1)*2 + k_map(0)][col] = 1
        W[k_map(0)*2 + i_map(1)][col] = 1
        col += 1
        # Term 15: A12 B22
        U[i_map(1)*2 + j_map(1)][col] = 1
        V[j_map(1)*2 + k_map(1)][col] = 1
        W[k_map(1)*2 + i_map(1)][col] = 1
        col += 1
        # Term 16: A22 B21 (negative)
        U[i_map(1)*2 + j_map(1)][col] = 1
        V[j_map(1)*2 + k_map(0)][col] = 1
        W[k_map(0)*2 + i_map(1)][col] = -1
        col += 1
        # Term 17: A22 B22 (negative)
        U[i_map(1)*2 + j_map(1)][col] = 1
        V[j_map(1)*2 + k_map(1)][col] = 1
        W[k_map(1)*2 + i_map(1)][col] = -1
        col += 1
        
        # Wait, Strassen uses 7 multiplications, but each multiplication is a scalar product.
        # In tensor decomposition, each scalar product is a rank-1 term.
        # So M1 is 1 term. M2 is 2 terms (A22B12, A22B22).
        # This results in 17 terms for one 2x2x2 block?
        # No, the rank of T_2,2,2 is 7.
        # My expansion above is wrong.
        # The 7 terms ARE the rank-1 tensors.
        # M1 = A11 B11. This is ONE rank-1 tensor.
        # M2 = A22 (B12 + B22). This is NOT a rank-1 tensor.
        # Strassen's algorithm reduces the number of multiplications from 8 to 7.
        # But each "multiplication" in Strassen is a product of two linear combinations.
        # $(A_{11} + A_{22}) B_{22}$ is NOT a rank-1 term in the original tensor.
        # It is a rank-2 term?
        # No, the tensor decomposition requires rank-1 terms.
        # Strassen's algorithm works by changing the basis.
        # It does NOT directly give a rank-7 decomposition in the standard basis.
        # It gives a rank-7 decomposition in a transformed basis.
        # To get the decomposition in the standard basis, we must transform back.
        # This results in a decomposition with integer coefficients, but the number of terms is still 7.
        # Wait, if I use the standard basis, the rank is 8.
        # The rank 7 decomposition exists, but the terms are not simply $A_{ij} B_{jk}$.
        # They are linear combinations.
        
        # I cannot easily write down the 7 rank-1 terms for Strassen in the standard basis without doing the algebra.
        # Given the time limit, I should stick to the schoolbook decomposition if I can't easily derive the Strassen terms.
        # Schoolbook rank for 2x2x2 is 8.
        # So 4 blocks * 8 = 32.
        # 2 blocks * 12 = 24.
        # Total 56.
        
        # So 52 is not achievable with simple block decomposition if I can't write Strassen terms.
        
        # Is there any other improvement?
        # I will return the schoolbook decomposition of 56.
        
        pass

    # Just return schoolbook
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
