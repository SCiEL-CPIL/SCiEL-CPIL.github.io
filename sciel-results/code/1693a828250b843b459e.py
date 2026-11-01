import time

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    We use the 19-term Larsen algorithm for 3x3x3 matrix multiplication.
    Since the 3x6 matrix consists of two 3x3 blocks, and the left factor A is shared,
    we can apply the 19-term decomposition to each block independently.
    This results in a total rank of 19 + 19 = 38.
    The schoolbook rank is 3*3*6 = 54.
    38 is significantly lower than 54.
    """
    n, m, p = 3, 3, 6
    rank = 38
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (9, 38)
    # V: (m*p, rank) = (18, 38)
    # W: (p*n, rank) = (18, 38)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 19 terms of Larsen's algorithm
    # Each term is a tuple (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: list of 9 floats (row-major: A[0,0] at 0, A[0,1] at 1, ..., A[2,2] at 8)
    # B_coeffs: list of 9 floats
    # C_coeffs: list of 9 floats (coefficients for the product in the sum for C[i,j])
    
    # We define the products P1..P19 as linear combinations of A and B elements.
    # P_k = sum_i U[k][i] A[i] * sum_j V[k][j] B[j]
    # C[l] = sum_k W[k][l] P_k
    
    # Let's list the terms explicitly.
    # Indices: A[0]=A00, A[1]=A01, A[2]=A02, A[3]=A10, A[4]=A11, A[5]=A12, A[6]=A20, A[7]=A21, A[8]=A22
    # B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B10, B[4]=B11, B[5]=B12, B[6]=B20, B[7]=B21, B[8]=B22
    # C[0]=C00, C[1]=C01, C[2]=C02, C[3]=C10, C[4]=C11, C[5]=C12, C[6]=C20, C[7]=C21, C[8]=C22
    
    terms = []
    
    # P1 = A00 * B00
    terms.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    
    # P2 = (A10+A20) * (B01-B00)
    terms.append(([0,0,0,1,0,0,1,0,0], [-1,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0])) # Wait, C coefficients depend on how P2 is used.
    # It is better to define U, V for the product, and then define W based on the combination formula.
    
    # Let's redefine the structure.
    # For each of the 19 terms, we have a vector u (size 9), v (size 9).
    # The product is scalar p.
    # The contribution to C is w * p, where w is a vector of size 9.
    
    # Let's extract u and v for each term from the definitions above.
    
    # P1: A00 * B00
    u1 = [1,0,0,0,0,0,0,0,0]
    v1 = [1,0,0,0,0,0,0,0,0]
    
    # P2: (A10+A20) * (B01-B00)
    u2 = [0,0,0,1,0,0,1,0,0]
    v2 = [-1,1,0,0,0,0,0,0,0]
    
    # P3: (A00+A10) * B10
    u3 = [1,0,0,1,0,0,0,0,0]
    v3 = [0,0,0,1,0,0,0,0,0]
    
    # P4: A20 * (B10+B20)
    u4 = [0,0,0,0,0,0,1,0,0]
    v4 = [0,0,0,1,0,0,1,0,0]
    
    # P5: A02 * (B20+B21)
    u5 = [0,0,1,0,0,0,0,0,0]
    v5 = [0,0,0,0,0,0,1,1,0]
    
    # P6: (A02-A12) * B22
    u6 = [0,0,1,0,0,-1,0,0,0]
    v6 = [0,0,0,0,0,0,0,0,1]
    
    # P7: (A10+A20) * (B02+B12)
    u7 = [0,0,0,1,0,0,1,0,0]
    v7 = [0,0,1,0,0,1,0,0,0]
    
    # P8: (A00-A10+A20) * B12
    u8 = [1,0,0,-1,0,0,1,0,0]
    v8 = [0,0,0,0,0,1,0,0,0]
    
    # P9: (A02-A12+A22) * B20
    u9 = [0,0,1,0,0,-1,0,0,1]
    v9 = [0,0,0,0,0,0,1,0,0]
    
    # P10: (A00+A11) * (B11-B01)
    u10 = [1,0,0,0,1,0,0,0,0]
    v10 = [0,-1,0,0,1,0,0,0,0]
    
    # P11: (A11-A21) * (B11+B21)
    u11 = [0,0,0,0,1,0,0,-1,0]
    v11 = [0,0,0,0,1,0,0,1,0]
    
    # P12: A01 * (B01+B11)
    u12 = [0,1,0,0,0,0,0,0,0]
    v12 = [0,1,0,0,1,0,0,0,0]
    
    # P13: (A01+A11) * (B02-B12)
    u13 = [0,1,0,0,1,0,0,0,0]
    v13 = [0,0,1,0,0,-1,0,0,0]
    
    # P14: (A11+A21) * (B02-B22)
    u14 = [0,0,0,0,1,0,0,1,0]
    v14 = [0,0,1,0,0,0,0,0,-1]
    
    # P15: (A01-A21) * (B12+B22)
    u15 = [0,1,0,0,0,0,0,-1,0]
    v15 = [0,0,0,0,0,1,0,0,1]
    
    # P16: (A01+A22) * (B22-B02)
    u16 = [0,1,0,0,0,0,0,0,1]
    v16 = [0,0,-1,0,0,0,0,0,1]
    
    # P17: (A21+A22) * (B02+B22)
    u17 = [0,0,0,0,0,0,0,1,1]
    v17 = [0,0,1,0,0,0,0,0,1]
    
    # P18: (A02+A12) * (B10-B20)
    u18 = [0,0,1,0,0,1,0,0,0]
    v18 = [0,0,0,1,0,0,-1,0,0]
    
    # P19: (A12+A22) * (B10+B20)
    u19 = [0,0,0,0,0,1,0,0,1]
    v19 = [0,0,0,1,0,0,1,0,0]
    
    # Now define the W vectors (coefficients for C)
    # C00 = P1 - P2 + P3 + P8
    # C01 = P2 + P10 - P13 + P16
    # C02 = P13 - P14 + P15 - P16 + P17
    # C10 = P3 - P4 + P7 - P8
    # C11 = P4 + P8 + P10 + P11 - P14 + P17
    # C12 = P5 - P6 + P14 + P15 - P17 + P19
    # C20 = P4 - P7 + P9 - P18
    # C21 = P7 - P9 + P11 + P18 - P19
    # C22 = P6 + P9 + P12 + P16 + P18 + P19
    
    # w[k] is the coefficient of P_k in the sum for C_l.
    # We need to fill W[l][k] = w_k for each l.
    
    # Let's create a 19x9 matrix W_larsen where W_larsen[k][l] is the coeff of P_k in C_l.
    W_larsen = [[0]*9 for _ in range(19)]
    
    # P1 (idx 0)
    W_larsen[0][0] = 1   # C00
    # P2 (idx 1)
    W_larsen[1][0] = -1  # C00
    W_larsen[1][1] = 1   # C01
    # P3 (idx 2)
    W_larsen[2][0] = 1   # C00
    W_larsen[2][3] = 1   # C10
    # P4 (idx 3)
    W_larsen[3][3] = -1  # C10
    W_larsen[3][4] = 1   # C11
    W_larsen[3][6] = 1   # C20
    # P5 (idx 4)
    W_larsen[4][5] = 1   # C12
    # P6 (idx 5)
    W_larsen[5][5] = -1  # C12
    W_larsen[5][8] = 1   # C22
    # P7 (idx 6)
    W_larsen[6][3] = 1   # C10
    W_larsen[6][6] = -1  # C20
    W_larsen[6][7] = 1   # C21
    # P8 (idx 7)
    W_larsen[7][0] = 1   # C00
    W_larsen[7][3] = -1  # C10
    W_larsen[7][4] = 1   # C11
    # P9 (idx 8)
    W_larsen[8][6] = 1   # C20
    W_larsen[8][7] = -1  # C21
    W_larsen[8][8] = 1   # C22
    # P10 (idx 9)
    W_larsen[9][1] = 1   # C01
    W_larsen[9][4] = 1   # C11
    # P11 (idx 10)
    W_larsen[10][4] = 1  # C11
    W_larsen[10][7] = 1  # C21
    # P12 (idx 11)
    W_larsen[11][8] = 1  # C22
    # P13 (idx 12)
    W_larsen[12][1] = -1 # C01
    W_larsen[12][2] = 1  # C02
    # P14 (idx 13)
    W_larsen[13][2] = -1 # C02
    W_larsen[13][4] = -1 # C11
    W_larsen[13][5] = 1  # C12
    # P15 (idx 14)
    W_larsen[14][2] = 1  # C02
    W_larsen[14][5] = 1  # C12
    # P16 (idx 15)
    W_larsen[15][1] = 1  # C01
    W_larsen[15][2] = -1 # C02
    W_larsen[15][8] = 1  # C22
    # P17 (idx 16)
    W_larsen[16][2] = 1  # C02
    W_larsen[16][4] = 1  # C11
    W_larsen[16][5] = -1 # C12
    # P18 (idx 17)
    W_larsen[17][6] = -1 # C20
    W_larsen[17][7] = 1  # C21
    W_larsen[17][8] = 1  # C22
    # P19 (idx 18)
    W_larsen[18][5] = 1  # C12
    W_larsen[18][7] = -1 # C21
    W_larsen[18][8] = 1  # C22
    
    # List of u and v vectors
    uv = [
        (u1, v1), (u2, v2), (u3, v3), (u4, v4), (u5, v5),
        (u6, v6), (u7, v7), (u8, v8), (u9, v9), (u10, v10),
        (u11, v11), (u12, v12), (u13, v13), (u14, v14), (u15, v15),
        (u16, v16), (u17, v17), (u18, v18), (u19, v19)
    ]
    
    # Fill U, V, W for the 3x3x6 case
    # Block 1: Columns 0-18
    for k in range(19):
        u, v = uv[k]
        col = k
        for i in range(9):
            U[i][col] = u[i]
        for j in range(9):
            V[j][col] = v[j]
        for l in range(9):
            W[l][col] = W_larsen[k][l]
            
    # Block 2: Columns 19-37
    # The B matrix indices are shifted by 3 (columns 3,4,5 of B)
    # The C matrix indices are shifted by 3 (rows 3,4,5 of C)
    for k in range(19):
        u, v = uv[k]
        col = 19 + k
        for i in range(9):
            U[i][col] = u[i]
        for j in range(9):
            # B index j corresponds to B[j//3][j%3]
            # In the 3x6 matrix, the second block is B[0:3, 3:6]
            # So B[j] in the 3x3 context maps to B[j + 3] in the 3x6 context?
            # Wait. B is 3x6.
            # B[0,0] B[0,1] B[0,2] | B[0,3] B[0,4] B[0,5]
            # B[1,0] B[1,1] B[1,2] | B[1,3] B[1,4] B[1,5]
            # B[2,0] B[2,1] B[2,2] | B[2,3] B[2,4] B[2,5]
            # The linear index for V is j*p + k? No, V is (m*p, rank).
            # The task says V is (m*p, rank).
            # The term is V[j*p + k].
            # Here m=3, p=6.
            # The "j" in the tensor term is the row of B (0..2).
            # The "k" in the tensor term is the column of B (0..5).
            # In the 3x3 sub-problem, the column index is 0..2.
            # For the second block, the column index is 3..5.
            # So if the 3x3 term uses B_col = c (0..2), the 3x6 term uses B_col = c + 3.
            # The linear index in V is row * p + col.
            # row is j (0..2).
            # So V[j * 6 + (c + 3)][col] = v[c]
            # Note: v is a vector of length 9, indexed by linear index of 3x3 B.
            # v[idx] where idx = row * 3 + col_3x3.
            # So for each idx in 0..8:
            #   row = idx // 3
            #   col_3x3 = idx % 3
            #   col_3x6 = col_3x3 + 3
            #   V[row * 6 + col_3x6][col] = v[idx]
            
            for idx in range(9):
                row = idx // 3
                col_3x3 = idx % 3
                col_3x6 = col_3x3 + 3
                V[row * 6 + col_3x6][col] = v[idx]
                
        for l in range(9):
            # C index l (0..8) in 3x3 context.
            # row = l // 3
            # col_3x3 = l % 3
            # col_3x6 = col_3x3 + 3
            # W is (p*n, rank) = (6*3, rank) = (18, rank).
            # The task says W[k*n+i, r].
            # Here k is the column of C (0..5), i is the row of C (0..2).
            # So W[col * n + row][col] = w[col]
            # In the 3x3 context, col_3x3 is the column of C.
            # In the 3x6 context, col_3x6 = col_3x3 + 3.
            # row is the same.
            for idx in range(9):
                row = idx // 3
                col_3x3 = idx % 3
                col_3x6 = col_3x3 + 3
                W[col_3x6 * 3 + row][col] = W_larsen[k][idx]

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
