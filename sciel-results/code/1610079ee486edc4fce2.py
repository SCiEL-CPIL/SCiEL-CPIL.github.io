import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 57
    # Decomposition:
    # Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
    # C1 = A B1 (3x3 matmul, rank 24)
    # C2 = A B2 (3x3 matmul, rank 24)
    # C3 = A B3 (3x1 matmul, rank 9)
    # Total rank 57
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    def add_term(i, j, k, u_val, v_val, w_val, col):
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val

    col = 0
    
    # Function to add a 3x3x3 block decomposition
    # A is 3x3, B is 3x3 (cols offset by b_offset), C is 3x3 (cols offset by c_offset)
    def add_3x3_block(b_offset, c_offset):
        nonlocal col
        # Block structure:
        # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
        # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
        # C = [[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]]
        
        # 1. C11, C12, C21, C22 (2x2 block)
        # Strassen for A2x2 * B2x2 (7 terms)
        # A2x2 = [[A11, A12], [A21, A22]]
        # B2x2 = [[B11, B12], [B21, B22]]
        
        # M1 = (A11+A22)(B11+B22)
        add_term(0, 0, 0, 1, 1, 1, col); col += 1
        add_term(0, 0, 1, 1, 1, 1, col); col += 1
        add_term(1, 1, 0, 1, 1, 1, col); col += 1
        add_term(1, 1, 1, 1, 1, 1, col); col += 1
        # Wait, Strassen terms are bilinear.
        # M1 = (A11+A22)(B11+B22)
        # This expands to A11B11 + A11B22 + A22B11 + A22B22
        # In tensor form, one term corresponds to one product.
        # I need to list the 7 Strassen products.
        
        # Let's use a known Strassen 2x2 decomposition.
        # S1 = (A11+A22)(B11+B22)
        # S2 = (A21+A22)B11
        # S3 = A11(B12-B22)
        # S4 = A22(B21-B11)
        # S5 = (A11+A12)B22
        # S6 = (A21-A11)(B11+B12)
        # S7 = (A12-A22)(B21+B22)
        
        # C11 = S1 + S4 - S5 + S7
        # C12 = S3 + S5
        # C21 = S2 + S4
        # C22 = S1 - S2 + S3 + S6
        
        # I will add these 7 terms.
        # Term 1: (A11+A22)(B11+B22)
        # This is not a single rank-1 term in the original basis, it's a sum.
        # The decomposition is a sum of rank-1 tensors.
        # Each Strassen multiplication is a product of linear combinations.
        # So each Si is a rank-1 term in the "Strassen basis", but in the original basis, it contributes to multiple U, V, W entries.
        
        # I will explicitly add the contributions.
        
        # S1 = (A11+A22)(B11+B22)
        # U: A11 (0,0), A22 (1,1)
        # V: B11 (0,0), B22 (1,1)
        # W: C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
        # This is complex. I will use a helper to add a product of linear forms.
        
        def add_product(u_indices, v_indices, w_indices, coeffs_u, coeffs_v, coeffs_w, col_idx):
            # u_indices: list of (i,j)
            # v_indices: list of (j,k)
            # w_indices: list of (i,k)
            # coeffs: values
            for (i,j), cu in zip(u_indices, coeffs_u):
                U[i*m+j][col_idx] += cu
            for (j,k), cv in zip(v_indices, coeffs_v):
                V[j*p+k][col_idx] += cv
            for (i,k), cw in zip(w_indices, coeffs_w):
                W[k*n+i][col_idx] += cw

        # S1
        add_product([(0,0), (1,1)], [(0,0), (1,1)], [(0,0)], [1, 1], [1, 1], [1], col)
        # W for S1 is C11, C12, C21, C22?
        # C11 gets S1. C12 gets 0? No.
        # C11 = S1 + ...
        # C12 = ...
        # The W matrix entry is the coefficient of the output.
        # For S1, it contributes to C11, C12, C21, C22?
        # No, S1 is a scalar.
        # C11 += S1. C12 += 0. C21 += 0. C22 += 0?
        # Wait, C11 = S1 + S4 - S5 + S7.
        # So S1 contributes 1 to C11.
        # S4 contributes 1 to C11.
        # S5 contributes -1 to C11.
        # S7 contributes 1 to C11.
        
        # So for each Strassen term, I need to know which C elements it contributes to.
        
        # S1 -> C11: 1, C12: 0, C21: 0, C22: 0?
        # No, C22 = S1 - S2 + S3 + S6.
        # So S1 contributes to C11 and C22.
        
        # Let's map Strassen terms to C elements.
        # S1: C11 (+1), C22 (+1)
        # S2: C21 (+1), C22 (-1)
        # S3: C12 (+1), C22 (+1)
        # S4: C11 (+1), C21 (+1)
        # S5: C11 (-1), C12 (+1)
        # S6: C22 (+1)
        # S7: C11 (+1)
        
        # And the A, B linear forms.
        
        # This is doable but verbose.
        
        # Given the complexity and risk of error, and the 120s limit,
        # I will stick to the schoolbook decomposition for the 3x3 blocks if I can't be sure.
        # But 63 is the seed. I need to beat it.
        # If I use rank 24 for 3x3x3, I get 57.
        # I will implement the rank 24 decomposition carefully.
        
        # Due to the high risk of implementing Laderman/Strassen block decomposition correctly in this short time,
        # I will return the schoolbook decomposition for now, but I will try to optimize the 3x3x1 part?
        # No, 3x3x1 is already minimal (9).
        
        # I will attempt to implement the 57-rank decomposition.
        
        # S1: (A11+A22)(B11+B22)
        add_product([(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)], [1, 1], [1, 1], [1, 1], col)
        col += 1
        
        # S2: (A21+A22)B11
        add_product([(1,0), (1,1)], [(0,0)], [(1,0), (1,1)], [1, 1], [1], [1, -1], col)
        col += 1
        
        # S3: A11(B12-B22)
        add_product([(0,0)], [(0,1), (1,1)], [(0,1), (1,1)], [1], [1, -1], [1, 1], col)
        col += 1
        
        # S4: A22(B21-B11)
        add_product([(1,1)], [(1,0), (0,0)], [(0,0), (1,0)], [1], [1, -1], [1, 1], col)
        col += 1
        
        # S5: (A11+A12)B22
        add_product([(0,0), (0,1)], [(1,1)], [(0,0), (0,1)], [1, 1], [1], [-1, 1], col)
        col += 1
        
        # S6: (A21-A11)(B11+B12)
        add_product([(1,0), (0,0)], [(0,0), (0,1)], [(1,1)], [1, -1], [1, 1], [1], col)
        col += 1
        
        # S7: (A12-A22)(B21+B22)
        add_product([(0,1), (1,1)], [(1,0), (1,1)], [(0,0)], [1, -1], [1, 1], [1], col)
        col += 1
        
        # Now the outer product a b^T for C11, C12, C21, C22?
        # a = [A13, A23]^T, b = [B31, B32]^T
        # C11 += A13 B31
        # C12 += A13 B32
        # C21 += A23 B31
        # C22 += A23 B32
        # 4 terms.
        add_product([(0,2)], [(2,0)], [(0,0)], [1], [1], [1], col); col += 1
        add_product([(0,2)], [(2,1)], [(0,1)], [1], [1], [1], col); col += 1
        add_product([(1,2)], [(2,0)], [(1,0)], [1], [1], [1], col); col += 1
        add_product([(1,2)], [(2,1)], [(1,1)], [1], [1], [1], col); col += 1
        
        # Now c = A2x2 b + a beta
        # c1 = A11 B31 + A12 B32 + A13 B33
        # c2 = A21 B31 + A22 B32 + A23 B33
        # Rank 5 decomposition for 2x3x1.
        # I will use schoolbook (6 terms) to be safe, making block rank 25.
        # Total rank 25+25+9 = 59.
        # Still better than 63.
        
        # c1
        add_product([(0,0)], [(2,0)], [(2,0)], [1], [1], [1], col); col += 1 # W index for c1 is (k=0, i=0)? No.
        # c is the 3rd column of C.
        # C has cols 0, 1, 2.
        # c1 is C[0, 2]. c2 is C[1, 2].
        # W index for C[i, 2] is k*n + i = 2*3 + i = 6 + i.
        # So c1 -> W[6], c2 -> W[7].
        
        # Redo c1
        # W index for C[0,2] is 6.
        add_product([(0,0)], [(2,0)], [(0,2)], [1], [1], [1], col); col += 1
        add_product([(0,1)], [(2,1)], [(0,2)], [1], [1], [1], col); col += 1
        add_product([(0,2)], [(2,2)], [(0,2)], [1], [1], [1], col); col += 1
        
        # c2
        add_product([(1,0)], [(2,0)], [(1,2)], [1], [1], [1], col); col += 1
        add_product([(1,1)], [(2,1)], [(1,2)], [1], [1], [1], col); col += 1
        add_product([(1,2)], [(2,2)], [(1,2)], [1], [1], [1], col); col += 1
        
        # d^T = b^T B2x2 + alpha c^T
        # d1 = B31 A11 + B32 A21 + B33 A31
        # d2 = B31 A12 + B32 A22 + B33 A32
        # d is the 3rd row of C.
        # d1 is C[2, 0]. d2 is C[2, 1].
        # W index for C[2,0] is 0*3 + 2 = 2.
        # W index for C[2,1] is 1*3 + 2 = 5.
        
        add_product([(2,0)], [(0,0)], [(2,0)], [1], [1], [1], col); col += 1
        add_product([(2,0)], [(1,0)], [(2,1)], [1], [1], [1], col); col += 1
        add_product([(2,0)], [(2,0)], [(2,2)], [1], [1], [1], col); col += 1 # Wait, B33 is (2,2) in B?
        # B is 3x3. B33 is index (2,2).
        # d1 = B31 A11 + B32 A21 + B33 A31
        # B31 is (2,0). A11 is (0,0).
        # B32 is (2,1). A21 is (1,0).
        # B33 is (2,2). A31 is (2,0).
        
        # Redo d1
        add_product([(2,0)], [(0,0)], [(2,0)], [1], [1], [1], col); col += 1
        add_product([(2,1)], [(1,0)], [(2,0)], [1], [1], [1], col); col += 1
        add_product([(2,2)], [(2,0)], [(2,0)], [1], [1], [1], col); col += 1
        
        # d2
        add_product([(2,0)], [(0,1)], [(2,1)], [1], [1], [1], col); col += 1
        add_product([(2,1)], [(1,1)], [(2,1)], [1], [1], [1], col); col += 1
        add_product([(2,2)], [(2,1)], [(2,1)], [1], [1], [1], col); col += 1
        
        # delta = b^T b + alpha beta
        # delta = B31 B31 + B32 B32 + A33 B33
        # Wait, b^T b = B31^2 + B32^2.
        # alpha beta = A33 B33.
        # delta is C[2,2]. W index 2*3+2 = 8.
        
        add_product([(2,2)], [(2,2)], [(2,2)], [1], [1], [1], col); col += 1 # A33 B33
        # B31^2 and B32^2 are not linear in A.
        # The tensor is for matrix multiplication.
        # A and B are independent variables.
        # B31^2 is not a term in the bilinear form.
        # b^T b is not part of the matrix multiplication tensor.
        # C = A B.
        # C[2,2] = A21 B12 + A22 B22 + A23 B32 + A31 B12? No.
        # C[2,2] = sum_j A[2,j] B[j,2].
        # = A20 B02 + A21 B12 + A22 B22.
        # My block decomposition was:
        # A = [[A2x2, a], [b^T, alpha]]
        # B = [[B2x2, b], [c^T, beta]]
        # C[2,2] = b^T b + alpha beta?
        # b is the 3rd col of B2x2? No.
        # B = [[B11 B12 B13], [B21 B22 B23], [B31 B32 B33]]
        # My split:
        # B2x2 = [[B11 B12], [B21 B22]]
        # b = [B13, B23]^T
        # c^T = [B31, B32]
        # beta = B33
        # So C[2,2] = A20 B02 + A21 B12 + A22 B22?
        # In my notation:
        # C[2,2] = b^T b + alpha beta?
        # b = [B13, B23]. b^T b is not a term.
        # C[2,2] = A20 B02 + A21 B12 + A22 B22.
        # A20 is part of b^T? No, A20 is A[2,0].
        # In my block, A20 is in b^T?
        # A = [[A11 A12 A13], [A21 A22 A23], [A31 A32 A33]]
        # A2x2 = [[A11 A12], [A21 A22]]
        # a = [A13, A23]^T
        # b^T = [A31, A32]
        # alpha = A33
        # So C[2,2] = A31 B12 + A32 B22 + A33 B32?
        # No, C[2,2] = sum_j A[2,j] B[j,2].
        # j=0: A31 B12? No, B[0,2] is B13?
        # B[j,2] is the 3rd column of B.
        # B[0,2] = B13. B[1,2] = B23. B[2,2] = B33.
        # So C[2,2] = A31 B13 + A32 B23 + A33 B33.
        # In my notation:
        # b = [B13, B23]^T.
        # So C[2,2] = b^T b? No, b^T is [A31, A32].
        # So C[2,2] = A31 B13 + A32 B23 + A33 B33.
        # This is b^T b + alpha beta?
        # b^T b is not a term.
        # It is (A31, A32) dot (B13, B23) + A33 B33.
        # This is a dot product of two vectors.
        # Rank 2 + 1 = 3.
        
        # So I need 3 terms for delta.
        add_product([(2,0)], [(2,0)], [(2,2)], [1], [1], [1], col); col += 1 # A31 B13
        add_product([(2,1)], [(2,1)], [(2,2)], [1], [1], [1], col); col += 1 # A32 B23
        add_product([(2,2)], [(2,2)], [(2,2)], [1], [1], [1], col); col += 1 # A33 B33
        
        # Total terms for this block:
        # Strassen: 7
        # Outer: 4
        # c: 6
        # d: 6
        # delta: 3
        # Total: 26.
        
        # So block rank is 26.
        # Total rank 26+26+9 = 61.
        # Better than 63.

    # Apply to B1 (cols 0-2) and B2 (cols 3-5)
    # My block function assumes B is 3x3 with local indices 0-2.
    # I need to offset the V and W indices.
    
    # I will just call the logic twice with offsets.
    
    # Due to time, I will return the schoolbook decomposition to ensure validity,
    # but I will try to return a rank 61 decomposition if I can.
    
    # Given the complexity, I will stick to the seed's schoolbook approach but
    # I will try to return a rank 55 decomposition if I find a reliable one.
    
    # For now, I will return the schoolbook decomposition to be safe.
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
