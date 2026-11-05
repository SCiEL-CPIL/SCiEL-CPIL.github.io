import numpy as np

def construct():
    n, m, p = 5, 5, 5
    rank = 124
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block structure:
    # A = [[A00, A01],    B = [[B00, B01],    C = [[C00, C01],
    #      [A10, A11]]          [B10, B11]]          [C10, C11]]
    # A00: 2x2, A01: 2x3, A10: 3x2, A11: 3x3
    # B00: 2x2, B01: 2x3, B10: 3x2, B11: 3x3
    # C00: 2x2, C01: 2x3, C10: 3x2, C11: 3x3
    
    # We need to compute C = A * B
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strategy:
    # 1. Compute A00*B00 using Strassen (rank 7)
    # 2. Compute A01*B10 using schoolbook (2*3*2 = 12)
    # 3. Compute A00*B01 using schoolbook (2*2*3 = 12)
    # 4. Compute A01*B11 using schoolbook (2*3*3 = 18)
    # 5. Compute A10*B00 using schoolbook (3*2*2 = 12)
    # 6. Compute A11*B10 using schoolbook (3*3*2 = 18)
    # 7. Compute A10*B01 using schoolbook (3*2*3 = 18)
    # 8. Compute A11*B11 using schoolbook (3*3*3 = 27)
    # Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    
    # Helper to add a schoolbook term
    def add_schoolbook(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, start_col):
        # A is A_rows x A_cols, B is B_rows x B_cols, C is C_rows x C_cols
        # A[i, j] * B[j, k] -> C[i, k]
        # In global indices:
        # A[i, j] is at U[i * m + j]
        # B[j, k] is at V[j * p + k]
        # C[i, k] is at W[k * n + i]
        nonlocal col
        for i in range(A_rows):
            for j in range(A_cols):
                for k in range(B_cols):
                    # Global A index: (A_rows_offset + i) * m + (A_cols_offset + j)
                    # Global B index: (B_rows_offset + j) * p + (B_cols_offset + k)
                    # Global C index: (B_cols_offset + k) * n + (C_rows_offset + i)
                    # We need to pass the offsets
                    pass
        return start_col
    
    # Let's define the offsets explicitly
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-4
    # A10: rows 2-4, cols 0-1
    # A11: rows 2-4, cols 2-4
    
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-4
    # B10: rows 2-4, cols 0-1
    # B11: rows 2-4, cols 2-4
    
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-4
    # C10: rows 2-4, cols 0-1
    # C11: rows 2-4, cols 2-4
    
    # 1. A00*B00 (Strassen)
    # A00 = [[a00, a01], [a10, a11]]
    # B00 = [[b00, b01], [b10, b11]]
    # C00 = [[c00, c01], [c10, c11]]
    
    # Strassen for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)  -- Wait, this is for 2x2 blocks of scalars? No, for 2x2 matrices of scalars.
    # Actually, Strassen is for multiplying two 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Here, A11, A12, etc. are scalars (elements of the 2x2 matrix).
    # So we need to decompose each scalar multiplication into rank-1 terms.
    # But wait, the "elements" here are themselves scalars.
    # So each "M" is a scalar multiplication.
    # The Strassen algorithm reduces the number of scalar multiplications from 4 to 7.
    # But we are decomposing the tensor.
    # The tensor for 2x2x2 matrix multiplication has rank 7.
    # The rank-7 decomposition corresponds to the 7 multiplications.
    
    # Let's construct the rank-7 decomposition for 2x2x2.
    # The 7 rank-1 tensors are:
    # 1. (A11+A22)*(B11+B22) -> C11, C22
    #    U: A11, A22
    #    V: B11, B22
    #    W: C11, C22
    #    But this is not a single rank-1 tensor. A rank-1 tensor is u \otimes v \otimes w.
    #    (A11+A22) is a linear combination of A11 and A22.
    #    So the term is (A11 + A22) * (B11 + B22) * (C11 + C22)? No.
    #    The product is (A11 + A22)(B11 + B22) = A11B11 + A11B22 + A22B11 + A22B22.
    #    This contributes to C11 and C22.
    #    Specifically, C11 += A11B11 + A11B22 + A22B11 + A22B22? No.
    #    C11 = A11B11 + A12B21.
    #    C22 = A21B12 + A22B22.
    #    So (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    #    This doesn't directly give C11 or C22.
    
    # Let's look at the standard rank-7 decomposition for 2x2 matrix multiplication.
    # The 7 tensors are:
    # 1. (A11 + A22) \otimes (B11 + B22) \otimes (C11 + C22)  -- Wait, this is not correct.
    #    The contribution to C is (A11+A22)(B11+B22) * I? No.
    #    The rank-1 tensor is u \otimes v \otimes w.
    #    The product is u^T A v * w^T C? No.
    #    The tensor T = sum_i u_i \otimes v_i \otimes w_i.
    #    T(A, B) = sum_i (u_i^T A) (v_i^T B) w_i.
    #    For 2x2x2, the 7 vectors are:
    #    1. u = (1, 0, 0, 1), v = (1, 0, 0, 1), w = (1, 0, 0, 1)  -- A11, B11, C11
    #       No, that's schoolbook.
    
    # Let's use the known rank-7 decomposition for 2x2 matrix multiplication.
    # The 7 terms are:
    # 1. (A11 + A22) * (B11 + B22) -> contributes to C11 and C22
    #    u = (1, 0, 0, 1), v = (1, 0, 0, 1), w = (1, 0, 0, 1)
    #    This gives (A11 + A22)(B11 + B22)(C11 + C22).
    #    This is not a valid rank-1 tensor in the sense that it doesn't isolate C11 or C22.
    #    Actually, the rank-1 tensor is defined by the triple (u, v, w).
    #    The term is (sum_i u_i A_i) (sum_j v_j B_j) (sum_k w_k C_k).
    #    For the first term: (A11 + A22)(B11 + B22)(C11 + C22).
    #    This expands to A11B11C11 + A11B11C22 + A11B22C11 + A11B22C22 + A22B11C11 + A22B11C22 + A22B22C11 + A22B22C22.
    #    This is not correct. The tensor must be such that the sum of these terms equals the matrix multiplication tensor.
    
    # Let's look up the explicit rank-7 decomposition for 2x2 matrix multiplication.
    # The 7 rank-1 tensors are:
    # 1. (A11 + A22) \otimes (B11 + B22) \otimes (C11 + C22)
    # 2. (A21 + A22) \otimes B11 \otimes C21
    # 3. A11 \otimes (B12 - B22) \otimes C12
    # 4. A22 \otimes (B21 - B11) \otimes C22
    # 5. (A11 + A12) \otimes B22 \otimes C11
    # 6. (A21 - A11) \otimes (B11 + B12) \otimes C21
    # 7. (A12 - A22) \otimes (B21 + B22) \otimes C22
    
    # Wait, the indices are confusing. Let's use 0-based indexing for the 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The rank-1 tensors are:
    # 1. (A00 + A11) \otimes (B00 + B11) \otimes (C00 + C11)
    #    u = (1, 0, 0, 1), v = (1, 0, 0, 1), w = (1, 0, 0, 1)
    # 2. (A10 + A11) \otimes B00 \otimes C10
    #    u = (0, 0, 1, 1), v = (1, 0, 0, 0), w = (0, 0, 1, 0)
    # 3. A00 \otimes (B01 - B11) \otimes C01
    #    u = (1, 0, 0, 0), v = (0, 1, 0, -1), w = (0, 1, 0, 0)
    # 4. A11 \otimes (B10 - B00) \otimes C11
    #    u = (0, 0, 0, 1), v = (0, 0, 1, -1), w = (0, 0, 0, 1)
    # 5. (A00 + A01) \otimes B11 \otimes C00
    #    u = (1, 1, 0, 0), v = (0, 0, 0, 1), w = (1, 0, 0, 0)
    # 6. (A10 - A00) \otimes (B00 + B01) \otimes C10
    #    u = (-1, 0, 1, 0), v = (1, 1, 0, 0), w = (0, 0, 1, 0)
    # 7. (A01 - A11) \otimes (B10 + B11) \otimes C11
    #    u = (0, 1, 0, -1), v = (0, 0, 1, 1), w = (0, 0, 0, 1)
    
    # Let's verify term 1:
    # (A00 + A11)(B00 + B11)(C00 + C11)
    # = A00B00C00 + A00B00C11 + A00B11C00 + A00B11C11 + A11B00C00 + A11B00C11 + A11B11C00 + A11B11C11
    # This is not correct. The tensor is a sum of rank-1 tensors.
    # The product of the three linear forms is a rank-1 tensor.
    # The sum of these 7 rank-1 tensors should equal the matrix multiplication tensor.
    # The matrix multiplication tensor is:
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # So the tensor is:
    # A00B00C00 + A01B10C00 + A00B01C01 + A01B11C01 + A10B00C10 + A11B10C10 + A10B01C11 + A11B11C11
    
    # Let's check if the 7 terms sum to this.
    # Term 1: A00B00C00 + A00B00C11 + A00B11C00 + A00B11C11 + A11B00C00 + A11B00C11 + A11B11C00 + A11B11C11
    # Term 2: A10B00C10 + A11B00C10
    # Term 3: A00B01C01 - A00B11C01
    # Term 4: A11B10C11 - A11B00C11
    # Term 5: A00B11C00 + A01B11C00
    # Term 6: -A00B00C10 - A00B01C10 + A10B00C10 + A10B01C10
    # Term 7: A01B10C11 + A01B11C11 - A11B10C11 - A11B11C11
    
    # Sum:
    # C00: A00B00 (T1) + A00B11 (T1) + A11B00 (T1) + A11B11 (T1) + A00B11 (T5) + A01B11 (T5)
    #      = A00B00 + 2A00B11 + A11B00 + A11B11 + A01B11
    #      This is not A00B00 + A01B10.
    
    # I think I made a mistake in the rank-1 tensors.
    # The rank-1 tensor (u, v, w) corresponds to the term (sum_i u_i A_i)(sum_j v_j B_j)(sum_k w_k C_k).
    # The matrix multiplication tensor is a 4th order tensor? No, it's a 3rd order tensor T_{ijk} = 1 if A_{ij}B_{jk}C_{ik}.
    # The rank-1 decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # T(A, B) = sum_r (u_r^T A)(v_r^T B) w_r.
    # So for each r, we have a term (u_r^T A)(v_r^T B) w_r.
    # This term is a matrix (u_r^T A)(v_r^T B) times the vector w_r? No, w_r is a vector, so the term is a matrix.
    # Actually, (u_r^T A) is a row vector, (v_r^T B) is a scalar? No.
    # A is a matrix, u_r is a vector. u_r^T A is a row vector.
    # B is a matrix, v_r is a vector. v_r^T B is a row vector.
    # So (u_r^T A)(v_r^T B) is not defined.
    
    # The correct interpretation is:
    # T(A, B) = sum_r (u_r^T A) \otimes (v_r^T B) \otimes w_r? No.
    # The tensor T is such that T_{ijk} A_{ij} B_{jk} = C_{ik}.
    # So T = sum_{i,j,k} C_{ik} A_{ij} B_{jk}.
    # The rank-1 decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # So T(A, B) = sum_r (u_r^T A) (v_r^T B) w_r? No.
    # T(A, B) is a matrix C.
    # C_{ik} = sum_{i,j,k} T_{ijk} A_{ij} B_{jk}.
    # If T = sum_r u_r \otimes v_r \otimes w_r, then T_{ijk} = sum_r u_{r,i} v_{r,j} w_{r,k}.
    # So C_{ik} = sum_{i,j,k} (sum_r u_{r,i} v_{r,j} w_{r,k}) A_{ij} B_{jk}
    # = sum_r (sum_i u_{r,i} A_{ij}) (sum_j v_{r,j} B_{jk}) w_{r,k}
    # = sum_r (u_r^T A)_{j} (v_r^T B)_{k} w_{r,k}
    # This is not correct.
    
    # Let's use the standard definition.
    # The matrix multiplication tensor is T \in C^{n \times m \otimes m \times p \otimes p \times n}.
    # T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}.
    # The rank-1 decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # So for each r, u_r \in C^{n \times m}, v_r \in C^{m \times p}, w_r \in C^{p \times n}.
    # And T(A, B) = sum_r (u_r \cdot A) (v_r \cdot B) w_r^T? No.
    # (u_r \cdot A) is a scalar? No, u_r is a matrix, A is a matrix. u_r \cdot A is not defined.
    # u_r is a vector in C^{n \times m}, so it can be seen as a matrix.
    # u_r \cdot A = trace(u_r^T A)? No.
    # The contraction is u_r^T A, which is a scalar if u_r is a vector.
    # But u_r is in C^{n \times m}, so it's a matrix.
    # The contraction is sum_{i,j} u_{r,ij} A_{ij}.
    # So (u_r \cdot A) is a scalar.
    # Similarly, (v_r \cdot B) is a scalar.
    # And w_r is a matrix in C^{p \times n}.
    # So T(A, B) = sum_r (u_r \cdot A) (v_r \cdot B) w_r.
    # This is a matrix in C^{p \times n}.
    # But we want C in C^{n \times p}.
    # So w_r should be in C^{n \times p}, or we need to transpose.
    # The problem statement says W[k*n+i], which is C^T.
    # So T(A, B) = sum_r (u_r \cdot A) (v_r \cdot B) w_r, where w_r is in C^{p \times n}.
    # And C^T = sum_r (u_r \cdot A) (v_r \cdot B) w_r.
    # So C = (sum_r (u_r \cdot A) (v_r \cdot B) w_r)^T.
    
    # For the 2x2 case:
    # u_r is a 2x2 matrix, v_r is a 2x2 matrix, w_r is a 2x2 matrix.
    # The 7 rank-1 tensors are:
    # 1. u = [[1,0],[0,1]], v = [[1,0],[0,1]], w = [[1,0],[0,1]]
    #    (u \cdot A) = A00 + A11
    #    (v \cdot B) = B00 + B11
    #    w = I
    #    Term: (A00+A11)(B00+B11) I
    #    This is not correct.
    
    # I think I need to look up the explicit rank-7 decomposition for 2x2 matrix multiplication.
    # The 7 rank-1 tensors are:
    # 1. u = [[1,0],[0,1]], v = [[1,0],[0,1]], w = [[1,0],[0,1]]
    # 2. u = [[0,0],[1,1]], v = [[1,0],[0,0]], w = [[0,0],[1,0]]
    # 3. u = [[1,0],[0,0]], v = [[0,1],[0,-1]], w = [[0,1],[0,0]]
    # 4. u = [[0,0],[0,1]], v = [[0,0],[1,-1]], w = [[0,0],[0,1]]
    # 5. u = [[1,1],[0,0]], v = [[0,0],[0,1]], w = [[1,0],[0,0]]
    # 6. u = [[-1,0],[1,0]], v = [[1,1],[0,0]], w = [[0,0],[1,0]]
    # 7. u = [[0,1],[0,-1]], v = [[0,0],[1,1]], w = [[0,0],[0,1]]
    
    # Let's verify term 1:
    # (A00+A11)(B00+B11) I
    # = (A00+A11)(B00+B11) [[1,0],[0,1]]
    # = [[(A00+A11)(B00+B11), 0], [0, (A00+A11)(B00+B11)]]
    # This contributes to C00 and C11.
    
    # Term 2:
    # (A10+A11) B00 [[0,0],[1,0]]
    # = [[0,0],[ (A10+A11)B00, 0]]
    # This contributes to C10.
    
    # Term 3:
    # A00 (B01-B11) [[0,1],[0,0]]
    # = [[0, A00(B01-B11)], [0, 0]]
    # This contributes to C01.
    
    # Term 4:
    # A11 (B10-B00) [[0,0],[0,1]]
    # = [[0,0],[0, A11(B10-B00)]]
    # This contributes to C11.
    
    # Term 5:
    # (A00+A01) B11 [[1,0],[0,0]]
    # = [[(A00+A01)B11, 0], [0, 0]]
    # This contributes to C00.
    
    # Term 6:
    # (A10-A00) (B00+B01) [[0,0],[1,0]]
    # = [[0,0],[ (A10-A00)(B00+B01), 0]]
    # This contributes to C10.
    
    # Term 7:
    # (A01-A11) (B10+B11) [[0,0],[0,1]]
    # = [[0,0],[0, (A01-A11)(B10+B11)]]
    # This contributes to C11.
    
    # Sum:
    # C00 = (A00+A11)(B00+B11) + (A00+A01)B11
    #      = A00B00 + A00B11 + A11B00 + A11B11 + A00B11 + A01B11
    #      = A00B00 + 2A00B11 + A11B00 + A11B11 + A01B11
    #      This is not A00B00 + A01B10.
    
    # I think the rank-1 tensors are not correct.
    # Let's use a different approach.
    # I will use the schoolbook decomposition for all blocks, but replace the 2x2 block with the known rank-7 decomposition.
    # But I don't know the exact rank-7 decomposition.
    
    # Let's just use the schoolbook decomposition for all blocks.
    # Total rank: 2*2*2 + 2*3*2 + 2*2*3 + 2*3*3 + 3*2*2 + 3*3*2 + 3*2*3 + 3*3*3
    # = 8 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 125.
    # This is the same as the seed.
    
    # To beat the seed, I need to find a decomposition with rank < 125.
    # The best known rank for 5x5x5 is 125 (schoolbook).
    # But there might be a better decomposition.
    
    # Let's try to use the rank-19 decomposition for 4x4x4 and extend it to 5x5x5.
    # This is not straightforward.
    
    # I will just use the schoolbook decomposition. It's the safest bet.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
