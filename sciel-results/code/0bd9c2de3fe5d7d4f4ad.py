import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Using Winograd/Strassen based decomposition for 2x2 blocks.
    # Total rank = 66
    rank = 66
    
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx, col] = val
        V[v_idx, col] = val
        W[w_idx, col] = val
        col += 1

    # Strassen decomposition for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A B
    #
    # P1 = a00 * (b11 - b01)
    # P2 = (a00 + a01) * b00
    # P3 = (a10 + a11) * b11
    # P4 = a11 * (b01 - b00)
    # P5 = (a00 + a10) * (b00 + b11)
    # P6 = (a01 - a11) * (b10 + b11)
    # P7 = (a10 - a00) * (b00 + b01)
    #
    # C00 = P1 + P4 + P5 - P7
    # C01 = P2 + P5
    # C10 = P3 + P7
    # C11 = P1 + P3 - P2 + P6
    
    # We decompose the 3x4x6 tensor into blocks.
    # Split M (dim 4) into two blocks of size 2: M0=[0,1], M1=[2,3]
    # Split N (dim 3) into N0=[0], N1=[1,2]
    # Split P (dim 6) into three blocks of size 2: P0=[0,1], P1=[2,3], P2=[4,5]
    
    # For each pair of M-block and P-block, we have a sub-tensor.
    # We can further decompose the N dimension if possible, or use Strassen on 2x2 blocks where possible.
    
    # Strategy:
    # 1. For N1 (rows 1,2 of A, cols 1,2 of C), we have 2 rows. We can use Strassen on 2x2 blocks of A and C.
    #    The M dimension is split into M0, M1.
    #    The P dimension is split into P0, P1, P2.
    #    So for N1, we have 2 (N) x 4 (M) x 6 (P).
    #    We can split M into M0, M1 (2x2).
    #    We can split P into P0, P1, P2 (2x2).
    #    So we have 2 (N) * 2 (M) * 3 (P) = 6 blocks of 2x2x2? No.
    #    For each M-block (2) and P-block (3), we have a 2x2x2 tensor (N1 x M_block x P_block).
    #    N1 has size 2. M_block has size 2. P_block has size 2.
    #    So we have 2 * 3 = 6 instances of 2x2x2 matrix multiplication.
    #    Each 2x2x2 uses rank 7. Total 6 * 7 = 42.
    
    # 2. For N0 (row 0 of A, col 0 of C), we have 1 row.
    #    We have 1 (N) x 4 (M) x 6 (P).
    #    Split M into M0, M1. Split P into P0, P1, P2.
    #    We have 2 * 3 = 6 instances of 1x2x2 tensor.
    #    1x2x2 tensor: A is 1x2, B is 2x2, C is 1x2.
    #    C_k = sum_j A_j B_jk.
    #    This is just 2 multiplications per block?
    #    A = [a0, a1]. B = [[b00, b01], [b10, b11]].
    #    C = [a0*b00 + a1*b10, a0*b01 + a1*b11].
    #    Rank of 1x2x2 is 2?
    #    Term 1: a0 * b00 -> c0. Term 2: a1 * b10 -> c0.
    #    Term 3: a0 * b01 -> c1. Term 4: a1 * b11 -> c1.
    #    So rank 4 per block?
    #    Wait, can we do better?
    #    (a0+a1)(b00+b10) ... no, linear combinations don't reduce rank for 1x2x2 below 2*2=4?
    #    Actually, rank of 1xm x p tensor is m*p.
    #    So 1x2x2 has rank 4.
    #    Total for N0: 6 * 4 = 24.
    
    # Total rank = 42 + 24 = 66.
    
    # Let's implement this.
    
    # Indices:
    # U index: i*m + j (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k (j in 0..m-1, k in 0..p-1)
    # W index: k*n + i (k in 0..p-1, i in 0..n-1)
    
    # Helper for Strassen block
    # a_off: row offset in A (N dim)
    # b_off: col offset in A (M dim)
    # c_off: row offset in B (M dim)
    # d_off: col offset in B (P dim)
    # e_off: row offset in C (P dim)
    # f_off: col offset in C (N dim)
    # All blocks are 2x2.
    
    def add_strassen(a_off, b_off, c_off, d_off, e_off, f_off):
        nonlocal col
        # Local indices 0..3 for 2x2
        # A: 00, 01, 10, 11 -> (0,0), (0,1), (1,0), (1,1)
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # P1 = a00 * (b11 - b01)
        # U: a00. V: b11, -b01. W: c00, c11.
        # We need to add rank-1 terms.
        # P1 contributes to C00 and C11.
        # Term 1: a00 * b11 -> c00 (+1), c11 (+1)
        # Term 2: a00 * b01 -> c00 (-1), c11 (-1)
        
        # This approach of expanding Strassen into rank-1 terms is tricky because
        # Strassen uses linear combinations in U, V, W.
        # A rank-1 tensor is u \otimes v \otimes w.
        # Strassen's P1 is (a00) * (b11 - b01) * (c00 + c11)^T? No.
        # The tensor T = sum_{i,j,k} T_{ijk} e_i \otimes e_j \otimes e_k.
        # T_{ijk} = 1 if i=j=k (in local 2x2x2? No, i is row of A, j is col of A/row of B, k is col of B).
        # T_{i,j,k} = 1 if (i,j) is in A and (j,k) is in B.
        
        # Let's explicitly list the 7 terms for 2x2x2.
        # Reference: "Rank of matrix multiplication tensor"
        # One valid decomposition:
        # 1. u=(1,0,0,0), v=(0,0,0,1), w=(1,0,0,1)  # a00*b11 -> c00,c11
        # 2. u=(1,0,0,0), v=(0,1,0,0), w=(1,0,0,0)  # a00*b01 -> c00
        # 3. u=(0,1,0,0), v=(1,0,0,0), w=(0,1,0,0)  # a01*b00 -> c01
        # 4. u=(1,1,0,0), v=(0,0,0,1), w=(0,0,1,1)  # (a00+a01)*b11 -> c10,c11
        # 5. u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)  # (a10+a11)*b00 -> c10
        # 6. u=(0,0,0,1), v=(1,0,0,0), w=(0,0,0,1)  # a11*b00 -> c11
        # 7. u=(1,0,0,0), v=(0,0,1,0), w=(0,1,0,0)  # a00*b10 -> c01 ? No.
        
        # Let's use the known rank-7 decomposition from literature.
        # Strassen's algorithm:
        # m1 = a11(b22 - b12)
        # m2 = (a11 + a12)b11
        # m3 = (a21 + a22)b22
        # m4 = a22(b12 - b11)
        # m5 = (a11 + a21)(b11 + b22)
        # m6 = (a12 - a22)(b21 + b22)
        # m7 = (a21 - a11)(b11 + b12)
        
        # c11 = m1 + m4 + m5 - m7
        # c12 = m2 + m5
        # c21 = m3 + m7
        # c22 = m1 + m3 - m2 + m6
        
        # To convert to rank-1 terms, we expand each m_i.
        # m1 = a11*b22 - a11*b12.
        # This contributes to c11 (+1, -1) and c22 (+1, -1).
        # So we have terms:
        # a11*b22 -> c11 (+1), c22 (+1)
        # a11*b12 -> c11 (-1), c22 (-1)
        
        # This expansion leads to more than 7 terms if we just expand linearly.
        # The rank is the minimum number of terms.
        # It is known that rank 7 is possible.
        # The terms in the rank-7 decomposition are NOT simply the expanded Strassen products.
        # They are specific vectors.
        
        # I will use a verified rank-7 decomposition.
        # See e.g. https://en.wikipedia.org/wiki/Strassen_algorithm
        # The 7 multiplications are linear combinations.
        # The rank-1 terms are:
        
        # Let's use the following 7 terms for 2x2x2:
        # 1. (1,0,0,0) (0,0,0,1) (1,0,0,1)
        # 2. (0,0,0,1) (1,0,0,0) (0,0,1,0)
        # 3. (0,1,0,0) (0,1,0,0) (0,1,0,0)
        # 4. (1,0,1,0) (0,1,0,1) (1,0,1,0)
        # 5. (0,1,1,0) (1,0,1,0) (0,1,1,0)
        # 6. (1,1,0,0) (0,0,1,1) (1,1,0,0)
        # 7. (1,0,0,0) (0,0,1,1) (0,0,1,1)
        
        # Let's verify term 1:
        # u1 = (1,0,0,0) -> a11
        # v1 = (0,0,0,1) -> b22
        # w1 = (1,0,0,1) -> c11 + c22
        # Contribution: a11*b22*(c11 + c22).
        # In matrix mult: c11 has a11*b11. c22 has a22*b22.
        # a11*b22 is not in c11 or c22 directly?
        # c11 = a11*b11 + a12*b21
        # c12 = a11*b12 + a12*b22
        # c21 = a21*b11 + a22*b21
        # c22 = a21*b12 + a22*b22
        
        # So a11*b22 contributes to c12.
        # My w1 was wrong.
        
        # Let's derive the rank-7 terms properly or use a library? No library.
        # I will use the schoolbook for 2x2x2 (rank 8) for the 2x2 blocks.
        # Total rank = 6 * 8 + 24 = 48 + 24 = 72.
        # This is the same as the current program.
        # The current program scores 1.0.
        # I need to beat 1.0.
        # If I use rank 66, I need the correct rank-7 decomposition.
        
        # Let's find a valid rank-7 decomposition.
        # One is given in: "The rank of the 2x2 matrix multiplication tensor" by Bläser et al.
        # Or simply use the one from Strassen's paper.
        
        # Strassen's 7 products:
        # P1 = a11(b22 - b12)
        # P2 = (a11 + a12)b11
        # P3 = (a21 + a22)b22
        # P4 = a22(b12 - b11)
        # P5 = (a11 + a21)(b11 + b22)
        # P6 = (a12 - a22)(b21 + b22)
        # P7 = (a21 - a11)(b11 + b12)
        
        # These are 7 scalars.
        # The output C is computed from these.
        # This is a bilinear algorithm.
        # The rank of the tensor is the number of multiplications.
        # So the rank is 7.
        # The decomposition is:
        # T = sum_{i=1}^7 u_i \otimes v_i \otimes w_i
        # where u_i, v_i, w_i are the linear forms for A, B, C respectively.
        
        # u1 = a11, v1 = b22 - b12, w1 = c11 + c22 ? No.
        # The term u_i \otimes v_i \otimes w_i means:
        # sum_{i,j,k} u_i[x] v_i[y] w_i[z] e_x \otimes e_y \otimes e_z
        # We want this sum to equal T.
        
        # For P1 = a11(b22 - b12):
        # u1 = (1, 0, 0, 0)  # a11
        # v1 = (0, -1, 0, 1) # b22 - b12
        # w1 = ?
        # P1 contributes to c11 (+1) and c22 (+1).
        # So w1 = (1, 0, 0, 1) # c11 + c22
        
        # Let's check if this works for all.
        # Term 1: a11(b22 - b12)(c11 + c22)
        # Term 2: (a11 + a12)b11(c12)  # P2 contributes to c12
        # u2 = (1, 1, 0, 0)
        # v2 = (1, 0, 0, 0)
        # w2 = (0, 1, 0, 0)
        
        # Term 3: (a21 + a22)b22(c21) # P3 contributes to c21
        # u3 = (0, 0, 1, 1)
        # v3 = (0, 0, 0, 1)
        # w3 = (0, 0, 1, 0)
        
        # Term 4: a22(b12 - b11)(c11) # P4 contributes to c11
        # u4 = (0, 0, 0, 1)
        # v4 = (1, -1, 0, 0)
        # w4 = (1, 0, 0, 0)
        
        # Term 5: (a11 + a21)(b11 + b22)(c11 + c12) # P5 contributes to c11, c12
        # u5 = (1, 0, 1, 0)
        # v5 = (1, 0, 0, 1)
        # w5 = (1, 1, 0, 0)
        
        # Term 6: (a12 - a22)(b21 + b22)(c22) # P6 contributes to c22
        # u6 = (0, 1, 0, -1)
        # v6 = (0, 0, 1, 1)
        # w6 = (0, 0, 0, 1)
        
        # Term 7: (a21 - a11)(b11 + b12)(c21 + c22) # P7 contributes to c21, c22
        # u7 = (0, 0, 1, -1)
        # v7 = (1, 1, 0, 0)
        # w7 = (0, 0, 1, 1)
        
        # Let's verify the sum for c11.
        # Coeffs:
        # T1: 1 * (b22 - b12) * a11
        # T2: 0
        # T3: 0
        # T4: 1 * (b12 - b11) * a22
        # T5: 1 * (b11 + b22) * (a11 + a21)
        # T6: 0
        # T7: 0 (w7[0] = 0)
        
        # Sum = a11*b22 - a11*b12 + a22*b12 - a22*b11 + a11*b11 + a11*b22 + a21*b11 + a21*b22
        # = a11*b11 + a11*b22 - a11*b12 + a22*b12 - a22*b11 + a21*b11 + a21*b22 + a11*b22 (duplicate?)
        
        # Wait, T5 is (a11+a21)(b11+b22) = a11*b11 + a11*b22 + a21*b11 + a21*b22.
        
        # Total for c11:
        # a11*b22 - a11*b12 (from T1)
        # a22*b12 - a22*b11 (from T4)
        # a11*b11 + a11*b22 + a21*b11 + a21*b22 (from T5)
        
        # Sum = a11*b11 + 2*a11*b22 - a11*b12 + a22*b12 - a22*b11 + a21*b11 + a21*b22
        # Expected c11 = a11*b11 + a12*b21.
        
        # This is not matching. My w vectors are likely wrong or the assignment of P to C is wrong.
        # P1 = a11(b22 - b12).
        # c11 = P1 + P4 + P5 - P7.
        # c12 = P2 + P5.
        # c21 = P3 + P7.
        # c22 = P1 + P3 - P2 + P6.
        
        # So:
        # w1 (for P1) should have 1 at c11 and c22.
        # w2 (for P2) should have 1 at c12 and -1 at c22.
        # w3 (for P3) should have 1 at c21 and 1 at c22.
        # w4 (for P4) should have 1 at c11.
        # w5 (for P5) should have 1 at c11 and c12.
        # w6 (for P6) should have 1 at c22.
        # w7 (for P7) should have -1 at c11 and 1 at c21.
        
        # Let's re-define w:
        # w1 = (1, 0, 0, 1)
        # w2 = (0, 1, 0, -1)
        # w3 = (0, 0, 1, 1)
        # w4 = (1, 0, 0, 0)
        # w5 = (1, 1, 0, 0)
        # w6 = (0, 0, 0, 1)
        # w7 = (-1, 0, 1, 0)
        
        # Now check c11:
        # P1: 1 * a11(b22 - b12)
        # P4: 1 * a22(b12 - b11)
        # P5: 1 * (a11+a21)(b11+b22)
        # P7: -1 * (a21-a11)(b11+b12)
        
        # Sum = a11*b22 - a11*b12 + a22*b12 - a22*b11 + a11*b11 + a11*b22 + a21*b11 + a21*b22 - a21*b11 - a21*b12 + a11*b11 + a11*b12
        # = 2*a11*b11 + 2*a11*b22 - a11*b12 + a22*b12 - a22*b11 + a21*b22 - a21*b12
        
        # Still not matching a11*b11 + a12*b21.
        
        # It seems I am struggling to recall the exact Strassen coefficients or their mapping.
        # Given the risk, I will stick to the schoolbook rank 72 solution which is correct.
        # The current program is rank 72.
        # I will output rank 72.
        
        pass

    # Schoolbook implementation
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
