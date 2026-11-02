"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Use a more efficient decomposition by leveraging the small n=2 dimension.
    # We can view the matrix multiplication as computing C = A*B where A is 2x4 and B is 4x5.
    # The tensor has n*m*p = 2*4*5 = 40 entries.
    # 
    # Approach: Use a bilinear decomposition that minimizes the number of scalar multiplications.
    # For 2x4 * 4x5, we can use a hybrid approach:
    # - Split B into B1 (4x2) and B2 (4x3)
    # - C1 = A*B1 (2x2), C2 = A*B2 (2x3)
    # 
    # For C1 (2x2 = 2x4 * 4x2):
    #   Split A into A1 (2x2) and A2 (2x2)
    #   C1 = A1*B11 + A2*B21 where B11 is 2x2, B21 is 2x2
    #   Use Strassen for each: 7+7 = 14 mults
    # 
    # For C2 (2x3 = 2x4 * 4x3):
    #   Split B2 into B21 (4x1) and B22 (4x2)
    #   C21 = A*B21 (2x1), C22 = A*B22 (2x2)
    #   
    #   For C21 (2x1 = 2x4 * 4x1):
    #     This is just two dot products of length 4.
    #     We can compute:
    #       c0 = a00*b0 + a01*b1 + a02*b2 + a03*b3
    #       c1 = a10*b0 + a11*b1 + a12*b2 + a13*b3
    #     Schoolbook: 8 mults.
    #     But we can do better by precomputing sums:
    #       s1 = a00+a10, s2 = a01+a11, s3 = a02+a12, s4 = a03+a13
    #       t1 = b0, t2 = b1, t3 = b2, t4 = b3
    #       m1 = s1*t1, m2 = s2*t2, m3 = s3*t3, m4 = s4*t4
    #       d1 = a00-a10, d2 = a01-a11, d3 = a02-a12, d4 = a03-a13
    #       m5 = d1*t1, m6 = d2*t2, m7 = d3*t3, m8 = d4*t4
    #       c0 = (m1+m2+m3+m4+m5+m6+m7+m8)/2 + ... this doesn't work directly.
    #     
    #     Actually, for 2x1, the minimum is 4 mults if we allow precomputation:
    #       Let u1 = a00, u2 = a01, u3 = a02, u4 = a03
    #       Let v1 = b0, v2 = b1, v3 = b2, v4 = b3
    #       c0 = u1*v1 + u2*v2 + u3*v3 + u4*v4
    #       c1 = a10*v1 + a11*v2 + a12*v3 + a13*v4
    #     
    #     We can use:
    #       m1 = (a00+a10)*b0
    #       m2 = (a01+a11)*b1
    #       m3 = (a02+a12)*b2
    #       m4 = (a03+a13)*b3
    #       m5 = (a00-a10)*b0
    #       m6 = (a01-a11)*b1
    #       m7 = (a02-a12)*b2
    #       m8 = (a03-a13)*b3
    #       c0 = (m1+m2+m3+m4+m5+m6+m7+m8)/2 ... no, this gives c0+c1
    #     
    #     Actually: c0 + c1 = (a00+a10)*b0 + (a01+a11)*b1 + (a02+a12)*b2 + (a03+a13)*b3
    #                 c0 - c1 = (a00-a10)*b0 + (a01-a11)*b1 + (a02-a12)*b2 + (a03-a13)*b3
    #     So we need 8 mults for 2x1 if we want both c0 and c1.
    #     But wait, we can share the b's:
    #       m1 = (a00+a10)*b0
    #       m2 = (a01+a11)*b1
    #       m3 = (a02+a12)*b2
    #       m4 = (a03+a13)*b3
    #       m5 = (a00-a10)*b0
    #       m6 = (a01-a11)*b1
    #       m7 = (a02-a12)*b2
    #       m8 = (a03-a13)*b3
    #     That's 8 mults.
    #     
    #     Alternatively, schoolbook is also 8 mults. So no gain here.
    #   
    #   For C22 (2x2 = 2x4 * 4x2):
    #     Split A into A1 (2x2) and A2 (2x2)
    #     C22 = A1*B22_top + A2*B22_bot
    #     Use Strassen for each: 7+7 = 14 mults
    # 
    # Total: 14 + 8 + 14 = 36 mults. Same as before.
    # 
    # Let me try a different approach: use a more global optimization.
    # 
    # Actually, let's try to use the fact that n=2 to get a better bound.
    # The tensor rank of 2x4 * 4x5 is at most 2*4*5/2 = 20? No, that's not right.
    # 
    # Let me think about this differently. We have:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # 
    # We can group by k:
    # For each k, we compute a 2x1 vector c_k = A * B[:,k]
    # This is a 2x4 * 4x1 multiplication, which requires 8 mults (schoolbook).
    # For 5 values of k, that's 40 mults.
    # 
    # But we can share computations across k if we precompute linear combinations of A's columns.
    # 
    # Let's try: Precompute 4 linear combinations of A's rows:
    #   s1 = A[0,:] + A[1,:]  (element-wise)
    #   s2 = A[0,:] - A[1,:]
    #   s3 = A[0,:]
    #   s4 = A[1,:]
    # 
    # Then for each k:
    #   c0[k] = A[0,:] . B[:,k]
    #   c1[k] = A[1,:] . B[:,k]
    #   c0[k] + c1[k] = s1 . B[:,k]
    #   c0[k] - c1[k] = s2 . B[:,k]
    # 
    # So we need:
    #   m1_k = s1 . B[:,k] for k=0..4 (5 mults if we do it naively, but each is a dot product of length 4)
    #   m2_k = s2 . B[:,k] for k=0..4
    # 
    # Wait, that's not helping. Each dot product of length 4 requires 4 mults.
    # 
    # Let me just stick with the current approach but see if I can reduce the rank.
    # 
    # Actually, let me try a completely different decomposition:
    # Use the fact that 2x4 * 4x5 can be seen as 2*(4x5) matrix-vector products.
    # 
    # Alternative: Use a Kronecker product approach.
    # A is 2x4 = (2x2) ⊗ (1x2)? No, that doesn't work cleanly.
    # 
    # Let me just verify the current solution is correct and see if I can find a better one.
    # 
    # For now, let's keep the same structure but try to optimize the 2x1 case.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_00 = b_row_offset * p + b_col_offset
        v_idx_01 = b_row_offset * p + b_col_offset + 1
        v_idx_10 = (b_row_offset + 1) * p + b_col_offset
        v_idx_11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        w_idx_00 = c_col_offset * n + c_row_offset
        w_idx_01 = (c_col_offset + 1) * n + c_row_offset
        w_idx_10 = c_col_offset * n + c_row_offset + 1
        w_idx_11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        # M1
        U[u_idx_00][col] += 1
        U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M2
        U[u_idx_10][col] += 1
        U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1
        W[w_idx_10][col] += 1
        W[w_idx_11][col] -= 1
        col += 1
        
        # M3
        U[u_idx_00][col] += 1
        V[v_idx_01][col] += 1
        V[v_idx_11][col] -= 1
        W[w_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M4
        U[u_idx_11][col] += 1
        V[v_idx_10][col] += 1
        V[v_idx_00][col] -= 1
        W[w_idx_00][col] += 1
        W[w_idx_10][col] += 1
        col += 1
        
        # M5
        U[u_idx_00][col] += 1
        U[u_idx_01][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] -= 1
        W[w_idx_01][col] += 1
        col += 1
        
        # M6
        U[u_idx_10][col] += 1
        U[u_idx_00][col] -= 1
        V[v_idx_00][col] += 1
        V[v_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M7
        U[u_idx_01][col] += 1
        U[u_idx_11][col] -= 1
        V[v_idx_10][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        col += 1

    def add_2x1_optimized(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2 (rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1)
        # B is 2x1 (rows b_row_offset..b_row_offset+1, col b_col_offset)
        # C is 2x1 (rows c_row_offset..c_row_offset+1, col c_col_offset)
        # Use 4 mults with precomputation:
        # m1 = (A00+A10) * B0
        # m2 = (A01+A11) * B1
        # m3 = (A00-A10) * B0
        # m4 = (A01-A11) * B1
        # C0 = (m1+m2+m3+m4)/2 ... no
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # C0+C1 = (A00+A10)*B0 + (A01+A11)*B1 = m1+m2
        # C0-C1 = (A00-A10)*B0 + (A01-A11)*B1 = m3+m4
        # So C0 = (m1+m2+m3+m4)/2, C1 = (m1+m2-m3-m4)/2
        # This uses 4 mults but requires division by 2, which is allowed (half-integer entries).
        
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_0 = b_row_offset * p + b_col_offset
        v_idx_1 = (b_row_offset + 1) * p + b_col_offset
        
        w_idx_0 = c_col_offset * n + c_row_offset
        w_idx_1 = c_col_offset * n + c_row_offset + 1
        
        # m1 = (A00+A10) * B0
        U[u_idx_00][col] = 1
        U[u_idx_10][col] = 1
        V[v_idx_0][col] = 1
        W[w_idx_0][col] = 0.5
        W[w_idx_1][col] = 0.5
        col += 1
        
        # m2 = (A01+A11) * B1
        U[u_idx_01][col] = 1
        U[u_idx_11][col] = 1
        V[v_idx_1][col] = 1
        W[w_idx_0][col] = 0.5
        W[w_idx_1][col] = 0.5
        col += 1
        
        # m3 = (A00-A10) * B0
        U[u_idx_00][col] = 1
        U[u_idx_10][col] = -1
        V[v_idx_0][col] = 1
        W[w_idx_0][col] = 0.5
        W[w_idx_1][col] = -0.5
        col += 1
        
        # m4 = (A01-A11) * B1
        U[u_idx_01][col] = 1
        U[u_idx_11][col] = -1
        V[v_idx_1][col] = 1
        W[w_idx_0][col] = 0.5
        W[w_idx_1][col] = -0.5
        col += 1

    # Part 1: C1 = A B1 (2x2)
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Part 2: C2 = A B2 (2x3)
    # C21 = A B21 (2x1) using optimized 2x1
    add_2x1_optimized(0, 0, 0, 2, 0, 2)
    add_2x1_optimized(0, 2, 2, 2, 0, 2)
    
    # C22 = A B22 (2x2)
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
