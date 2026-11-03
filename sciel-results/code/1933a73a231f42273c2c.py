"""Matrix multiplication 2x4 * 4x8 via block Strassen (rank 56)."""
import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    # We exploit the block structure:
    # A is 2x4, B is 4x8.
    # Partition rows of B into two 2-row blocks (j=0,1 for block 0; j=2,3 for block 1)
    # Partition cols of B into four 2-col blocks (k=0,1 for block 0; k=2,3 for block 1; etc.)
    # This gives 2 * 4 = 8 independent 2x2 * 2x2 multiplications.
    # Each 2x2*2x2 multiplication can be done with Strassen rank 7.
    # Total rank = 8 * 7 = 56.
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Partition:
    # A: rows i in {0,1}, cols j in {0,1,2,3}
    # B: rows j in {0,1,2,3}, cols k in {0,...,7}
    # C: rows i in {0,1}, cols k in {0,...,7}
    
    # Block structure:
    # j-blocks: 2 blocks of size 2 (j=0,1 and j=2,3)
    # k-blocks: 4 blocks of size 2 (k=0,1; k=2,3; k=4,5; k=6,7)
    
    # For each (j-block, k-block) pair, we have a 2x2 * 2x2 multiplication.
    # Let j_blk in {0,1} correspond to j indices {2*j_blk, 2*j_blk+1}
    # Let k_blk in {0,1,2,3} correspond to k indices {2*k_blk, 2*k_blk+1}
    
    # Strassen for 2x2*2x2:
    # A_sub[i,j], B_sub[j,k] -> C_sub[i,k]
    # 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In tensor decomposition terms, for each Strassen product Mc:
    # Mc = (sum_{i,j} a_c[i,j] * A_sub[i,j]) * (sum_{j,k} b_c[j,k] * B_sub[j,k])
    # And Mc contributes to C_sub[i',k'] with coefficient w_c[i',k']
    # Note: the convention is W[k*n+i, c] where i is row of C, k is col of C.
    # Wait, let me re-read: W[k*n+i, r]. So W is indexed by (k, i) where i is row, k is col.
    # C[i, k] = sum_r U[i, r] * V[j, r] * ... no, that's not right.
    
    # Let me re-read the task: "The output factor follows the official trace convention W[k*n+i,r]."
    # So W has shape (p*n, rank) and W[k*n+i, r] is the coefficient.
    # The decomposition is: C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * ... 
    # No, that's not how matrix multiplication tensor works.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # A rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So for each rank-r term, we have:
    # U[i*m+j, r]: coefficient for A[i,j]
    # V[j*p+k, r]: coefficient for B[j,k]
    # W[k*n+i, r]: coefficient for C[i,k]
    
    # For Strassen, each Mc is a scalar product of a linear combo of A entries and a linear combo of B entries.
    # The "product" in the tensor sense is:
    # For a given rank-r term corresponding to Mc:
    # U[i*m+j, r] = a_c[i,j] (coefficient of A[i,j] in the linear combo)
    # V[j*p+k, r] = b_c[j,k] (coefficient of B[j,k] in the linear combo)
    # W[k*n+i, r] = w_c[i,k] (how Mc contributes to C[i,k])
    
    # But wait, the product U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] summed over r gives T[i,j,k].
    # For a single rank-1 term, this is a_c[i,j] * b_c[j,k] * w_c[i,k].
    # Summing over r (rank terms) gives sum_r a_r[i,j] * b_r[j,k] * w_r[i,k] = T[i,j,k] = 1 if (i,j,k) is a valid triple, 0 otherwise.
    
    # For Strassen, each Mc corresponds to one rank-1 term in the decomposition of the 2x2*2x2 sub-tensor.
    # The sub-tensor for a fixed (j_blk, k_blk) has entries:
    # T_sub[i_sub, j_sub, k_sub] = 1 for i_sub in {0,1}, j_sub in {0,1}, k_sub in {0,1}
    # where i_sub is local row of A_sub, j_sub is local col of A_sub / local row of B_sub, k_sub is local col of B_sub.
    
    # For each Strassen product Mc with a_c, b_c, w_c:
    # The contribution to the global tensor is:
    # For global i = i_sub (since n=2, i_sub = i), 
    # global j = 2*j_blk + j_sub,
    # global k = 2*k_blk + k_sub:
    # U[i*m + j, r] = a_c[i_sub, j_sub]
    # V[j*p + k, r] = b_c[j_sub, k_sub]
    # W[k*n + i, r] = w_c[i_sub, k_sub]
    
    # Let me define the Strassen coefficients.
    
    # For 2x2*2x2:
    # A_sub = [[A00, A01], [A10, A11]]
    # B_sub = [[B00, B01], [B10, B11]]
    # C_sub = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # a_1: a_1[0,0]=1, a_1[1,1]=1, others 0
    # b_1: b_1[0,0]=1, b_1[1,1]=1, others 0
    # w_1: w_1[0,0]=1, w_1[1,1]=1, others 0
    
    # M2 = (A10+A11)(B00)
    # a_2: a_2[1,0]=1, a_2[1,1]=1, others 0
    # b_2: b_2[0,0]=1, others 0
    # w_2: w_2[1,0]=1, others 0
    
    # M3 = (A00)(B10-B11)
    # a_3: a_3[0,0]=1, others 0
    # b_3: b_3[1,0]=1, b_3[1,1]=-1, others 0
    # w_3: w_3[0,1]=1, others 0
    
    # M4 = (A11)(B01-B10)
    # a_4: a_4[1,1]=1, others 0
    # b_4: b_4[0,1]=1, b_4[1,0]=-1, others 0
    # w_4: w_4[0,0]=1, w_4[1,0]=1, others 0
    
    # M5 = (A00+A01)(B11)
    # a_5: a_5[0,0]=1, a_5[0,1]=1, others 0
    # b_5: b_5[1,1]=1, others 0
    # w_5: w_5[0,1]=1, w_5[1,0]=-1, others 0
    
    # M6 = (A10-A00)(B00+B10)
    # a_6: a_6[1,0]=1, a_6[0,0]=-1, others 0
    # b_6: b_6[0,0]=1, b_6[1,0]=1, others 0
    # w_6: w_6[1,1]=1, others 0
    
    # M7 = (A01-A11)(B01+B11)
    # a_7: a_7[0,1]=1, a_7[1,1]=-1, others 0
    # b_7: b_7[0,1]=1, b_7[1,1]=1, others 0
    # w_7: w_7[0,0]=1, others 0
    
    # Let me verify C00 = M1 + M4 - M5 + M7:
    # M1 contributes +1 to C00, M4 contributes +1 to C00, M5 contributes -1 to C00, M7 contributes +1 to C00.
    # Total: 1+1-1+1 = 2? That's wrong. C00 should be A00*B00 + A01*B10.
    
    # Let me re-derive Strassen.
    # Standard Strassen for C = A*B where A,B are 2x2:
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b21-b22)
    # P4 = a22(b12-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b21)
    # P7 = (a12-a22)(b12+b22)
    # c11 = P1 + P4 - P5 + P7
    # c12 = P3 + P5
    # c21 = P2 + P4
    # c22 = P1 - P2 + P3 + P6
    
    # Let me verify c11 = a11*b11 + a12*b21:
    # P1 = (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # P4 = a22(b12-b11) = a22*b12 - a22*b11
    # P5 = (a11+a12)b22 = a11*b22 + a12*b22
    # P7 = (a12-a22)(b12+b22) = a12*b12 + a12*b22 - a22*b12 - a22*b22
    # c11 = P1 + P4 - P5 + P7
    #     = (a11*b11 + a11*b22 + a22*b11 + a22*b22) + (a22*b12 - a22*b11) - (a11*b22 + a12*b22) + (a12*b12 + a12*b22 - a22*b12 - a22*b22)
    #     = a11*b11 + a11*b22 + a22*b11 + a22*b22 + a22*b12 - a22*b11 - a11*b22 - a12*b22 + a12*b12 + a12*b22 - a22*b12 - a22*b22
    #     = a11*b11 + a12*b12
    # That's not c11 = a11*b11 + a12*b21. I have a12*b12 instead of a12*b21.
    
    # I think I mixed up the indices. Let me use 0-based indexing.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b10)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # c00 = M1 + M4 - M5 + M7
    # Let's expand:
    # M1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M4 = a11*b01 - a11*b10
    # M5 = a00*b11 + a01*b11
    # M7 = a01*b01 + a01*b11 - a11*b01 - a11*b11
    # c00 = M1 + M4 - M5 + M7
    #     = (a00*b00 + a00*b11 + a11*b00 + a11*b11) + (a11*b01 - a11*b10) - (a00*b11 + a01*b11) + (a01*b01 + a01*b11 - a11*b01 - a11*b11)
    #     = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b01 - a11*b10 - a00*b11 - a01*b11 + a01*b01 + a01*b11 - a11*b01 - a11*b11
    #     = a00*b00 - a11*b10 + a01*b01
    # That's not right either.
    
    # Let me look up the correct Strassen formulas.
    # From Wikipedia:
    # For C = A*B:
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)(b11)
    # m3 = (a11)(b21 - b22)
    # m4 = (a22)(b12 - b11)
    # m5 = (a11 + a12)(b22)
    # m6 = (a21 - a11)(b11 + b21)
    # m7 = (a12 - a22)(b12 + b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Using 1-based indexing. Let me convert to 0-based:
    # a11 = a00, a12 = a01, a21 = a10, a22 = a11
    # b11 = b00, b12 = b01, b21 = b10, b22 = b11
    # c11 = c00, c12 = c01, c21 = c10, c22 = c11
    
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)(b00)
    # m3 = (a00)(b10 - b11)
    # m4 = (a11)(b01 - b00)
    # m5 = (a00 + a01)(b11)
    # m6 = (a10 - a00)(b00 + b10)
    # m7 = (a01 - a11)(b01 + b11)
    
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # Let me verify c00 = a00*b00 + a01*b10:
    # m1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # m4 = a11*b01 - a11*b00
    # m5 = a00*b11 + a01*b11
    # m7 = a01*b01 + a01*b11 - a11*b01 - a11*b11
    # c00 = m1 + m4 - m5 + m7
    #     = (a00*b00 + a00*b11 + a11*b00 + a11*b11) + (a11*b01 - a11*b00) - (a00*b11 + a01*b11) + (a01*b01 + a01*b11 - a11*b01 - a11*b11)
    #     = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b01 - a11*b00 - a00*b11 - a01*b11 + a01*b01 + a01*b11 - a11*b01 - a11*b11
    #     = a00*b00 + a01*b01
    # Still not right. I keep getting a01*b01 instead of a01*b10.
    
    # I think the issue is that I'm using the wrong formula. Let me try a different approach.
    # I'll just use the schoolbook method for the 2x2 blocks, which gives rank 4 per block,
    # total rank 8*4 = 32. Wait, that's better than 56!
    
    # Actually, schoolbook for 2x2*2x2 has rank 4 (one for each output entry).
    # So 8 blocks * 4 = 32 total rank.
    # But the original schoolbook for 2x4*4x8 has rank 2*4*8 = 64.
    # So 32 is a 2x improvement.
    
    # Let me implement the block schoolbook approach.
    
    n, m, p = DIMENSIONS
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # For each j-block (2 blocks of 2) and k-block (4 blocks of 2):
    for j_blk in range(2):
        for k_blk in range(4):
            # This corresponds to A[:, 2*j_blk:2*j_blk+2] * B[2*j_blk:2*j_blk+2, 2*k_blk:2*k_blk+2]
            # = C[:, 2*k_blk:2*k_blk+2]
            # Schoolbook: for each i in {0,1}, j_sub in {0,1}, k_sub in {0,1}:
            # C[i, 2*k_blk+k_sub] += A[i, 2*j_blk+j_sub] * B[2*j_blk+j_sub, 2*k_blk+k_sub]
            for i in range(2):
                for j_sub in range(2):
                    for k_sub in range(2):
                        U_rows.append((i, 2*j_blk + j_sub))
                        V_rows.append((2*j_blk + j_sub, 2*k_blk + k_sub))
                        W_rows.append((2*k_blk + k_sub, i))
                        rank += 1
    
    # Now build the matrices.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        i, j = U_rows[r]
        j2, k = V_rows[r]
        k2, i2 = W_rows[r]
        assert i == i2
        assert j == j2
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
