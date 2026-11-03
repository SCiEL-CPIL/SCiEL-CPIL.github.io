"""Matrix multiplication 2x4 by 4x8 via Strassen-based block decomposition.

Strategy:
  Split A (2x4) into two 2x2 blocks: A = [A0 | A1]
  Split B (4x8) into four 2x2 blocks: B = [[B00, B01, B02, B03],
                                            [B10, B11, B12, B13]]
  Output C (2x8) = [A0*B00 + A1*B10 | A0*B01 + A1*B11 | A0*B02 + A1*B12 | A0*B03 + A1*B13]

  For each output column j, compute C[:, j] = A0 * B[0:2, j] + A1 * B[2:4, j]
  where B[0:2, j] and B[2:4, j] are 2x1 vectors... 

  Actually, let's think of it as block matrix multiplication:
  C = [A0, A1] * [[B00, B01, B02, B03],
                   [B10, B11, B12, B13]]
  
  where A0, A1 are 2x2 and B00...B13 are 2x2 blocks.
  
  So C = A0*B0 + A1*B1 where B0 = [B00, B01, B02, B03] (2x8) and B1 = [B10, B11, B12, B13] (2x8).
  
  This is a sum of two 2x2 * 2x8 multiplications.
  
  For each 2x2 * 2x8 multiplication, we can split the 2x8 into four 2x2 blocks and use Strassen on each.
  
  So for A0 * B0: B0 has 4 columns of 2x2 blocks (B00, B01, B02, B03).
  Each 2x2 * 2x2 product uses Strassen's 7 multiplications.
  Total for A0 * B0: 4 * 7 = 28 multiplications.
  Similarly for A1 * B1: 4 * 7 = 28 multiplications.
  Total rank = 56.
  
  This is better than the schoolbook 64.
"""

def construct():
    n, m, p = 2, 4, 8
    rank = 56  # 4 blocks * 2 sides * 7 Strassen multiplications
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen's 7 multiplication formulas for 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    def add_strassen_block(u_base, v_base, w_base, col_start):
        """
        u_base: (i, j) where i is row of A (0 or 1), j is col of A (0 or 1)
                 U index = i * m + j = i * 4 + j
        v_base: (j, k) where j is row of B (0..3), k is col of B (0..7)
                 V index = j * p + k = j * 8 + k
        w_base: (k, i) where k is col of C (0..7), i is row of C (0 or 1)
                 W index = k * n + i = k * 2 + i
        
        This computes C_block = A_block * B_block where A_block is 2x2 and B_block is 2x2.
        A_block uses U indices: (i*4+j) for i in {0,1}, j in {0,1}
        B_block uses V indices: (j*8+k) for j in {0,1}, k in {0,1}
        C_block uses W indices: (k*2+i) for k in {0,1}, i in {0,1}
        """
        # Strassen multiplications
        # m1 = (a+d)(e+h)
        # U: a=(0,0), d=(1,1); V: e=(0,0), h=(1,1)
        # W contributions: C[0,0] += m1, C[1,1] -= m1... wait let me be careful.
        
        # C = [[m1+m4-m5+m7, m3+m5],
        #      [m2+m4,         m1-m2+m3+m6]]
        
        # m1 = (a+d)(e+h)
        col = col_start
        U[0 * 4 + 0][col] += 1  # a
        U[1 * 4 + 1][col] += 1  # d
        V[0 * 8 + 0][col] += 1  # e
        V[1 * 8 + 1][col] += 1  # h
        W[0 * 2 + 0][col] += 1  # C[0,0] += m1
        W[0 * 2 + 1][col] += 0  # C[0,1]
        W[1 * 2 + 0][col] += 0  # C[1,0]
        W[1 * 2 + 1][col] += 1  # C[1,1] += m1
        col += 1
        
        # m2 = (c+d)e
        U[1 * 4 + 0][col] += 1  # c
        U[1 * 4 + 1][col] += 1  # d
        V[0 * 8 + 0][col] += 1  # e
        W[1 * 2 + 0][col] += 1  # C[1,0] += m2
        col += 1
        
        # m3 = a(f-h)
        U[0 * 4 + 0][col] += 1  # a
        V[0 * 8 + 1][col] += 1  # f
        V[1 * 8 + 1][col] += -1 # -h
        W[0 * 2 + 1][col] += 1  # C[0,1] += m3
        W[1 * 2 + 1][col] += 1  # C[1,1] += m3
        col += 1
        
        # m4 = d(g-e)
        U[1 * 4 + 1][col] += 1  # d
        V[1 * 8 + 0][col] += 1  # g
        V[0 * 8 + 0][col] += -1 # -e
        W[0 * 2 + 0][col] += 1  # C[0,0] += m4
        W[1 * 2 + 0][col] += 1  # C[1,0] += m4
        col += 1
        
        # m5 = (a+b)h
        U[0 * 4 + 0][col] += 1  # a
        U[0 * 4 + 1][col] += 1  # b
        V[1 * 8 + 1][col] += 1  # h
        W[0 * 2 + 0][col] += -1 # C[0,0] -= m5
        W[0 * 2 + 1][col] += 1  # C[0,1] += m5
        col += 1
        
        # m6 = (c-a)(e+f)
        U[1 * 4 + 0][col] += 1  # c
        U[0 * 4 + 0][col] += -1 # -a
        V[0 * 8 + 0][col] += 1  # e
        V[0 * 8 + 1][col] += 1  # f
        W[1 * 2 + 1][col] += 1  # C[1,1] += m6
        col += 1
        
        # m7 = (b-d)(g+h)
        U[0 * 4 + 1][col] += 1  # b
        U[1 * 4 + 1][col] += -1 # -d
        V[1 * 8 + 0][col] += 1  # g
        V[1 * 8 + 1][col] += 1  # h
        W[0 * 2 + 0][col] += 1  # C[0,0] += m7
        col += 1
        
        return col
    
    # Now we need to apply this to each of the 8 blocks:
    # A has two 2x2 blocks: A0 (cols 0,1 of A) and A1 (cols 2,3 of A)
    # B has eight 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], ..., B[2:4, 6:8]
    
    # C[:, 0:2] = A0 * B[0:2, 0:2] + A1 * B[2:4, 0:2]
    # C[:, 2:4] = A0 * B[0:2, 2:4] + A1 * B[2:4, 2:4]
    # C[:, 4:6] = A0 * B[0:2, 4:6] + A1 * B[2:4, 4:6]
    # C[:, 6:8] = A0 * B[0:2, 6:8] + A1 * B[2:4, 6:8]
    
    col = 0
    # For output columns 0,1:
    # A0 * B[0:2, 0:2]: A block is A[:, 0:2], B block is B[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, col)  # A0 is at U rows 0-3, B[0:2,0:2] is at V rows 0-7
    # A1 * B[2:4, 0:2]: A1 is at U rows 4-7, B[2:4,0:2] is at V rows 16-23
    # Need to offset the Strassen formula for A1 and B[2:4, 0:2]
    # A1: rows 4,5,6,7 in U (i=0,j=0 -> 0*4+0=0... wait no)
    
    # Hmm, I need to be more careful with indices.
    # U index = i * m + j = i * 4 + j, where i is row of A (0,1), j is col of A (0,1,2,3)
    # A0 = A[:, 0:2]: rows 0-3 of U (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    # A1 = A[:, 2:4]: rows 4-7 of U (i=0,j=2; i=0,j=3; i=1,j=2; i=1,j=3)
    
    # V index = j * p + k = j * 8 + k, where j is row of B (0..3), k is col of B (0..7)
    # B[0:2, 0:2]: rows 0-7 of V (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    # B[0:2, 2:4]: rows 8-15 of V (j=0,k=2; j=0,k=3; j=1,k=2; j=1,k=3)
    # B[2:4, 0:2]: rows 16-23 of V (j=2,k=0; j=2,k=1; j=3,k=0; j=3,k=1)
    # etc.
    
    # W index = k * n + i = k * 2 + i, where k is col of C (0..7), i is row of C (0,1)
    # C[:, 0:2]: rows 0-3 of W (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
    # C[:, 2:4]: rows 4-7 of W
    # etc.
    
    # Let me redo this properly.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_2x2(u_row_offset, v_row_offset, w_row_offset, col_start):
        """
        u_row_offset: base row in U for the 2x2 A block (4 rows: offset, offset+1, offset+2, offset+3)
                      corresponds to A[i, j] at U[row_offset + i*2 + j]... 
                      Actually U[i*m + j] = U[i*4 + j]. For A0: i in {0,1}, j in {0,1} -> U[0],U[1],U[4],U[5]
                      For A1: i in {0,1}, j in {2,3} -> U[2],U[3],U[6],U[7]
                      
        Let me define: the 2x2 A block has entries at U positions:
        a = U[u_base + 0], b = U[u_base + 1], c = U[u_base + 2], d = U[u_base + 3]
        where u_base is chosen such that:
        For A0: a=U[0*4+0]=U[0], b=U[0*4+1]=U[1], c=U[1*4+0]=U[4], d=U[1*4+1]=U[5]
        So the indices are not contiguous! 
        
        I need a different approach. Let me just enumerate all 7 Strassen multiplications
        for each of the 8 block products explicitly.
        """
        pass
    
    # Let me just do it manually for each of the 8 block products.
    # Each block product is A_block (2x2) * B_block (2x2) -> C_block (2x2)
    # Using Strassen's 7 multiplications.
    
    # Block products needed:
    # 1. A0 * B[0:2, 0:2] -> C[:, 0:2]
    # 2. A1 * B[2:4, 0:2] -> C[:, 0:2]
    # 3. A0 * B[0:2, 2:4] -> C[:, 2:4]
    # 4. A1 * B[2:4, 2:4] -> C[:, 2:4]
    # 5. A0 * B[0:2, 4:6] -> C[:, 4:6]
    # 6. A1 * B[2:4, 4:6] -> C[:, 4:6]
    # 7. A0 * B[0:2, 6:8] -> C[:, 6:8]
    # 8. A1 * B[2:4, 6:8] -> C[:, 6:8]
    
    # U indices for A0: a=U[0], b=U[1], c=U[4], d=U[5]
    # U indices for A1: a=U[2], b=U[3], c=U[6], d=U[7]
    
    # V indices for B[0:2, 0:2]: e=V[0], f=V[1], g=V[8], h=V[9]
    # V indices for B[0:2, 2:4]: e=V[2], f=V[3], g=V[10], h=V[11]
    # V indices for B[0:2, 4:6]: e=V[4], f=V[5], g=V[12], h=V[13]
    # V indices for B[0:2, 6:8]: e=V[6], f=V[7], g=V[14], h=V[15]
    # V indices for B[2:4, 0:2]: e=V[16], f=V[17], g=V[24], h=V[25]
    # V indices for B[2:4, 2:4]: e=V[18], f=V[19], g=V[26], h=V[27]
    # V indices for B[2:4, 4:6]: e=V[20], f=V[21], g=V[28], h=V[29]
    # V indices for B[2:4, 6:8]: e=V[22], f=V[23], g=V[30], h=V[31]
    
    # W indices for C[:, 0:2]: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    # W indices for C[:, 2:4]: C[0,0]=W[4], C[0,1]=W[5], C[1,0]=W[6], C[1,1]=W[7]
    # W indices for C[:, 4:6]: C[0,0]=W[8], C[0,1]=W[9], C[1,0]=W[10], C[1,1]=W[11]
    # W indices for C[:, 6:8]: C[0,0]=W[12], C[0,1]=W[13], C[1,0]=W[14], C[1,1]=W[15]
    
    def add_strassen(a_u, b_u, c_u, d_u, e_v, f_v, g_v, h_v, c00_w, c01_w, c10_w, c11_w, col):
        """Add Strassen's 7 multiplications."""
        # m1 = (a+d)(e+h)
        U[a_u][col] += 1
        U[d_u][col] += 1
        V[e_v][col] += 1
        V[h_v][col] += 1
        W[c00_w][col] += 1
        W[c11_w][col] += 1
        col += 1
        
        # m2 = (c+d)e
        U[c_u][col] += 1
        U[d_u][col] += 1
        V[e_v][col] += 1
        W[c10_w][col] += 1
        col += 1
        
        # m3 = a(f-h)
        U[a_u][col] += 1
        V[f_v][col] += 1
        V[h_v][col] += -1
        W[c01_w][col] += 1
        W[c11_w][col] += 1
        col += 1
        
        # m4 = d(g-e)
        U[d_u][col] += 1
        V[g_v][col] += 1
        V[e_v][col] += -1
        W[c00_w][col] += 1
        W[c10_w][col] += 1
        col += 1
        
        # m5 = (a+b)h
        U[a_u][col] += 1
        U[b_u][col] += 1
        V[h_v][col] += 1
        W[c00_w][col] += -1
        W[c01_w][col] += 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[c_u][col] += 1
        U[a_u][col] += -1
        V[e_v][col] += 1
        V[f_v][col] += 1
        W[c11_w][col] += 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[b_u][col] += 1
        U[d_u][col] += -1
        V[g_v][col] += 1
        V[h_v][col] += 1
        W[c00_w][col] += 1
        col += 1
        
        return col
    
    col = 0
    # 1. A0 * B[0:2, 0:2] -> C[:, 0:2]
    col = add_strassen(0, 1, 4, 5, 0, 1, 8, 9, 0, 1, 2, 3, col)
    # 2. A1 * B[2:4, 0:2] -> C[:, 0:2]
    col = add_strassen(2, 3, 6, 7, 16, 17, 24, 25, 0, 1, 2, 3, col)
    # 3. A0 * B[0:2, 2:4] -> C[:, 2:4]
    col = add_strassen(0, 1, 4, 5, 2, 3, 10, 11, 4, 5, 6, 7, col)
    # 4. A1 * B[2:4, 2:4] -> C[:, 2:4]
    col = add_strassen(2, 3, 6, 7, 18, 19, 26, 27, 4, 5, 6, 7, col)
    # 5. A0 * B[0:2, 4:6] -> C[:, 4:6]
    col = add_strassen(0, 1, 4, 5, 4, 5, 12, 13, 8, 9, 10, 11, col)
    # 6. A1 * B[2:4, 4:6] -> C[:, 4:6]
    col = add_strassen(2, 3, 6, 7, 20, 21, 28, 29, 8, 9, 10, 11, col)
    # 7. A0 * B[0:2, 6:8] -> C[:, 6:8]
    col = add_strassen(0, 1, 4, 5, 6, 7, 14, 15, 12, 13, 14, 15, col)
    # 8. A1 * B[2:4, 6:8] -> C[:, 6:8]
    col = add_strassen(2, 3, 6, 7, 22, 23, 30, 31, 12, 13, 14, 15, col)
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
