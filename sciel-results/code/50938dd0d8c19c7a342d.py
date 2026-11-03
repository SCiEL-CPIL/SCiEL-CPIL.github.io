"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-17 decomposition of the 2x3 x 3x3 matrix multiplication tensor.
    
    Uses Strassen's 7-product algorithm for the 2x2 core embedded in the 2x3 problem,
    plus 10 additional products for the remaining columns, totaling 17.
    
    The multiplication tensor T has entries T[(i,j,k), (a,b,c)] = 1 if i==a and j==b and k==c.
    We need U, V, W such that for all (i,j,k), (a,b,c):
      T[(i,j,k), (a,b,c)] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Where n=2, m=3, p=3.
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (9, rank)  
    W has shape (p*n, rank) = (6, rank)
    
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    
    # We'll build a rank-17 decomposition.
    # Strategy: The 2x3 x 3x3 multiplication can be seen as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # For the 2x2 block (i in {0,1}, j in {0,1}), use Strassen (7 products)
    # For the remaining column j=2, we need 2 more products (one per row i)
    # But we also need to handle the interaction...
    #
    # Actually, let me think more carefully.
    # 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    #
    # The known result is that the rank of 2x3 x 3x3 matrix multiplication is 17.
    # This comes from embedding Strassen's 7-product scheme for the 2x2 part
    # and adding 10 products for the third column/row interactions.
    #
    # Let me construct this explicitly.
    #
    # Strassen's 7 products for 2x2:
    # Given A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # 
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now for the full 2x3 x 3x3, we have an extra column j=2 in A and rows 2 in B.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Let me define the 17 rank-1 factors.
    #
    # Factors 1-7: Strassen products for the 2x2 sub-block (j in {0,1})
    # Factor 8-9: Products involving A[:,2] with B[2,:] for each row
    # Factor 10-17: Additional cross terms
    
    # Actually, let me use a known explicit construction.
    # The rank of [2,3,3] is 17. Here's one construction:
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    # U indices: i*m+j where i in [0,1], j in [0,2]
    # So U[0]=i0j0, U[1]=i0j1, U[2]=i0j2, U[3]=i1j0, U[4]=i1j1, U[5]=i1j2
    # V indices: j*p+k where j in [0,2], k in [0,2]
    # So V[0]=j0k0, V[1]=j0k1, V[2]=j0k2, V[3]=j1k0, V[4]=j1k1, V[5]=j1k2, V[6]=j2k0, V[7]=j2k1, V[8]=j2k2
    # W indices: k*n+i where k in [0,2], i in [0,1]
    # So W[0]=k0i0, W[1]=k0i1, W[2]=k1i0, W[3]=k1i1, W[4]=k2i0, W[5]=k2i1
    
    # Strassen's 7 products for the 2x2 core (rows 0,1 of A; cols 0,1 of B; output rows 0,1)
    # These handle contributions from j=0 and j=1 to C[i,k] for k=0,1
    
    # M1 = (a00+a11)*(b00+b11) -> contributes to C00, C11
    # U: a00 (idx 0), a11 (idx 4); V: b00 (idx 0), b11 (idx 4); W: k0i0 (idx 0), k1i1 (idx 3)
    U[0][0] = 1  # a00
    U[4][0] = 1  # a11
    V[0][0] = 1  # b00
    V[4][0] = 1  # b11
    W[0][0] = 1  # k0i0
    W[3][0] = 1  # k1i1
    
    # M2 = (a10+a11)*b00 -> contributes to C10, C11
    U[3][1] = 1  # a10
    U[4][1] = 1  # a11
    V[0][1] = 1  # b00
    W[1][1] = 1  # k0i1
    W[3][1] = -1 # k1i1 (negative because C11 = M1 - M2 + ...)
    
    # M3 = a00*(b01-b11) -> contributes to C01, C11
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[4][2] = -1 # b11
    W[2][2] = 1  # k1i0
    W[3][2] = 1  # k1i1
    
    # M4 = a11*(b10-b00) -> contributes to C00, C10
    U[4][3] = 1  # a11
    V[3][3] = 1  # b10
    V[0][3] = -1 # b00
    W[0][3] = 1  # k0i0
    W[1][3] = 1  # k0i1
    
    # M5 = (a00+a01)*b11 -> contributes to C00, C01
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[4][4] = 1  # b11
    W[0][4] = -1 # k0i0 (negative: C00 = M1 + M4 - M5 + M7)
    W[2][4] = 1  # k1i0 (C01 = M3 + M5)
    
    # M6 = (a10-a00)*(b00+b01) -> contributes to C11
    U[3][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    W[3][5] = 1  # k1i1
    
    # M7 = (a01-a11)*(b10+b11) -> contributes to C00, C11
    U[1][6] = 1  # a01
    U[4][6] = -1 # a11
    V[3][6] = 1  # b10
    V[4][6] = 1  # b11
    W[0][6] = 1  # k0i0 (C00 = ... + M7)
    W[3][6] = 1  # k1i1 (C11 = ... + M6, but wait let me recheck)
    
    # Let me verify the Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Wait, I need to be more careful. Let me redo this.
    # 
    # Standard Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So for the W matrix (output coefficients):
    # M1: contributes +1 to C00 (W[0]) and +1 to C11 (W[3])
    # M2: contributes +1 to C10 (W[1]) and -1 to C11 (W[3])
    # M3: contributes +1 to C01 (W[2]) and +1 to C11 (W[3])
    # M4: contributes +1 to C00 (W[0]) and +1 to C10 (W[1])
    # M5: contributes -1 to C00 (W[0]) and +1 to C01 (W[2])
    # M6: contributes +1 to C11 (W[3])
    # M7: contributes +1 to C00 (W[0]) and +1 to C11 (W[3])
    
    # Let me redo all 7 factors with correct W coefficients:
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Factor 0: M1 = (a00+a11)*(b00+b11)
    U[0][0] = 1  # a00
    U[4][0] = 1  # a11
    V[0][0] = 1  # b00
    V[4][0] = 1  # b11
    W[0][0] = 1  # C00 += M1
    W[3][0] = 1  # C11 += M1
    
    # Factor 1: M2 = (a10+a11)*b00
    U[3][1] = 1  # a10
    U[4][1] = 1  # a11
    V[0][1] = 1  # b00
    W[1][1] = 1  # C10 += M2
    W[3][1] = -1 # C11 -= M2
    
    # Factor 2: M3 = a00*(b01-b11)
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[4][2] = -1 # b11
    W[2][2] = 1  # C01 += M3
    W[3][2] = 1  # C11 += M3
    
    # Factor 3: M4 = a11*(b10-b00)
    U[4][3] = 1  # a11
    V[3][3] = 1  # b10
    V[0][3] = -1 # b00
    W[0][3] = 1  # C00 += M4
    W[1][3] = 1  # C10 += M4
    
    # Factor 4: M5 = (a00+a01)*b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[4][4] = 1  # b11
    W[0][4] = -1 # C00 -= M5
    W[2][4] = 1  # C01 += M5
    
    # Factor 5: M6 = (a10-a00)*(b00+b01)
    U[3][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    W[3][5] = 1  # C11 += M6
    
    # Factor 6: M7 = (a01-a11)*(b10+b11)
    U[1][6] = 1  # a01
    U[4][6] = -1 # a11
    V[3][6] = 1  # b10
    V[4][6] = 1  # b11
    W[0][6] = 1  # C00 += M7
    W[3][6] = 1  # C11 += M7
    
    # Now factors 7-16 handle the third column j=2 and third row k=2.
    # C[i,k] for k=2: C[0,2] = a00*b02 + a01*b12 + a02*b22
    #                  C[1,2] = a10*b02 + a11*b12 + a12*b22
    # C[i,0] additional: C[0,0] += a02*b20, C[1,0] += a12*b20
    # C[i,1] additional: C[0,1] += a02*b21, C[1,1] += a12*b21
    #
    # We need 10 more factors for the 2x3 interaction with the third index.
    # 
    # For k=2 (output column 2):
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    #
    # For k=0,1 additional terms from j=2:
    # C[0,0] += a02*b20
    # C[0,1] += a02*b21
    # C[1,0] += a12*b20
    # C[1,1] += a12*b21
    #
    # That's 6 terms for k=2 and 4 terms for the j=2 contributions to k=0,1.
    # Total = 10, giving 7+10=17.
    
    # Factor 7: a00*b02 -> C[0,2]
    U[0][7] = 1  # a00
    V[2][7] = 1  # b02
    W[4][7] = 1  # C[0,2] (k=2,i=0)
    
    # Factor 8: a01*b12 -> C[0,2]
    U[1][8] = 1  # a01
    V[5][8] = 1  # b12
    W[4][8] = 1  # C[0,2]
    
    # Factor 9: a02*b22 -> C[0,2]
    U[2][9] = 1  # a02
    V[8][9] = 1  # b22
    W[4][9] = 1  # C[0,2]
    
    # Factor 10: a10*b02 -> C[1,2]
    U[3][10] = 1  # a10
    V[2][10] = 1  # b02
    W[5][10] = 1  # C[1,2] (k=2,i=1)
    
    # Factor 11: a11*b12 -> C[1,2]
    U[4][11] = 1  # a11
    V[5][11] = 1  # b12
    W[5][11] = 1  # C[1,2]
    
    # Factor 12: a12*b22 -> C[1,2]
    U[5][12] = 1  # a12
    V[8][12] = 1  # b22
    W[5][12] = 1  # C[1,2]
    
    # Factor 13: a02*b20 -> C[0,0]
    U[2][13] = 1  # a02
    V[6][13] = 1  # b20
    W[0][13] = 1  # C[0,0]
    
    # Factor 14: a02*b21 -> C[0,1]
    U[2][14] = 1  # a02
    V[7][14] = 1  # b21
    W[2][14] = 1  # C[0,1]
    
    # Factor 15: a12*b20 -> C[1,0]
    U[5][15] = 1  # a12
    V[6][15] = 1  # b20
    W[1][15] = 1  # C[1,0]
    
    # Factor 16: a12*b21 -> C[1,1]
    U[5][16] = 1  # a12
    V[7][16] = 1  # b21
    W[3][16] = 1  # C[1,1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
