"""Strassen-based rank 28 decomposition for 2x4 x 4x4 matrix multiplication."""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    Uses Strassen's 7-product algorithm on 2x2 blocks, where each 2x2 block
    multiplication uses the 7 Strassen bilinear factors.
    
    n=2, m=4, p=4:
    U shape: (n*m, rank) = (8, 28)
    V shape: (m*p, rank) = (16, 28)
    W shape: (p*n, rank) = (8, 28)
    
    Convention: W[k*n + i, r]
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # We split A (2x4) into two 2x2 blocks: A = [A_L | A_R]
    # A_L = A[:, 0:2], A_R = A[:, 2:4]
    # B (4x4) into two 2x2 blocks: B = [B_TL | B_TR; B_BL | B_BR]
    # B_TL = B[0:2, 0:2], B_TR = B[0:2, 2:4]
    # B_BL = B[2:4, 0:2], B_BR = B[2:4, 2:4]
    
    # C = A * B = [A_L*B_TL + A_R*B_BL | A_L*B_TR + A_R*B_BR]
    # C_L = A_L * B_TL + A_R * B_BL (2x2)
    # C_R = A_L * B_TR + A_R * B_BR (2x2)
    
    # For each 2x2 matrix multiplication, Strassen uses 7 products.
    # For C_L: A_L * B_TL + A_R * B_BL
    # Let X = A_L + A_R (2x2), Y = A_R (2x2)
    # Then C_L = A_L * B_TL + A_R * B_BL
    # We can use Strassen on the pair (A_L, A_R) against (B_TL, B_BL)
    
    # Actually, let's think of this differently.
    # We have 4 scalar 2x2 multiplications:
    # C_L = A_L @ B_TL + A_R @ B_BL
    # C_R = A_L @ B_TR + A_R @ B_BR
    
    # Each 2x2 @ 2x2 can be done with 7 bilinear operations.
    # Total: 4 * 7 = 28 bilinear operations.
    
    # Let's define the Strassen decomposition for a single 2x2@2x2:
    # Given P (2x2) and Q (2x2), compute P@Q using 7 products:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q10-Q11)
    # M4 = P11*(Q00+Q01)
    # M5 = (P01-P11)*(Q00+Q10)
    # M6 = (P00-P01)*(Q11-Q10)
    # M7 = (P10+P00)*(Q11-Q01)
    
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # Now for our problem:
    # C_L[i,j] = sum over appropriate indices
    # C_R[i,j] = sum over appropriate indices
    
    # Let me index the U, V, W matrices properly.
    # U rows: index = i*m + j where i in [0,1] (row of A), j in [0,3] (col of A)
    # V rows: index = j*p + k where j in [0,3] (row of B), k in [0,3] (col of B)
    # W rows: index = k*n + i where k in [0,3] (row of C), i in [0,1] (col of C)
    
    # A is 2x4: A[i, j] for i in [0,1], j in [0,3]
    # B is 4x4: B[j, k] for j in [0,3], k in [0,3]
    # C is 2x4: C[i, k] for i in [0,1], k in [0,3]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split: A_L = A[:, 0:2] (cols 0,1), A_R = A[:, 2:4] (cols 2,3)
    # B_TL = B[0:2, 0:2], B_TR = B[0:2, 2:4]
    # B_BL = B[2:4, 0:2], B_BR = B[2:4, 2:4]
    
    # C[:, 0:2] = A_L @ B_TL + A_R @ B_BL  (this is C_L, 2x2)
    # C[:, 2:4] = A_L @ B_TR + A_R @ B_BR  (this is C_R, 2x2)
    
    # For C_L: we need to compute A_L @ B_TL + A_R @ B_BL
    # For C_R: we need to compute A_L @ B_TR + A_R @ B_BR
    
    # Each of these is a sum of two 2x2 matrix products.
    # We can use Strassen's trick: express sum of two products using 7 bilinear ops.
    
    # Actually, let's just use 7 Strassen ops for each of the 4 sub-products:
    # A_L @ B_TL: 7 ops
    # A_R @ B_BL: 7 ops
    # A_L @ B_TR: 7 ops
    # A_R @ B_BR: 7 ops
    # Total: 28 ops
    
    # But wait, that's just doing 4 separate Strassen decompositions.
    # Each 2x2 @ 2x2 uses 7 bilinear operations.
    
    # Let me define a helper to add Strassen factors for P @ Q where
    # P has entries at certain (i, j) positions and Q at certain (j, k) positions.
    
    # For a 2x2 matrix P with entries P[0,0]=a, P[0,1]=b, P[1,0]=c, P[1,1]=d
    # and Q with entries Q[0,0]=e, Q[0,1]=f, Q[1,0]=g, Q[1,1]=h:
    
    # The 7 bilinear forms are:
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(g-h)
    # M4 = d*(e+f)
    # M5 = (b-d)*(e+g)
    # M6 = (a-b)*(h-g)
    # M7 = (c+a)*(h-f)
    
    # Result:
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # Now, for A_L @ B_TL:
    # A_L entries: A[0,0], A[0,1], A[1,0], A[1,1]
    #   a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B_TL entries: B[0,0], B[0,1], B[1,0], B[1,1]
    #   e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # For A_R @ B_BL:
    # A_R entries: A[0,2], A[0,3], A[1,2], A[1,3]
    #   a=A[0,2], b=A[0,3], c=A[1,2], d=A[1,3]
    # B_BL entries: B[2,0], B[2,1], B[3,0], B[3,1]
    #   e=B[2,0], f=B[2,1], g=B[3,0], h=B[3,1]
    
    # For A_L @ B_TR:
    # A_L entries: A[0,0], A[0,1], A[1,0], A[1,1]
    #   a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B_TR entries: B[0,2], B[0,3], B[1,2], B[1,3]
    #   e=B[0,2], f=B[0,3], g=B[1,2], h=B[1,3]
    
    # For A_R @ B_BR:
    # A_R entries: A[0,2], A[0,3], A[1,2], A[1,3]
    #   a=A[0,2], b=A[0,3], c=A[1,2], d=A[1,3]
    # B_BR entries: B[2,2], B[2,3], B[3,2], B[3,3]
    #   e=B[2,2], f=B[2,3], g=B[3,2], h=B[3,3]
    
    def add_strassen_factors(offset, a_idx, b_idx, c_idx, d_idx, 
                               e_idx, f_idx, g_idx, h_idx,
                               r00_idx, r01_idx, r10_idx, r11_idx):
        """
        Add 7 Strassen bilinear factors for a 2x2 @ 2x2 multiplication.
        
        offset: column offset in the factor matrices
        a_idx, b_idx, c_idx, d_idx: row indices in U for A entries
          a = A[i0,j0], b = A[i0,j1], c = A[i1,j0], d = A[i1,j1]
          where i0,i1 are the two rows, j0,j1 are the two columns
        e_idx, f_idx, g_idx, h_idx: row indices in V for B entries
          e = B[j0,k0], f = B[j0,k1], g = B[j1,k0], h = B[j1,k1]
        r00_idx, r01_idx, r10_idx, r11_idx: row indices in W for result entries
          R00 = C[i0,k0], R01 = C[i0,k1], R10 = C[i1,k0], R11 = C[i1,k1]
        """
        # M1 = (a+d)*(e+h)
        U[a_idx][offset] += 1
        U[d_idx][offset] += 1
        V[e_idx][offset] += 1
        V[h_idx][offset] += 1
        W[r00_idx][offset] += 1
        W[r11_idx][offset] += 1
        
        # M2 = (c+d)*e
        U[c_idx][offset+1] += 1
        U[d_idx][offset+1] += 1
        V[e_idx][offset+1] += 1
        W[r10_idx][offset+1] += 1
        W[r11_idx][offset+1] -= 1
        
        # M3 = a*(g-h)
        U[a_idx][offset+2] += 1
        V[g_idx][offset+2] += 1
        V[h_idx][offset+2] -= 1
        W[r01_idx][offset+2] += 1
        W[r11_idx][offset+2] += 1
        
        # M4 = d*(e+f)
        U[d_idx][offset+3] += 1
        V[e_idx][offset+3] += 1
        V[f_idx][offset+3] += 1
        W[r00_idx][offset+3] += 1
        W[r10_idx][offset+3] += 1
        
        # M5 = (b-d)*(e+g)
        U[b_idx][offset+4] += 1
        U[d_idx][offset+4] -= 1
        V[e_idx][offset+4] += 1
        V[g_idx][offset+4] += 1
        W[r00_idx][offset+4] -= 1
        W[r01_idx][offset+4] += 1
        
        # M6 = (a-b)*(h-g)
        U[a_idx][offset+5] += 1
        U[b_idx][offset+5] -= 1
        V[h_idx][offset+5] += 1
        V[g_idx][offset+5] -= 1
        W[r11_idx][offset+5] += 1
        
        # M7 = (c+a)*(h-f)
        U[c_idx][offset+6] += 1
        U[a_idx][offset+6] += 1
        V[h_idx][offset+6] += 1
        V[f_idx][offset+6] -= 1
        W[r00_idx][offset+6] += 1
    
    # A_L @ B_TL:
    # A_L: rows 0,1; cols 0,1
    #   a=A[0,0] -> U[0*4+0]=U[0]
    #   b=A[0,1] -> U[0*4+1]=U[1]
    #   c=A[1,0] -> U[1*4+0]=U[4]
    #   d=A[1,1] -> U[1*4+1]=U[5]
    # B_TL: rows 0,1; cols 0,1
    #   e=B[0,0] -> V[0*4+0]=V[0]
    #   f=B[0,1] -> V[0*4+1]=V[1]
    #   g=B[1,0] -> V[1*4+0]=V[4]
    #   h=B[1,1] -> V[1*4+1]=V[5]
    # Result C_L[0,0], C_L[0,1], C_L[1,0], C_L[1,1]
    # C[i,k] where i=0,1 and k=0,1
    # W row index = k*n + i = k*2 + i
    #   R00 = C[0,0] -> W[0*2+0]=W[0]
    #   R01 = C[0,1] -> W[1*2+0]=W[2]
    #   R10 = C[1,0] -> W[0*2+1]=W[1]
    #   R11 = C[1,1] -> W[1*2+1]=W[3]
    
    add_strassen_factors(0, 0, 1, 4, 5, 0, 1, 4, 5, 0, 2, 1, 3)
    
    # A_R @ B_BL:
    # A_R: rows 0,1; cols 2,3
    #   a=A[0,2] -> U[0*4+2]=U[2]
    #   b=A[0,3] -> U[0*4+3]=U[3]
    #   c=A[1,2] -> U[1*4+2]=U[6]
    #   d=A[1,3] -> U[1*4+3]=U[7]
    # B_BL: rows 2,3; cols 0,1
    #   e=B[2,0] -> V[2*4+0]=V[8]
    #   f=B[2,1] -> V[2*4+1]=V[9]
    #   g=B[3,0] -> V[3*4+0]=V[12]
    #   h=B[3,1] -> V[3*4+1]=V[13]
    # Result contributes to C[:, 0:2]
    #   R00 = C[0,0] -> W[0]
    #   R01 = C[0,1] -> W[2]
    #   R10 = C[1,0] -> W[1]
    #   R11 = C[1,1] -> W[3]
    
    add_strassen_factors(7, 2, 3, 6, 7, 8, 9, 12, 13, 0, 2, 1, 3)
    
    # A_L @ B_TR:
    # A_L: rows 0,1; cols 0,1
    #   a=A[0,0] -> U[0]
    #   b=A[0,1] -> U[1]
    #   c=A[1,0] -> U[4]
    #   d=A[1,1] -> U[5]
    # B_TR: rows 0,1; cols 2,3
    #   e=B[0,2] -> V[0*4+2]=V[2]
    #   f=B[0,3] -> V[0*4+3]=V[3]
    #   g=B[1,2] -> V[1*4+2]=V[6]
    #   h=B[1,3] -> V[1*4+3]=V[7]
    # Result contributes to C[:, 2:4]
    #   R00 = C[0,2] -> W[2*2+0]=W[4]
    #   R01 = C[0,3] -> W[3*2+0]=W[6]
    #   R10 = C[1,2] -> W[2*2+1]=W[5]
    #   R11 = C[1,3] -> W[3*2+1]=W[7]
    
    add_strassen_factors(14, 0, 1, 4, 5, 2, 3, 6, 7, 4, 6, 5, 7)
    
    # A_R @ B_BR:
    # A_R: rows 0,1; cols 2,3
    #   a=A[0,2] -> U[2]
    #   b=A[0,3] -> U[3]
    #   c=A[1,2] -> U[6]
    #   d=A[1,3] -> U[7]
    # B_BR: rows 2,3; cols 2,3
    #   e=B[2,2] -> V[2*4+2]=V[10]
    #   f=B[2,3] -> V[2*4+3]=V[11]
    #   g=B[3,2] -> V[3*4+2]=V[14]
    #   h=B[3,3] -> V[3*4+3]=V[15]
    # Result contributes to C[:, 2:4]
    #   R00 = C[0,2] -> W[4]
    #   R01 = C[0,3] -> W[6]
    #   R10 = C[1,2] -> W[5]
    #   R11 = C[1,3] -> W[7]
    
    add_strassen_factors(21, 2, 3, 6, 7, 10, 11, 14, 15, 4, 6, 5, 7)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
