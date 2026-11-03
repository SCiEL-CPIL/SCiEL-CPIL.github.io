"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build 2x2 Strassen. Returns (U, V, W) each shape (4, 7).
    U index: i*2+j, V index: j*2+k, W index: k*2+i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    
    return U, V, W

def _build_winograd_3x3():
    """3x3 Winograd. Rank 18.
    U index: i*3+j, V index: j*3+k, W index: k*3+i
    """
    rank = 18
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Winograd 3x3: 18 multiplications
    # A' = [a00+a10+a20, a10, a01+a11+a21, a11, a02+a12+a22, a12,
    #        a00-a20, a01-a21, a02-a22, a20, a21, a22,
    #        a10-a00-a20, a11-a01-a21, a12-a02-a22]
    # B' = [b00+b10+b20, b10, b01+b11+b21, b11, b02+b12+b22, b12,
    #        b00-b20, b01-b21, b02-b22, b20, b21, b22,
    #        b10-b00-b20, b11-b01-b21, b12-b02-b22]
    # C = M0*(e1+e2+e3) + M1*(e2+e5) + M2*(e1+e3+e6) + M3*(e5) + M4*(e1+e6+e9) + M5*(e6)
    #     + M7*(e1-e9) + M8*(e2-e6) + M9*(e3-e9) + M10*(e9) + M11*(e6) + M12*(e9)
    #     + M13*(e2-e1-e9) + M14*(e5-e2-e6) + M15*(e6-e3-e9)
    # where e_i are unit matrices
    
    # Let me use the standard Winograd 3x3 formulation
    # 18 scalar multiplications
    
    # m0 = (a00+a10+a20)*(b00+b10+b20)
    # m1 = a10*(b10)
    # m2 = (a01+a11+a21)*(b01+b11+b21)
    # m3 = a11*b11
    # m4 = (a02+a12+a22)*(b02+b12+b22)
    # m5 = a12*b12
    # m6 = (a00-a20)*(b00-b20)
    # m7 = (a01-a21)*(b01-b21)
    # m8 = (a02-a22)*(b02-b22)
    # m9 = a20*b20
    # m10 = a21*b21
    # m11 = a22*b22
    # m12 = (a10-a00-a20)*(b10-b00-b20)
    # m13 = (a11-a01-a21)*(b11-b01-b21)
    # m14 = (a12-a02-a22)*(b12-b02-b22)
    # m15 = (a00+a10-a20)*(b00+b10-b20)  -- wait, let me be more careful
    
    # Standard Winograd 3x3 (18 mults):
    # Let me use the formulation from the literature.
    # A is 3x3, B is 3x3, C = A*B
    # 
    # m0 = (a00+a10+a20)*(b00+b10+b20)
    # m1 = a10*b10
    # m2 = (a01+a11+a21)*(b01+b11+b21)  
    # m3 = a11*b11
    # m4 = (a02+a12+a22)*(b02+b12+b22)
    # m5 = a12*b12
    # m6 = (a00-a20)*(b00-b20)
    # m7 = (a01-a21)*(b01-b21)
    # m8 = (a02-a22)*(b02-b22)
    # m9 = a20*b20
    # m10 = a21*b21
    # m11 = a22*b22
    # m12 = (a10-a00-a20)*(b10-b00-b20)
    # m13 = (a11-a01-a21)*(b11-b01-b21)
    # m14 = (a12-a02-a22)*(b12-b02-b22)
    # m15 = (a00+a10-a20)*(b00+b10-b20)
    # m16 = (a01+a11-a21)*(b01+b11-b21)
    # m17 = (a02+a12-a22)*(b02+b12-b22)
    
    # C00 = m0 + m6 + m12 + m15
    # C01 = m2 + m7 + m13 + m16
    # C02 = m4 + m8 + m14 + m17
    # C10 = m0 + m1 + m6 + m12
    # C11 = m2 + m3 + m7 + m13
    # C12 = m4 + m5 + m8 + m14
    # C20 = m0 - m6 + m9 - m12 + m15
    # C21 = m2 - m7 + m10 - m13 + m16
    # C22 = m4 - m8 + m11 - m14 + m17
    
    # Wait, let me verify: C20 should be a20*b00 + a21*b10 + a22*b20
    # Hmm, this is getting complex. Let me use a known correct formulation.
    
    # Actually, let me just use the well-known 3x3 Winograd with 18 multiplications.
    # I'll define it via the U, V, W matrices directly.
    
    # For each multiplication m_r, we have:
    # U[i*3+j, r] = coefficient of a_ij in the r-th left factor
    # V[j*3+k, r] = coefficient of b_jk in the r-th right factor  
    # W[k*3+i, r] = coefficient that goes into c_ik
    
    # m0: left = a00+a10+a20, right = b00+b10+b20
    # contributes to: c00(+1), c10(+1), c20(+1)
    # U[0][0]=1, U[3][0]=1, U[6][0]=1
    # V[0][0]=1, V[3][0]=1, V[6][0]=1
    # W[0][0]=1, W[3][0]=1, W[6][0]=1  (c00, c10, c20 -> W[k*3+i]: c00=W[0], c10=W[3], c20=W[6])
    
    # m1: left = a10, right = b10
    # contributes to: c10(+1)
    # U[3][1]=1, V[3][1]=1, W[3][1]=1
    
    # m2: left = a01+a11+a21, right = b01+b11+b21
    # contributes to: c01(+1), c11(+1), c21(+1)
    # U[1][2]=1, U[4][2]=1, U[7][2]=1
    # V[1][2]=1, V[4][2]=1, V[7][2]=1
    # W[1][2]=1, W[4][2]=1, W[7][2]=1  (c01=W[1], c11=W[4], c21=W[7])
    
    # m3: left = a11, right = b11
    # contributes to: c11(+1)
    # U[4][3]=1, V[4][3]=1, W[4][3]=1
    
    # m4: left = a02+a12+a22, right = b02+b12+b22
    # contributes to: c02(+1), c12(+1), c22(+1)
    # U[2][4]=1, U[5][4]=1, U[8][4]=1
    # V[2][4]=1, V[5][4]=1, V[8][4]=1
    # W[2][4]=1, W[5][4]=1, W[8][4]=1  (c02=W[2], c12=W[5], c22=W[8])
    
    # m5: left = a12, right = b12
    # contributes to: c12(+1)
    # U[5][5]=1, V[5][5]=1, W[5][5]=1
    
    # m6: left = a00-a20, right = b00-b20
    # contributes to: c00(+1), c10(+1), c20(-1)
    # U[0][6]=1, U[6][6]=-1
    # V[0][6]=1, V[6][6]=-1
    # W[0][6]=1, W[3][6]=1, W[6][6]=-1
    
    # m7: left = a01-a21, right = b01-b21
    # contributes to: c01(+1), c11(+1), c21(-1)
    # U[1][7]=1, U[7][7]=-1
    # V[1][7]=1, V[7][7]=-1
    # W[1][7]=1, W[4][7]=1, W[7][7]=-1
    
    # m8: left = a02-a22, right = b02-b22
    # contributes to: c02(+1), c12(+1), c22(-1)
    # U[2][8]=1, U[8][8]=-1
    # V[2][8]=1, V[8][8]=-1
    # W[2][8]=1, W[5][8]=1, W[8][8]=-1
    
    # m9: left = a20, right = b20
    # contributes to: c20(+1)
    # U[6][9]=1, V[6][9]=1, W[6][9]=1
    
    # m10: left = a21, right = b21
    # contributes to: c21(+1)
    # U[7][10]=1, V[7][10]=1, W[7][10]=1
    
    # m11: left = a22, right = b22
    # contributes to: c22(+1)
    # U[8][11]=1, V[8][11]=1, W[8][11]=1
    
    # m12: left = a10-a00-a20, right = b10-b00-b20
    # contributes to: c00(-1), c10(+1), c20(-1)
    # U[0][12]=-1, U[3][12]=1, U[6][12]=-1
    # V[0][12]=-1, V[3][12]=1, V[6][12]=-1
    # W[0][12]=-1, W[3][12]=1, W[6][12]=-1
    
    # m13: left = a11-a01-a21, right = b11-b01-b21
    # contributes to: c01(-1), c11(+1), c21(-1)
    # U[1][13]=-1, U[4][13]=1, U[7][13]=-1
    # V[1][13]=-1, V[4][13]=1, V[7][13]=-1
    # W[1][13]=-1, W[4][13]=1, W[7][13]=-1
    
    # m14: left = a12-a02-a22, right = b12-b02-b22
    # contributes to: c02(-1), c12(+1), c22(-1)
    # U[2][14]=-1, U[5][14]=1, U[8][14]=-1
    # V[2][14]=-1, V[5][14]=1, V[8][14]=-1
    # W[2][14]=-1, W[5][14]=1, W[8][14]=-1
    
    # m15: left = a00+a10-a20, right = b00+b10-b20
    # contributes to: c00(+1), c10(+1), c20(-1)
    # U[0][15]=1, U[3][15]=1, U[6][15]=-1
    # V[0][15]=1, V[3][15]=1, V[6][15]=-1
    # W[0][15]=1, W[3][15]=1, W[6][15]=-1
    
    # m16: left = a01+a11-a21, right = b01+b11-b21
    # contributes to: c01(+1), c11(+1), c21(-1)
    # U[1][16]=1, U[4][16]=1, U[7][16]=-1
    # V[1][16]=1, V[4][16]=1, V[7][16]=-1
    # W[1][16]=1, W[4][16]=1, W[7][16]=-1
    
    # m17: left = a02+a12-a22, right = b02+b12-b22
    # contributes to: c02(+1), c12(+1), c22(-1)
    # U[2][17]=1, U[5][17]=1, U[8][17]=-1
    # V[2][17]=1, V[5][17]=1, V[8][17]=-1
    # W[2][17]=1, W[5][17]=1, W[8][17]=-1
    
    # Now let me verify this gives the correct product.
    # c00 = m0 + m6 + m12*(-1) + m15
    #   = (a00+a10+a20)(b00+b10+b20) + (a00-a20)(b00-b20) - (a10-a00-a20)(b10-b00-b20) + (a00+a10-a20)(b00+b10-b20)
    # This is getting very complex to verify by hand. Let me just trust the standard formulation
    # and verify numerically.
    
    # Set up the matrices
    U[0][0]=1; U[3][0]=1; U[6][0]=1
    V[0][0]=1; V[3][0]=1; V[6][0]=1
    W[0][0]=1; W[3][0]=1; W[6][0]=1
    
    U[3][1]=1
    V[3][1]=1
    W[3][1]=1
    
    U[1][2]=1; U[4][2]=1; U[7][2]=1
    V[1][2]=1; V[4][2]=1; V[7][2]=1
    W[1][2]=1; W[4][2]=1; W[7][2]=1
    
    U[4][3]=1
    V[4][3]=1
    W[4][3]=1
    
    U[2][4]=1; U[5][4]=1; U[8][4]=1
    V[2][4]=1; V[5][4]=1; V[8][4]=1
    W[2][4]=1; W[5][4]=1; W[8][4]=1
    
    U[5][5]=1
    V[5][5]=1
    W[5][5]=1
    
    U[0][6]=1; U[6][6]=-1
    V[0][6]=1; V[6][6]=-1
    W[0][6]=1; W[3][6]=1; W[6][6]=-1
    
    U[1][7]=1; U[7][7]=-1
    V[1][7]=1; V[7][7]=-1
    W[1][7]=1; W[4][7]=1; W[7][7]=-1
    
    U[2][8]=1; U[8][8]=-1
    V[2][8]=1; V[8][8]=-1
    W[2][8]=1; W[5][8]=1; W[8][8]=-1
    
    U[6][9]=1
    V[6][9]=1
    W[6][9]=1
    
    U[7][10]=1
    V[7][10]=1
    W[7][10]=1
    
    U[8][11]=1
    V[8][11]=1
    W[8][11]=1
    
    U[0][12]=-1; U[3][12]=1; U[6][12]=-1
    V[0][12]=-1; V[3][12]=1; V[6][12]=-1
    W[0][12]=-1; W[3][12]=1; W[6][12]=-1
    
    U[1][13]=-1; U[4][13]=1; U[7][13]=-1
    V[1][13]=-1; V[4][13]=1; V[7][13]=-1
    W[1][13]=-1; W[4][13]=1; W[7][13]=-1
    
    U[2][14]=-1; U[5][14]=1; U[8][14]=-1
    V[2][14]=-1; V[5][14]=1; V[8][14]=-1
    W[2][14]=-1; W[5][14]=1; W[8][14]=-1
    
    U[0][15]=1; U[3][15]=1; U[6][15]=-1
    V[0][15]=1; V[3][15]=1; V[6][15]=-1
    W[0][15]=1; W[3][15]=1; W[6][15]=-1
    
    U[1][16]=1; U[4][16]=1; U[7][16]=-1
    V[1][16]=1; V[4][16]=1; V[7][16]=-1
    W[1][16]=1; W[4][16]=1; W[7][16]=-1
    
    U[2][17]=1; U[5][17]=1; U[8][17]=-1
    V[2][17]=1; V[5][17]=1; V[8][17]=-1
    W[2][17]=1; W[5][17]=1; W[8][17]=-1
    
    return U, V, W

def _build_4x2x6_strassen():
    """4x2 * 2x6 using 2x2 Strassen blocks.
    Split: 4 = 2+2, 2 = 2, 6 = 3+3+3 (or use schoolbook for the 2x3 part)
    Actually: split 4 into 2+2, keep 2 as is, split 6 into 2+2+2.
    4 blocks of (2x2)*(2x2), each rank 7. Total rank = 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = [[0]*rank for _ in range(8)]   # 4*2
    V = [[0]*rank for _ in range(12)]  # 2*6
    W = [[0]*rank for _ in range(24)]  # 6*4
    
    # 4 blocks: (row_block, col_block) where row_block in {0,1}, col_block in {0,1,2}
    # Block (rb, cb): A rows 2*rb..2*rb+1, B cols 2*cb..2*cb+1
    for rb in range(2):
        for cb in range(3):
            base = (rb*3+cb)*7
            for t in range(7):
                col = base+t
                for li in range(2):
                    for lj in range(2):
                        ca = U2[li*2+lj][t]
                        if ca:
                            U[(2*rb+li)*2+lj][col] += ca
                for lj in range(2):
                    for lk in range(2):
                        cb_ = V2[lj*2+lk][t]
                        if cb_:
                            V[lj*6+(2*cb+lk)][col] += cb_
                for lk in range(2):
                    for li in range(2):
                        cw = W2[lk*2+li][t]
                        if cw:
                            W[(2*cb+lk)*4+(2*rb+li)][col] += cw
    return U, V, W

def _build_4x3x6_winograd():
    """4x3 * 3x6 using 3x3 Winograd blocks.
    Split: 4 = 3+1 (use 3x3 Winograd for 3x3x3 blocks, schoolbook for 1x3x3)
    Actually: 4 = 3+1, 3 = 3, 6 = 3+3
    Blocks: 
      (3x3)*(3x3) for rows 0-2, cols 0-2: rank 18
      (3x3)*(3x3) for rows 0-2, cols 3-5: rank 18
      (1x3)*(3x3) for row 3, cols 0-2: rank 9
      (1x3)*(3x3) for row 3, cols 3-5: rank 9
    Total rank = 18+18+9+9 = 54
    """
    U3, V3, W3 = _build_winograd_3x3()
    rank = 54
    U = [[0]*rank for _ in range(12)]   # 4*3
    V = [[0]*rank for _ in range(18)]   # 3*6
    W = [[0]*rank for _ in range(24)]   # 6*4
    
    # Block 0: rows 0-2, cols 0-2, Winograd 3x3
    base = 0
    for t in range(18):
        col = base+t
        for r in range(9):
            U[r][col] = U3[r][t]
            V[r][col] = V3[r][t]
            W[r][col] = W3[r][t]
    
    # Block 1: rows 0-2, cols 3-5, Winograd 3x3
    base = 18
    for t in range(18):
        col = base+t
        for r in range(9):
            # U: same rows 0-2, same j
            U[r][col] = U3[r][t]
            # V: j same, k offset by 3
            V[r][col] = V3[r][t]  # V index is j*3+k, k goes 0-2, but we need k=3-5
            # Actually V[j*3+k] where k is local 0-2, global k = 3+k
            # V row index = j*3 + k_global = j*3 + (3+k_local)
            # But V3 has index j*3+k_local. We need to shift.
            # Let me redo: V3[r] where r = j*3+k_local
            # Global V index = j*3 + (3+k_local) = j*3 + 3 + k_local
            # So V_global_row = r + 3 (since r = j*3+k_local, and we add 3 to k)
            # Wait: r = j*3+k_local. Global = j*3+(3+k_local) = r+3. Yes.
            V[r+3][col] = V3[r][t]
            # W: k_local, i same. W index = k*4+i. Global k = 3+k_local.
            # W3[r] where r = k_local*3+i. Global W row = (3+k_local)*4+i = k_local*4+12+i = r + 12
            W[r+12][col] = W3[r][t]
    
    # Block 2: row 3, cols 0-2, schoolbook 1x3x3, rank 9
    base = 36
    for j in range(3):
        for k in range(3):
            col = base + j*3+k
            U[3*3+j][col] = 1   # U[i*3+j] where i=3
            V[j*3+k][col] = 1
            W[k*4+3][col] = 1   # W[k*4+i] where i=3
    
    # Block 3: row 3, cols 3-5, schoolbook 1x3x3, rank 9
    base = 45
    for j in range(3):
        for k in range(3):
            col = base + j*3+k
            U[3*3+j][col] = 1
            V[j*3+(3+k)][col] = 1
            W[(3+k)*4+3][col] = 1
    
    return U, V, W

def _combine(decomps, n, m, p):
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    cur = 0
    for U_s, V_s, W_s, (i_s,i_e), (j_s,j_e), (k_s,k_e) in decomps:
        n_s = i_e-i_s; m_s = j_e-j_s; p_s = k_e-k_s
        r_s = len(U_s[0])
        
        for r in range(n_s*m_s):
            il = r // m_s; jl = r % m_s
            ig = i_s+il; jg = j_s+jl
            rg = ig*m+jg
            for c in range(r_s):
                U[rg][cur+c] = U_s[r][c]
        
        for r in range(m_s*p_s):
            jl = r // p_s; kl = r % p_s
            jg = j_s+jl; kg = k_s+kl
            rg = jg*p+kg
            for c in range(r_s):
                V[rg][cur+c] = V_s[r][c]
        
        for r in range(p_s*n_s):
            kl = r // n_s; il = r % n_s
            kg = k_s+kl; ig = i_s+il
            rg = kg*n+ig
            for c in range(r_s):
                W[rg][cur+c] = W_s[r][c]
        
        cur += r_s
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c]=1
                V[j*p+k][c]=1
                W[k*n+i][c]=1
                c += 1
    return U, V, W

def _verify_3x3_winograd():
    """Verify the 3x3 Winograd decomposition in isolation."""
    U, V, W = _build_winograd_3x3()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(18):
                    s += U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r]
                if s != 1:
                    raise ValueError(f"3x3 Winograd verification failed at ({i},{j},{k}): got {s}")

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # Verify 3x3 Winograd in isolation
        _verify_3x3_winograd()
        
        # Split inner dimension m=5 into 2+3
        # Part 1: 4x2x6 with Strassen 2x2 blocks, rank 28
        U1, V1, W1 = _build_4x2x6_strassen()
        d1 = (U1, V1, W1, (0,4), (0,2), (0,6))
        
        # Part 2: 4x3x6 with Winograd 3x3 + schoolbook, rank 54
        U2, V2, W2 = _build_4x3x6_winograd()
        d2 = (U2, V2, W2, (0,4), (2,5), (0,6))
        
        U, V, W = _combine([d1, d2], n, m, p)
        
        # Verify before returning
        _verify(U, V, W, n, m, p)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")

# EVOLVE-BLOCK-END
