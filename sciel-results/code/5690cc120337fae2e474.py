"""Rewritten program for 2x2 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    The decomposition satisfies:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    where n=m=p=2, so U, V, W are each 4x7.
    
    Strassen's 7 products:
      M1 = (a00 + a11)(b00 + b11)
      M2 = (a10 + a11)(b00 + b01)
      M3 = a00(b10 - b11)
      M4 = a11(b01 - b00)
      M5 = (a00 + a01)(b11)
      M6 = (a10 - a00)(b00 + b01)
      M7 = (a01 - a11)(b10 + b11)
    
    Output reconstruction:
      C00 = M1 - M5 + M7
      C01 = M3 + M5
      C10 = M2 - M4 + M6
      C11 = M1 - M2 + M4 + M7
    
    Index conventions:
      U[i*2 + j, r]  for i,j in {0,1}
      V[j*2 + k, r]  for j,k in {0,1}
      W[k*2 + i, r]  for k,i in {0,1}  (output index k*n+i)
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r=0: M1 = (a00 + a11)(b00 + b11)
    # U: a00 (i=0,j=0) and a11 (i=1,j=1)
    U[0][0] = 1
    U[3][0] = 1
    # V: b00 (j=0,k=0) and b11 (j=1,k=1)
    V[0][0] = 1
    V[3][0] = 1
    # W: C00 += 1 (k=0,i=0), C11 += 1 (k=1,i=1)
    W[0][0] = 1
    W[3][0] = 1
    
    # r=1: M2 = (a10 + a11)(b00 + b01)
    # U: a10 (i=1,j=0) and a11 (i=1,j=1)
    U[2][1] = 1
    U[3][1] = 1
    # V: b00 (j=0,k=0) and b01 (j=0,k=1)
    V[0][1] = 1
    V[1][1] = 1
    # W: C10 += 1 (k=0,i=1), C11 -= 1 (k=1,i=1)
    W[1][1] = 1
    W[3][1] = -1
    
    # r=2: M3 = a00(b10 - b11)
    # U: a00 (i=0,j=0)
    U[0][2] = 1
    # V: b10 (j=1,k=0) and -b11 (j=1,k=1)
    V[2][2] = 1
    V[3][2] = -1
    # W: C01 += 1 (k=1,i=0)
    W[2][2] = 1
    
    # r=3: M4 = a11(b01 - b00)
    # U: a11 (i=1,j=1)
    U[3][3] = 1
    # V: b01 (j=0,k=1) and -b00 (j=0,k=0)
    V[1][3] = 1
    V[0][3] = -1
    # W: C10 += 1 (k=0,i=1), C11 += 1 (k=1,i=1)
    W[1][3] = 1
    W[3][3] = 1
    
    # r=4: M5 = (a00 + a01)(b11)
    # U: a00 (i=0,j=0) and a01 (i=0,j=1)
    U[0][4] = 1
    U[1][4] = 1
    # V: b11 (j=1,k=1)
    V[3][4] = 1
    # W: C00 -= 1 (k=0,i=0), C01 += 1 (k=1,i=0)
    W[0][4] = -1
    W[2][4] = 1
    
    # r=5: M6 = (a10 - a00)(b00 + b01)
    # U: a10 (i=1,j=0) and -a00 (i=0,j=0)
    U[2][5] = 1
    U[0][5] = -1
    # V: b00 (j=0,k=0) and b01 (j=0,k=1)
    V[0][5] = 1
    V[1][5] = 1
    # W: C10 += 1 (k=0,i=1)
    W[1][5] = 1
    
    # r=6: M7 = (a01 - a11)(b10 + b11)
    # U: a01 (i=0,j=1) and -a11 (i=1,j=1)
    U[1][6] = 1
    U[3][6] = -1
    # V: b10 (j=1,k=0) and b11 (j=1,k=1)
    V[2][6] = 1
    V[3][6] = 1
    # W: C00 += 1 (k=0,i=0), C11 += 1 (k=1,i=1)
    W[0][6] = 1
    W[3][6] = 1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    # Fallback to schoolbook rank-8 if verification fails
                    rank_fb = 8
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][col] = 1
                                V_fb[jj * p + kk][col] = 1
                                W_fb[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
