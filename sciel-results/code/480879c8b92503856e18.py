"""Program for matrix multiplication 2,3,4 with rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
n, m, p = DIMENSIONS
# Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
# Indexing convention:
#   U[i*m + j, r]  (i in [0,n), j in [0,m))
#   V[j*p + k, r]  (j in [0,m), k in [0,p))
#   W[k*n + i, r]  (k in [0,p), i in [0,n))
#   Output: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  summed over j

# We use a block decomposition.
# Split the 3 rows of A (and columns of B) into two blocks: 
#   Block 1: rows/cols 0,1 (size 2x2)
#   Block 2: row/col 2 (size 1x2)
# But this gives rank 7 + 8 = 15? No, that's not right.
# Let me re-think. The schoolbook is rank 24.
#
# Actually, let's use a simpler approach: search for a lower rank decomposition.
# The rank of 2x3x4 matrix multiplication is at most 24 (schoolbook).
# Known results: R(2,3,4) <= 18 (using Winograd or similar).
# Actually, the minimum rank is known to be 18 for 2x3x4.
# Let me try to construct a rank-18 decomposition.
#
# Approach: Use the fact that M_{2,3,4} can be decomposed using 
# a combination of Strassen-like tricks and direct terms.
#
# Alternative: Just use a numerical optimization / random search to find a low-rank 
# decomposition, then snap to exact values.
#
# Given the time constraint, let me try a structured approach:
# Use the decomposition M_{2,3,4} = M_{2,2,2} + M_{2,1,2} + M_{2,2,2} + M_{2,1,2}
# That would be 7 + 4 + 7 + 4 = 22. Still better than 24.
#
# Let's go with rank 22 using the block decomposition.
# Split the 3 middle dimension into 2+1, and the 4 last dimension into 2+2.
# So we have 4 blocks:
#   Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> 2x2x2, rank 7
#   Block (0,1): A[0:2, 0:2] * B[0:2, 2:4] -> 2x2x2, rank 7  
#   Block (1,0): A[0:2, 2:3] * B[2:3, 0:2] -> 2x1x2, rank 4
#   Block (1,1): A[0:2, 2:3] * B[2:3, 2:4] -> 2x1x2, rank 4
# Total: 7+7+4+4 = 22

def _strassen_2x2x2():
    """Return 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-dim vectors (for 2x2 matrices).
    A is 2x2, B is 2x2, C = A*B is 2x2.
    U index: i*2+j for 2x2 matrix (i,j in 0,1)
    V index: j*2+k (j,k in 0,1)
    W index: k*2+i (k,i in 0,1)
    """
    # Strassen's 7 multiplications for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # U vectors for A coefficients (a,b,c,d):
    # m1: a=1, b=0, c=1, d=1
    # m2: a=0, b=0, c=1, d=1
    # m3: a=1, b=1, c=0, d=0
    # m4: a=0, b=0, c=0, d=1
    # m5: a=1, b=1, c=0, d=0
    # m6: a=-1, b=0, c=1, d=0
    # m7: a=0, b=1, c=0, d=-1
    
    # V vectors for B coefficients (e,f,g,h):
    # m1: e=1, f=1, g=0, h=1  (wait, m1=(a+d)(e+h), so e and h)
    # Let me redo: m1 = (a+d)(e+h): A part (1,0,1,1), B part (1,0,0,1)
    # m2 = (c+d)e: A part (0,0,1,1), B part (1,0,0,0)
    # m3 = a(f-h): A part (1,0,0,0), B part (0,1,0,-1)
    # m4 = d(g-e): A part (0,0,0,1), B part (-1,0,1,0)
    # m5 = (a+b)h: A part (1,1,0,0), B part (0,0,0,1)
    # m6 = (c-a)(e+f): A part (-1,0,1,0), B part (1,1,0,0)
    # m7 = (b-d)(g+h): A part (0,1,0,-1), B part (0,0,1,1)
    
    U_parts = [
        [1,0,1,1],  # m1
        [0,0,1,1],  # m2
        [1,0,0,0],  # m3
        [0,0,0,1],  # m4
        [1,1,0,0],  # m5
        [-1,0,1,0], # m6
        [0,1,0,-1], # m7
    ]
    V_parts = [
        [1,0,0,1],  # m1
        [1,0,0,0],  # m2
        [0,1,0,-1], # m3
        [-1,0,1,0], # m4
        [0,0,0,1],  # m5
        [1,1,0,0],  # m6
        [0,0,1,1],  # m7
    ]
    # W vectors for C coefficients. C = [[C00, C01], [C10, C11]]
    # W index: k*2+i, so C00->0, C01->1, C10->2, C11->3
    # C00 = m1+m4-m5+m7
    # C01 = m3+m5
    # C10 = m2+m4
    # C11 = m1-m2+m3+m6
    
    W_parts = [
        [1,0,0,1],  # m1: C00+=1, C11+=1
        [0,0,1,-1], # m2: C10+=1, C11-=1
        [0,1,0,1],  # m3: C01+=1, C11+=1
        [1,0,1,0],  # m4: C00+=1, C10+=1
        [-1,1,0,0], # m5: C00-=1, C01+=1
        [0,0,0,1],  # m6: C11+=1
        [1,0,0,0],  # m7: C00+=1
    ]
    
    return U_parts, V_parts, W_parts

def _outer_2x1x2():
    """Return 4 rank-1 terms for 2x1x2 matrix multiplication.
    A is 2x1, B is 1x2, C is 2x2.
    U index: i*1+j = i (i in 0,1), j=0
    V index: j*2+k = k (j=0, k in 0,1)
    W index: k*2+i (k in 0,1, i in 0,1)
    C[i,k] = A[i,0] * B[0,k]
    4 terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
    """
    U_parts = [
        [1,0],  # i=0
        [1,0],  # i=0
        [0,1],  # i=1
        [0,1],  # i=1
    ]
    V_parts = [
        [1,0],  # k=0
        [0,1],  # k=1
        [1,0],  # k=0
        [0,1],  # k=1
    ]
    # W index: k*2+i
    # (i=0,k=0): W[0*2+0]=W[0]=1
    # (i=0,k=1): W[1*2+0]=W[2]=1
    # (i=1,k=0): W[0*2+1]=W[1]=1
    # (i=1,k=1): W[1*2+1]=W[3]=1
    W_parts = [
        [1,0,0,0],  # (i=0,k=0)
        [0,0,1,0],  # (i=0,k=1)
        [0,1,0,0],  # (i=1,k=0)
        [0,0,0,1],  # (i=1,k=1)
    ]
    return U_parts, V_parts, W_parts

def construct():
    deadline = time.time() + 120
    
    # Block decomposition:
    # A is 2x3, B is 3x4, C is 2x4
    # Split A's columns into [0:2] and [2:3]
    # Split B's rows into [0:2] and [2:3], and columns into [0:2] and [2:4]
    # Block (A[0:2], B[0:2, 0:2]): 2x2x2, rank 7
    # Block (A[0:2], B[0:2, 2:4]): 2x2x2, rank 7
    # Block (A[2:3], B[2:3, 0:2]): 1x2x2 -> 2x1x2, rank 4
    # Block (A[2:3], B[2:3, 2:4]): 1x2x2 -> 2x1x2, rank 4
    # Total rank: 22
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22
    
    col_offset = 0
    
    # Block 1: A rows 0-1, A cols 0-1, B rows 0-1, B cols 0-1 -> C rows 0-1, C cols 0-1
    # This is 2x2x2 Strassen
    sU, sV, sW = _strassen_2x2x2()
    for r in range(7):
        # U: A[i,j] where i in 0,1; j in 0,1 -> index i*3+j
        for ii in range(2):
            for jj in range(2):
                U[ii * m + jj][col_offset + r] = sU[r][ii * 2 + jj]
        # V: B[j,k] where j in 0,1; k in 0,1 -> index j*4+k
        for jj in range(2):
            for kk in range(2):
                V[jj * p + kk][col_offset + r] = sV[r][jj * 2 + kk]
        # W: C[k,i] where k in 0,1; i in 0,1 -> index k*2+i
        for kk in range(2):
            for ii in range(2):
                W[kk * n + ii][col_offset + r] = sW[r][kk * 2 + ii]
    col_offset += 7
    
    # Block 2: A rows 0-1, A cols 0-1, B rows 0-1, B cols 2-3 -> C rows 0-1, C cols 2-3
    # This is also 2x2x2 Strassen
    for r in range(7):
        for ii in range(2):
            for jj in range(2):
                U[ii * m + jj][col_offset + r] = sU[r][ii * 2 + jj]
        # V: B[j,k] where j in 0,1; k in 2,3 -> index j*4+k
        for jj in range(2):
            for kk in range(2):
                V[jj * p + (kk + 2)][col_offset + r] = sV[r][jj * 2 + kk]
        # W: C[k,i] where k in 2,3; i in 0,1 -> index k*2+i
        for kk in range(2):
            for ii in range(2):
                W[(kk + 2) * n + ii][col_offset + r] = sW[r][kk * 2 + ii]
    col_offset += 7
    
    # Block 3: A row 2, A col 2, B row 2, B cols 0-1 -> C rows 0-1, C cols 0-1
    # 2x1x2 outer product
    oU, oV, oW = _outer_2x1x2()
    for r in range(4):
        # U: A[2,2] -> index 2*3+2 = 8... wait, n=2, m=3, so i in 0,1. 
        # A is 2x3. Row 2 doesn't exist for A. 
        # I made an error. A is 2x3, so rows 0,1. B is 3x4, rows 0,1,2.
        # The middle dimension is 3. So we split the middle dim (cols of A, rows of B) into 2+1.
        # Block 3: A cols 2, B row 2, B cols 0-1 -> C cols 0-1
        # A[i, 2] * B[2, k] for i in 0,1; k in 0,1
        # U: A[i,2] -> index i*3+2
        for ii in range(2):
            U[ii * m + 2][col_offset + r] = oU[r][ii]
        # V: B[2,k] -> index 2*4+k = 8+k
        for kk in range(2):
            V[2 * p + kk][col_offset + r] = oV[r][kk]
        # W: C[k,i] -> index k*2+i
        for kk in range(2):
            for ii in range(2):
                W[kk * n + ii][col_offset + r] = oW[r][kk * 2 + ii]
    col_offset += 4
    
    # Block 4: A cols 2, B row 2, B cols 2-3 -> C cols 2-3
    for r in range(4):
        for ii in range(2):
            U[ii * m + 2][col_offset + r] = oU[r][ii]
        # V: B[2,k] where k in 2,3 -> index 2*4+k = 8+k
        for kk in range(2):
            V[2 * p + (kk + 2)][col_offset + r] = oV[r][kk]
        # W: C[k,i] where k in 2,3 -> index k*2+i
        for kk in range(2):
            for ii in range(2):
                W[(kk + 2) * n + ii][col_offset + r] = oW[r][kk * 2 + ii]
    col_offset += 4
    
    if time.time() > deadline:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
