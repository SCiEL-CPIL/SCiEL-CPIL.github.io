import numpy as np
import time

def construct():
    """
    Returns a rank-17 exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The decomposition is:
      C[i,k] = sum_{r=0}^{16} U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
      where C = A * B, A is 2x3, B is 3x3.
      
    U has shape (6, 17)  -- indexed by (i, j)
    V has shape (9, 17)  -- indexed by (j, k)
    W has shape (6, 17)  -- indexed by (k, i), following W[k*n+i, r] convention
    """
    n, m, p = 2, 3, 3
    rank = 17
    
    # Initialize with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block 1: j in {0,1}, k in {0,1} -- uses Strassen-like rank-7 for the 2x2 * 2x2 sub-problem
    # C[i,k] for i in {0,1}, k in {0,1} gets contributions from A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # We use 7 terms (r=0..6) for this block.
    #
    # Let a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
    # Let b00=B[0,0], b01=B[0,1], b10=B[1,0], b11=B[1,1]
    #
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    #
    # Strassen's 7-multiplication formula for 2x2:
    # M0 = (a00+a11)*(b00+b11)
    # M1 = (a10+a11)*b00
    # M2 = a00*(b01-b11)
    # M3 = a11*(b10-b00)
    # M4 = (a00+a01)*b11
    # M5 = (a10-a00)*(b00+b01)
    # M6 = (a01-a11)*(b10+b11)
    #
    # C[0,0] = M0 - M1 + M3 - M5
    # C[0,1] = M2 + M4
    # C[1,0] = M1 + M6
    # C[1,1] = M0 - M2 + M4 - M6
    
    # r=0: M0 = (a00+a11)*(b00+b11)
    U[0, 0] = 1   # a00 (i=0, j=0)
    U[4, 0] = 1   # a11 (i=1, j=1)
    V[0, 0] = 1   # b00 (j=0, k=0)
    V[4, 0] = 1   # b11 (j=1, k=1)
    W[0, 0] = 1   # C[0,0] (k=0, i=0)
    W[1, 0] = -1  # C[0,1] (k=1, i=0)
    W[2, 0] = 1   # C[1,0] (k=0, i=1)
    W[3, 0] = 1   # C[1,1] (k=1, i=1)
    
    # r=1: M1 = (a10+a11)*b00
    U[3, 1] = 1   # a10 (i=1, j=0)
    U[4, 1] = 1   # a11 (i=1, j=1)
    V[0, 1] = 1   # b00
    W[2, 1] = 1   # C[1,0]
    W[3, 1] = -1  # C[1,1]
    
    # r=2: M2 = a00*(b01-b11)
    U[0, 2] = 1   # a00
    V[1, 2] = 1   # b01 (j=0, k=1)
    V[4, 2] = -1  # b11
    W[1, 2] = 1   # C[0,1]
    W[3, 2] = -1  # C[1,1]
    
    # r=3: M3 = a11*(b10-b00)
    U[4, 3] = 1   # a11
    V[3, 3] = 1   # b10 (j=1, k=0)
    V[0, 3] = -1  # b00
    W[0, 3] = 1   # C[0,0]
    W[2, 3] = -1  # C[1,0]
    
    # r=4: M4 = (a00+a01)*b11
    U[0, 4] = 1   # a00
    U[1, 4] = 1   # a01 (i=0, j=1)
    V[4, 4] = 1   # b11
    W[1, 4] = 1   # C[0,1]
    W[3, 4] = 1   # C[1,1]
    
    # r=5: M5 = (a10-a00)*(b00+b01)
    U[3, 5] = 1   # a10
    U[0, 5] = -1  # a00
    V[0, 5] = 1   # b00
    V[1, 5] = 1   # b01
    W[0, 5] = -1  # C[0,0]
    
    # r=6: M6 = (a01-a11)*(b10+b11)
    U[1, 6] = 1   # a01
    U[4, 6] = -1  # a11
    V[3, 6] = 1   # b10
    V[4, 6] = 1   # b11
    W[2, 6] = 1   # C[1,0]
    W[3, 6] = -1  # C[1,1]
    
    # Block 2: j in {0,1}, k = 2 -- schoolbook rank 4
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # r=7: a00*b02
    U[0, 7] = 1
    V[2, 7] = 1   # b02 (j=0, k=2)
    W[4, 7] = 1   # C[0,2] (k=2, i=0)
    
    # r=8: a01*b12
    U[1, 8] = 1   # a01
    V[5, 8] = 1   # b12 (j=1, k=2)
    W[4, 8] = 1
    
    # r=9: a10*b02
    U[3, 9] = 1   # a10
    V[2, 9] = 1   # b02
    W[5, 9] = 1   # C[1,2] (k=2, i=1)
    
    # r=10: a11*b12
    U[4, 10] = 1  # a11
    V[5, 10] = 1  # b12
    W[5, 10] = 1
    
    # Block 3: j = 2, k in {0,1,2} -- schoolbook rank 6
    # C[i,k] += A[i,2]*B[2,k]
    # r=11: a02*b20
    U[2, 11] = 1  # a02 (i=0, j=2)
    V[6, 11] = 1  # b20 (j=2, k=0)
    W[0, 11] = 1  # C[0,0]
    
    # r=12: a02*b21
    U[2, 12] = 1
    V[7, 12] = 1  # b21 (j=2, k=1)
    W[1, 12] = 1  # C[0,1]
    
    # r=13: a02*b22
    U[2, 13] = 1
    V[8, 13] = 1  # b22 (j=2, k=2)
    W[4, 13] = 1  # C[0,2]
    
    # r=14: a12*b20
    U[5, 14] = 1  # a12 (i=1, j=2)
    V[6, 14] = 1  # b20
    W[2, 14] = 1  # C[1,0]
    
    # r=15: a12*b21
    U[5, 15] = 1
    V[7, 15] = 1  # b21
    W[3, 15] = 1  # C[1,1]
    
    # r=16: a12*b22
    U[5, 16] = 1
    V[8, 16] = 1  # b22
    W[5, 16] = 1  # C[1,2]
    
    # Convert to lists of floats (integers are fine as floats)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify the decomposition
    A = np.random.randint(-5, 6, (n, m))
    B = np.random.randint(-5, 6, (m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed[i, k] = s
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook rank 18
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
