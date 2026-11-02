import numpy as np
import time

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = a(e+f)
    U[0, 0] = 1
    V[0, 0] = 1
    V[1, 0] = 1
    
    # M2 = (a+b)h
    U[0, 1] = 1
    U[1, 1] = 1
    V[3, 1] = 1
    
    # M3 = (c+d)e
    U[2, 2] = 1
    U[3, 2] = 1
    V[0, 2] = 1
    
    # M4 = d(g+f)
    U[3, 3] = 1
    V[2, 3] = 1
    V[1, 3] = 1
    
    # M5 = (a-d)(e+h)
    U[0, 4] = 1
    U[3, 4] = -1
    V[0, 4] = 1
    V[3, 4] = 1
    
    # M6 = (b-c)(e+f)
    U[1, 5] = 1
    U[2, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    
    # M7 = (a+c)(h-g)
    U[0, 6] = 1
    U[2, 6] = 1
    V[3, 6] = 1
    V[2, 6] = -1
    
    # Output C[k*n+i] where n=2
    # C[0,0] = M5 + M4 - M6 - M7
    W[0, 4] = 1
    W[0, 3] = 1
    W[0, 5] = -1
    W[0, 6] = -1
    
    # C[0,1] = M1 + M2
    W[1, 0] = 1
    W[1, 1] = 1
    
    # C[1,0] = M3 + M7
    W[2, 2] = 1
    W[2, 6] = 1
    
    # C[1,1] = M5 + M1 - M2 - M4
    W[3, 4] = 1
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 3] = -1
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Returns U, V, W for 2x2x1 matrix multiplication with rank 4.
    U: (4, 4), V: (2, 4), W: (2, 4)
    """
    U = np.zeros((4, 4), dtype=float)
    V = np.zeros((2, 4), dtype=float)
    W = np.zeros((2, 4), dtype=float)
    
    # C[0] = a*e + b*f
    # C[1] = c*e + d*f
    
    # Term 0: a*e
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term 1: b*f
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Term 2: c*e
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Term 3: d*f
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _combine_2x2x3():
    """
    Returns U, V, W for 2x2x3 matrix multiplication with rank 11.
    U: (4, 11), V: (6, 11), W: (6, 11)
    """
    U222, V222, W222 = _strassen_2x2x2()
    U221, V221, W221 = _schoolbook_2x2x1()
    
    U = np.zeros((4, 11), dtype=float)
    V = np.zeros((6, 11), dtype=float)
    W = np.zeros((6, 11), dtype=float)
    
    # First 7 columns: 2x2x2 part (cols 0,1 of B)
    U[:, :7] = U222
    # V222 is 4x7. Rows correspond to j*2+k.
    # We need to map to j*3+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 2
    # j=1, k=1 -> 3
    # In V (6 rows), indices are j*3+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=0, k=2 -> 2 (not used in 2x2x2)
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # j=1, k=2 -> 5 (not used in 2x2x2)
    
    V[0, :7] = V222[0, :7] # j=0, k=0
    V[1, :7] = V222[1, :7] # j=0, k=1
    V[3, :7] = V222[2, :7] # j=1, k=0
    V[4, :7] = V222[3, :7] # j=1, k=1
    
    # W222 is 4x7. Rows correspond to k*2+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # In W (6 rows), indices are k*2+i? No, W dimension is p*n = 3*2 = 6.
    # Index k*2+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    
    W[0, :7] = W222[0, :7]
    W[1, :7] = W222[1, :7]
    W[2, :7] = W222[2, :7]
    W[3, :7] = W222[3, :7]
    
    # Last 4 columns: 2x2x1 part (col 2 of B)
    U[:, 7:] = U221
    # V221 is 2x4. Rows j.
    # j=0 -> 0
    # j=1 -> 1
    # In V (6 rows), index j*3+2.
    # j=0 -> 2
    # j=1 -> 5
    
    V[2, 7:] = V221[0, :]
    V[5, 7:] = V221[1, :]
    
    # W221 is 2x4. Rows i.
    # i=0 -> 0
    # i=1 -> 1
    # In W (6 rows), index k*2+i with k=2.
    # i=0 -> 4
    # i=1 -> 5
    
    W[4, 7:] = W221[0, :]
    W[5, 7:] = W221[1, :]
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    # 1. 4x4x4 part (first 4 cols of B)
    U22, V22, W22 = _strassen_2x2x2()
    
    # Kronecker product for 4x4x4
    U44 = np.zeros((16, 49), dtype=float)
    V44 = np.zeros((16, 49), dtype=float)
    W44 = np.zeros((16, 49), dtype=float)
    
    for t1 in range(7):
        for t2 in range(7):
            t = t1 * 7 + t2
            U44[:, t] = np.kron(U22[:, t1], U22[:, t2])
            V44[:, t] = np.kron(V22[:, t1], V22[:, t2])
            W44[:, t] = np.kron(W22[:, t1], W22[:, t2])
            
    # 2. 4x4x3 part (last 3 cols of B)
    U223, V223, W223 = _combine_2x2x3()
    
    # We need 4 blocks of 2x2x3.
    # A = [[A00, A01], [A10, A11]]
    # B = [[Btop], [Bbot]]
    # Ctop = A00 Btop + A01 Bbot
    # Cbot = A10 Btop + A11 Bbot
    
    # U443: (16, 44)
    # V443: (12, 44) -> m*p = 4*3 = 12
    # W443: (12, 44) -> p*n = 3*4 = 12
    
    U443 = np.zeros((16, 44), dtype=float)
    V443 = np.zeros((12, 44), dtype=float)
    W443 = np.zeros((12, 44), dtype=float)
    
    # Block 0: A00 Btop
    # U rows: 0..3 (A00)
    # V rows: 0..5 (Btop, j=0,1; k=0,1,2) -> j*3+k
    # W rows: 0..5 (Ctop, k=0,1,2; i=0,1) -> k*2+i
    
    U443[0:4, 0:11] = U223
    V443[0:6, 0:11] = V223
    W443[0:6, 0:11] = W223
    
    # Block 1: A01 Bbot
    # U rows: 4..7 (A01)
    # V rows: 6..11 (Bbot, j=2,3; k=0,1,2) -> j*3+k
    # W rows: 0..5 (Ctop, k=0,1,2; i=0,1) -> k*2+i
    
    U443[4:8, 11:22] = U223
    V443[6:12, 11:22] = V223
    W443[0:6, 11:22] = W223
    
    # Block 2: A10 Btop
    # U rows: 8..11 (A10)
    # V rows: 0..5 (Btop)
    # W rows: 6..11 (Cbot, k=0,1,2; i=0,1) -> k*2+i
    
    U443[8:12, 22:33] = U223
    V443[0:6, 22:33] = V223
    W443[6:12, 22:33] = W223
    
    # Block 3: A11 Bbot
    # U rows: 12..15 (A11)
    # V rows: 6..11 (Bbot)
    # W rows: 6..11 (Cbot)
    
    U443[12:16, 33:44] = U223
    V443[6:12, 33:44] = V223
    W443[6:12, 33:44] = W223
    
    # Combine
    total_rank = 49 + 44
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((28, total_rank), dtype=float)
    W = np.zeros((28, total_rank), dtype=float)
    
    # Copy 4x4x4 part
    U[:, :49] = U44
    # V44 is 16x49. V is 28x49.
    # V44 rows: j*4+k for j=0..3, k=0..3.
    # V rows: j*7+k for j=0..3, k=0..6.
    for j in range(4):
        for k in range(4):
            V[j*7+k, :49] = V44[j*4+k, :49]
            
    # W44 is 16x49. W is 28x49.
    # W44 rows: k*4+i for k=0..3, i=0..3.
    # W rows: k*4+i for k=0..6, i=0..3.
    W[:16, :49] = W44
    
    # Copy 4x4x3 part
    U[:, 49:] = U443
    # V443 is 12x44. V is 28x44.
    # V443 rows: j*3+k for j=0..3, k=0..2.
    # V rows: j*7+k for j=0..3, k=4..6.
    for j in range(4):
        for k in range(3):
            V[j*7 + (k+4), 49:] = V443[j*3+k, 49-49:] # Wait, V443 is 12x44, indices 0..11
            # V443 index j*3+k.
            # V index j*7 + (k+4).
            
    # W443 is 12x44. W is 28x44.
    # W443 rows: k*2+i for k=0..2, i=0..1? No.
    # W443 was constructed for 2x2x3.
    # In 4x4x3, W rows are k*4+i for k=4..6, i=0..3.
    # W443 rows are for Ctop (i=0,1) and Cbot (i=2,3)?
    # My W443 construction:
    # W443 rows 0..5 correspond to Ctop (i=0,1) and k=0..2.
    # Index in W443: k*2+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # Rows 6..11 correspond to Cbot (i=2,3) and k=0..2.
    # Index in W443: 6 + k*2 + (i-2).
    # k=0, i=2 -> 6
    # k=0, i=3 -> 7
    # k=1, i=2 -> 8
    # k=1, i=3 -> 9
    # k=2, i=2 -> 10
    # k=2, i=3 -> 11
    
    # In final W, index is k*4+i.
    # k=4, i=0 -> 16
    # k=4, i=1 -> 17
    # k=4, i=2 -> 18
    # k=4, i=3 -> 19
    # k=5, i=0 -> 20
    # ...
    # k=6, i=3 -> 27
    
    for k in range(3):
        for i in range(4):
            if i < 2:
                # Ctop
                idx_w443 = k*2 + i
            else:
                # Cbot
                idx_w443 = 6 + k*2 + (i-2)
            
            idx_w_final = (k+4)*4 + i
            W[idx_w_final, 49:] = W443[idx_w443, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
