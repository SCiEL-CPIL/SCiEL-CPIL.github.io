import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # Strassen 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # m1: (a+d)(e+h)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # m2: (c+d)e
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # m3: a(f-h)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # m4: d(g-e)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # m5: (a+b)h
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # m6: (c-a)(e+f)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # m7: (b-d)(g+h)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # M_444: Kronecker product
    U44 = np.kron(np.array(U2), np.array(U2))
    V44 = np.kron(np.array(V2), np.array(V2))
    W44 = np.kron(np.array(W2), np.array(W2))
    
    # M_442: 4 blocks of M_222
    # A is 4x4, B is 4x2, C is 4x2
    # B = [[B00], [B10]] where B00, B10 are 2x2
    # C = A B = [[A00 B00 + A01 B10], [A10 B00 + A11 B10]]
    # 4 multiplications:
    # C00 = A00 B00 (M_222)
    # C01 = A01 B10 (M_222)
    # C10 = A10 B00 (M_222)
    # C11 = A11 B10 (M_222)
    
    U42 = np.zeros((16, 28))
    V42 = np.zeros((8, 28))
    W42 = np.zeros((8, 28))
    
    # Block 0: A00 (0:2, 0:2), B00 (0:2, 0:2), C00 (0:2, 0:2)
    # U indices: 0-3
    # V indices: 0-3
    # W indices: 0-3
    U42[0:4, 0:7] = U2
    V42[0:4, 0:7] = V2
    W42[0:4, 0:7] = W2
    
    # Block 1: A01 (0:2, 2:4), B10 (2:4, 0:2), C01 (0:2, 2:4)
    # U indices: 4-7 (rows 0-1, cols 2-3)
    # V indices: 4-7 (rows 2-3, cols 0-1)
    # W indices: 4-7 (rows 0-1, cols 2-3)
    U42[4:8, 7:14] = U2
    V42[4:8, 7:14] = V2
    W42[4:8, 7:14] = W2
    
    # Block 2: A10 (2:4, 0:2), B00 (0:2, 0:2), C10 (2:4, 0:2)
    # U indices: 8-11 (rows 2-3, cols 0-1)
    # V indices: 0-3
    # W indices: 0-3 (rows 2-3, cols 0-1) -> W index is k*n+i.
    # C10 is rows 2-3, cols 0-1.
    # W index for C10: k=0,1; i=2,3.
    # W2 indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) in 2x2?
    # W2[k*n+i]. n=2.
    # W2[0] -> (0,0). W2[1] -> (0,1). W2[2] -> (1,0). W2[3] -> (1,1).
    # For C10 (rows 2-3, cols 0-1):
    # C[2,0] -> k=0, i=2. Index 0*4+2 = 2.
    # C[2,1] -> k=1, i=2. Index 1*4+2 = 6.
    # C[3,0] -> k=0, i=3. Index 0*4+3 = 3.
    # C[3,1] -> k=1, i=3. Index 1*4+3 = 7.
    # So W indices are 2, 6, 3, 7.
    # This mapping is complex.
    
    # Let's just use the Kronecker approach for M_444 and a simpler block approach for M_442.
    
    # Actually, M_442 can be seen as M_444 with B padded?
    # No, B is 4x2.
    
    # Let's stick to the block construction for M_442 carefully.
    
    # U42 is 16x28.
    # V42 is 8x28.
    # W42 is 8x28? No, W is p*n = 2*4 = 8.
    
    # Block 0: A00, B00, C00
    # A00 is 2x2 at (0,0). U indices 0,1,2,3.
    # B00 is 2x2 at (0,0). V indices 0,1,2,3.
    # C00 is 2x2 at (0,0). W indices 0,1,2,3.
    U42[0:4, 0:7] = U2
    V42[0:4, 0:7] = V2
    W42[0:4, 0:7] = W2
    
    # Block 1: A01, B10, C01
    # A01 is 2x2 at (0,2). U indices 4,5,6,7.
    # B10 is 2x2 at (2,0). V indices 4,5,6,7.
    # C01 is 2x2 at (0,2). W indices: k=0,1; i=0,1.
    # W index k*n+i. n=4.
    # C[0,2] -> k=2, i=0. Index 2*4+0 = 8.
    # C[0,3] -> k=3, i=0. Index 3*4+0 = 12.
    # C[1,2] -> k=2, i=1. Index 2*4+1 = 9.
    # C[1,3] -> k=3, i=1. Index 3*4+1 = 13.
    # So W indices are 8, 12, 9, 13.
    # W2 has 4 entries. We map them to 8, 12, 9, 13.
    U42[4:8, 7:14] = U2
    V42[4:8, 7:14] = V2
    W42[8, 7:14] = W2[0]
    W42[12, 7:14] = W2[1]
    W42[9, 7:14] = W2[2]
    W42[13, 7:14] = W2[3]
    
    # Block 2: A10, B00, C10
    # A10 is 2x2 at (2,0). U indices 8,9,10,11.
    # B00 is 2x2 at (0,0). V indices 0,1,2,3.
    # C10 is 2x2 at (2,0). W indices: k=0,1; i=2,3.
    # C[2,0] -> k=0, i=2. Index 2.
    # C[2,1] -> k=1, i=2. Index 6.
    # C[3,0] -> k=0, i=3. Index 3.
    # C[3,1] -> k=1, i=3. Index 7.
    U42[8:12, 14:21] = U2
    V42[0:4, 14:21] = V2
    W42[2, 14:21] = W2[0]
    W42[6, 14:21] = W2[1]
    W42[3, 14:21] = W2[2]
    W42[7, 14:21] = W2[3]
    
    # Block 3: A11, B10, C11
    # A11 is 2x2 at (2,2). U indices 12,13,14,15.
    # B10 is 2x2 at (2,0). V indices 4,5,6,7.
    # C11 is 2x2 at (2,2). W indices: k=2,3; i=2,3.
    # C[2,2] -> k=2, i=2. Index 10.
    # C[2,3] -> k=3, i=2. Index 14.
    # C[3,2] -> k=2, i=3. Index 11.
    # C[3,3] -> k=3, i=3. Index 15.
    U42[12:16, 21:28] = U2
    V42[4:8, 21:28] = V2
    W42[10, 21:28] = W2[0]
    W42[14, 21:28] = W2[1]
    W42[11, 21:28] = W2[2]
    W42[15, 21:28] = W2[3]
    
    # Combine
    U = np.hstack([U44, U42])
    V = np.zeros((24, 77))
    V[0:16, 0:49] = V44
    V[16:24, 49:77] = V42
    W = np.zeros((24, 77))
    W[0:16, 0:49] = W44
    W[16:24, 49:77] = W42
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
