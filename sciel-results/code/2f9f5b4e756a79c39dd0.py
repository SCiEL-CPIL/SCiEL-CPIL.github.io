import numpy as np

def construct():
    n, m, p = 2, 4, 8
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (a+d)g
    # M3 = a(f-h)
    # M4 = d(h-e)
    # M5 = (b+d)e
    # M6 = (a-c)(g+h)
    # M7 = (c+d)(e-f)
    
    # U: 4x7
    U222 = np.zeros((4, 7))
    # a=0, b=1, c=2, d=3
    U222[0, 0] = 1 # a
    U222[1, 0] = 1 # d? No, (a+d)
    # Let's define U such that U @ A_flat = intermediates
    # A_flat = [a, b, c, d]
    # M1 = (a+d)(e+h) -> U row for M1: [1, 0, 0, 1]
    # M2 = (a+d)g -> U row for M2: [1, 0, 0, 1]
    # M3 = a(f-h) -> U row for M3: [1, 0, 0, 0]
    # M4 = d(h-e) -> U row for M4: [0, 0, 0, 1]
    # M5 = (b+d)e -> U row for M5: [0, 1, 0, 1]
    # M6 = (a-c)(g+h) -> U row for M6: [1, 0, -1, 0]
    # M7 = (c+d)(e-f) -> U row for M7: [0, 0, 1, 1]
    
    U222 = [
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, -1, 0],
        [0, 0, 1, 1]
    ]
    U222 = np.array(U222).T # 4x7
    
    # V: 4x7
    # B_flat = [e, f, g, h]
    # M1 = (a+d)(e+h) -> V row: [1, 0, 0, 1]
    # M2 = (a+d)g -> V row: [0, 0, 1, 0]
    # M3 = a(f-h) -> V row: [0, 1, 0, -1]
    # M4 = d(h-e) -> V row: [-1, 0, 0, 1]
    # M5 = (b+d)e -> V row: [1, 0, 0, 0]
    # M6 = (a-c)(g+h) -> V row: [0, 0, 1, 1]
    # M7 = (c+d)(e-f) -> V row: [1, -1, 0, 0]
    
    V222 = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [1, -1, 0, 0]
    ]
    V222 = np.array(V222).T # 4x7
    
    # W: 4x7
    # C_flat = [c11, c12, c21, c22]
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    W222 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    W222 = np.array(W222).T # 4x7
    
    # Construct 2x2x4
    # Rank 14
    U224 = np.hstack([U222, U222]) # 4x14
    V224 = np.zeros((8, 14))
    V224[:4, :7] = V222
    V224[4:8, 7:14] = V222
    W224 = np.zeros((8, 14))
    W224[:4, :7] = W222
    W224[4:8, 7:14] = W222
    
    # Construct 2x2x8
    # Rank 28
    U228 = np.hstack([U224, U224]) # 4x28
    V228 = np.zeros((16, 28))
    V228[:8, :14] = V224
    V228[8:16, 14:28] = V224
    W228 = np.zeros((16, 28))
    W228[:8, :14] = W224
    W228[8:16, 14:28] = W224
    
    # Construct 2x4x8
    # Rank 56
    # A is 2x4. Flat indices:
    # 0: a00, 1: a01, 2: a02, 3: a03
    # 4: a10, 5: a11, 6: a12, 7: a13
    # A_L = [[a00, a01], [a10, a11]] -> indices 0,1,4,5
    # A_R = [[a02, a03], [a12, a13]] -> indices 2,3,6,7
    
    # U248: 8x56
    U248 = np.zeros((8, 56))
    # First 28 columns for A_L
    # Map U228 (4x28) to rows 0,1,4,5
    U248[0:2, :28] = U228[0:2, :] # a00, a01
    U248[4:6, :28] = U228[2:4, :] # a10, a11
    
    # Next 28 columns for A_R
    # Map U228 (4x28) to rows 2,3,6,7
    U248[2:4, 28:56] = U228[0:2, :] # a02, a03
    U248[6:8, 28:56] = U228[2:4, :] # a12, a13
    
    # V248: 32x56
    # B is 4x8. Flat indices:
    # 0-7: row 0
    # 8-15: row 1
    # 16-23: row 2
    # 24-31: row 3
    # B_T = rows 0,1 -> indices 0-15
    # B_B = rows 2,3 -> indices 16-31
    
    V248 = np.zeros((32, 56))
    # First 28 columns for B_T
    # Map V228 (16x28) to rows 0-15
    V248[0:16, :28] = V228
    
    # Next 28 columns for B_B
    # Map V228 (16x28) to rows 16-31
    V248[16:32, 28:56] = V228
    
    # W248: 16x56
    # C is 2x8. Flat indices:
    # 0-7: col 0-7 of row 0? No, k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # ...
    # W228 maps to 16 entries.
    # Both parts contribute to the same C.
    W248 = np.hstack([W228, W228])
    
    # Convert to lists of lists
    U = U248.tolist()
    V = V248.tolist()
    W = W248.tolist()
    
    return {"U": U, "V": V, "W": W}
