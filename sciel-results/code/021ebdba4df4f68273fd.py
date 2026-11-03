import numpy as np

def construct():
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    U222 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    # Wait, let's verify Strassen U, V, W.
    # c0 = (A00+A11)(B00+B11)
    # c1 = (A10+A11)B00
    # c2 = A00(B01-B11)
    # c3 = A11(B10-B00)
    # c4 = (A00+A01)B11
    # c5 = (A10-A00)(B00+B01)
    # c6 = (A01-A11)(B10+B11)
    
    # U columns:
    # c0: A00+A11 -> [1,0,0,1]
    # c1: A10+A11 -> [0,0,1,1]
    # c2: A00 -> [1,0,0,0]
    # c3: A11 -> [0,0,0,1]
    # c4: A00+A01 -> [1,1,0,0]
    # c5: A10-A00 -> [-1,0,1,0]
    # c6: A01-A11 -> [0,1,0,-1]
    
    U222 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    # V columns:
    # c0: B00+B11 -> [1,0,0,1]
    # c1: B00 -> [1,0,0,0]
    # c2: B01-B11 -> [0,1,0,-1]
    # c3: B10-B00 -> [-1,0,1,0]
    # c4: B11 -> [0,0,0,1]
    # c5: B00+B01 -> [1,1,0,0]
    # c6: B10+B11 -> [0,0,1,1]
    
    V222 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    # W columns:
    # C00 = c0 + c3 - c5 - c6
    # C01 = c4 + c6
    # C10 = c2 + c5
    # C11 = c0 + c1 - c2 + c3
    
    # W index order: k*n+i. n=2.
    # k=0, i=0 -> C00
    # k=0, i=1 -> C10
    # k=1, i=0 -> C01
    # k=1, i=1 -> C11
    
    # c0: C00+=1, C11+=1 -> [1,0,0,1]
    # c1: C11+=1 -> [0,0,0,1]
    # c2: C10+=1, C11-=1 -> [0,1,0,-1]
    # c3: C00+=1, C11+=1 -> [1,0,0,1]
    # c4: C01+=1 -> [0,0,1,0]
    # c5: C00-=1, C10+=1 -> [-1,1,0,0]
    # c6: C00-=1, C01+=1 -> [-1,0,1,0]
    
    W222 = np.array([
        [1, 0, 0, 1, 0, -1, -1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, -1, 1, 0, 0, 0]
    ], dtype=float)
    
    # Schoolbook for 1x1x2
    # n=1, m=1, p=2
    # U: 1x1, V: 1x2, W: 2x1
    # Rank 2
    # c0: A00 B00 -> C00
    # c1: A00 B01 -> C01
    
    U112 = np.array([[1, 1]], dtype=float)
    V112 = np.array([[1, 0], [0, 1]], dtype=float)
    # W index k*n+i. n=1.
    # k=0, i=0 -> C00
    # k=1, i=0 -> C01
    W112 = np.array([[1], [1]], dtype=float)
    
    # Kronecker Product
    # M444 = M222 x M222
    U444 = np.kron(U222, U222)
    V444 = np.kron(V222, V222)
    W444 = np.kron(W222, W222)
    
    # M448 = M444 x M112
    U448 = np.kron(U444, U112)
    V448 = np.kron(V444, V112)
    W448 = np.kron(W444, W112)
    
    return {"U": U448.tolist(), "V": V448.tolist(), "W": W448.tolist()}
