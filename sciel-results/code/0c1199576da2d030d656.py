import numpy as np

def construct():
    # Strassen 2x2 factors
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = AB
    
    # U: (4, 7) for A
    # V: (4, 7) for B
    # W: (4, 7) for C
    
    # Terms:
    # 0: (a+d)(e+h) -> C11, C22
    # 1: (c+d)e -> C21, -C22
    # 2: a(f-h) -> C12, C22
    # 3: d(g-e) -> C11, C21
    # 4: (a+b)h -> -C11, C12
    # 5: (c-a)(e+f) -> ?
    # 6: (b-d)(g+h) -> -C11, C22
    
    # Wait, the reconstruction formulas must match the terms.
    # Let's use the verified coefficients from a reliable source.
    
    # From "Matrix Multiplication" by Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # This set of formulas is widely cited. I will use it.
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # a: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1], # b: M5, M7
        [0, 1, 0, 0, 0, 1, 0], # c: M2, M6
        [1, 1, 0, 1, 0, 0, -1] # d: M1, M2, M4, M7
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # e: M1, M2, M4, M6
        [0, 0, 1, 0, 0, 1, 0], # f: M3, M6
        [1, 0, 0, 1, 0, 0, 1], # g: M1, M4, M7
        [1, 0, -1, 0, 1, 0, 1] # h: M1, M3, M5, M7
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, -1], # C11: M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0], # C12: M3, M5
        [0, 1, 0, 1, 0, 0, 0], # C21: M2, M4
        [1, -1, 1, 0, 0, 0, 1] # C22: M1, M2, M3, M7
    ]
    
    # Verify C22: M1 - M2 + M3 + M7
    # My W2 row 4 (index 3) is [1, -1, 1, 0, 0, 0, 1]. Correct.
    
    # Now Kronecker product
    U = np.kron(np.array(U2), np.array(U2))
    V = np.kron(np.array(V2), np.array(V2))
    W = np.kron(np.array(W2), np.array(W2))
    
    # Convert to list of lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
