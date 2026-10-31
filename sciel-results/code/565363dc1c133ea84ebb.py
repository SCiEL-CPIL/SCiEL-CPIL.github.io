import numpy as np

def construct():
    """4x4 matrix multiplication via rank-49 tensor decomposition using nested Strassen structure."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen 2x2: 7 multiplications
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11),
    # M4=A11(B10-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01),
    # M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    # Coefficients as lists of (row, col, coeff) for each of the 7 mults
    A_coeffs = [
        [(0,0,1),(1,1,1)],   # M1
        [(1,0,1),(1,1,1)],   # M2
        [(0,0,1)],            # M3
        [(1,1,1)],            # M4
        [(0,0,1),(0,1,1)],   # M5
        [(1,0,1),(0,0,-1)],  # M6
        [(0,1,1),(1,1,-1)]   # M7
    ]
    B_coeffs = [
        [(0,0,1),(1,1,1)],   # M1
        [(0,0,1)],            # M2
        [(0,1,1),(1,1,-1)],  # M3
        [(1,0,1),(0,0,-1)],  # M4
        [(1,1,1)],            # M5
        [(0,0,1),(0,1,1)],   # M6
        [(1,0,1),(1,1,1)]    # M7
    ]
    C_coeffs = [
        [(0,0,1),(1,1,1)],   # M1
        [(1,0,1),(1,1,-1)],  # M2
        [(0,1,1),(1,1,1)],   # M3
        [(0,0,1),(1,0,1)],   # M4
        [(0,0,-1),(0,1,1)],  # M5
        [(1,1,1)],            # M6
        [(0,0,1)]             # M7
    ]
    
    # Apply at two levels: top (2x2 blocks) and bottom (scalars within blocks)
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            
            # U: A coefficients
            for (br, bc, ca) in A_coeffs[t]:
                for (lr, lc, cb) in A_coeffs[s]:
                    i = 2*br + lr
                    j = 2*bc + lc
                    U[i*m + j, col] += ca * cb
            
            # V: B coefficients
            for (br, bc, cb) in B_coeffs[t]:
                for (lr, lc, cc) in B_coeffs[s]:
                    j = 2*br + lr
                    k = 2*bc + lc
                    V[j*p + k, col] += cb * cc
            
            # W: C coefficients (trace convention: W[k*n+i, col])
            for (br, bc, cc) in C_coeffs[t]:
                for (lr, lc, cd) in C_coeffs[s]:
                    i = 2*br + lr
                    k = 2*bc + lc
                    W[k*n + i, col] += cc * cd
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
