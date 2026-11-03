"""
Optimized matrix multiplication tensor decomposition.
Handles specific small cases with known optimal ranks (e.g., Strassen).
Falls back to schoolbook decomposition for general cases.
"""
import numpy as np

DIMENSIONS = [2, 5, 6]

def _strassen_222():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Indices:
    U: (2*2, 7) -> (4, 7)
    V: (2*2, 7) -> (4, 7)
    W: (2*2, 7) -> (4, 7)
    Convention: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products
    # M1 = (A11 - A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We need to map these to the tensor entries U, V, W.
    # The decomposition is M = sum_c u_c v_c w_c^T (conceptually).
    # Actually, the standard Strassen decomposition provides the coefficients.
    # Let's define the vectors u, v, w for each of the 7 terms.
    
    # u vectors are in R^4 (indices 00, 01, 10, 11)
    # v vectors are in R^4
    # w vectors are in R^4
    
    u = [
        [1, 0, 0, -1],  # M1: A11 - A22 -> indices (0,0) and (1,1)
        [0, 0, 1, 1],   # M2: A21 + A22 -> indices (1,0) and (1,1)
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1]   # M7: A12 - A22
    ]
    
    v = [
        [1, 0, 0, 1],   # M1: B11 + B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12 - B22
        [0, -1, 1, 0],  # M4: B21 - B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11 + B12
        [0, 0, 1, 1]    # M7: B21 + B22
    ]
    
    # The output C is given by:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # We need w vectors such that sum(w_c * M_c) = C.
    # The tensor decomposition is M = sum u_c \otimes v_c \otimes w_c.
    # The "w" part corresponds to the linear combination of M_c to form C.
    # So w_c is the vector of coefficients for C given term c.
    # C indices: 00, 01, 10, 11.
    
    w = [
        [1, 0, 0, 1],   # M1 contributes to C11 and C22
        [0, 0, 1, -1],  # M2 contributes to C21 and C22
        [0, 1, 0, 1],   # M3 contributes to C12 and C22
        [1, 0, 1, 0],   # M4 contributes to C11 and C21
        [-1, 1, 0, 0],  # M5 contributes to C11 and C12
        [0, 0, 0, 1],   # M6 contributes to C22
        [1, 0, 0, 0]    # M7 contributes to C11
    ]
    
    # Convert to lists of lists for the required interface
    U = [list(map(int, u[i])) for i in range(4)] # Wait, u is 7x4. U needs to be (4, 7) or (7, 4)?
    # Interface: U shape (n*m, rank). n*m=4, rank=7.
    # So U[row, col] = u[col, row]?
    # In the schoolbook: U[i*m+j, c] = 1.
    # So U is 4x7. Row is (i,j), Col is c.
    # My u list is 7 vectors of length 4.
    # So U = zip(*u) essentially.
    
    U = [[u[c][r] for c in range(7)] for r in range(4)]
    V = [[v[c][r] for c in range(7)] for r in range(4)]
    W = [[w[c][r] for c in range(7)] for r in range(4)]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()
    
    # For 2,5,6, use schoolbook
    return _schoolbook(n, m, p)
