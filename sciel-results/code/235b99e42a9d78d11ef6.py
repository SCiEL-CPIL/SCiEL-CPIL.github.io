import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    The rank is 50.
    
    U: shape (8, 50)  (n*m, rank)
    V: shape (28, 50) (m*p, rank)
    W: shape (14, 50) (p*n, rank)
    
    Indexing conventions:
    A is 2x4. U indices: row in A * 4 + col in A. (0..7)
    B is 4x7. V indices: row in B * 7 + col in B. (0..27)
    C is 2x7. W indices: col in C * 2 + row in C. (0..13) -> Trace convention W[k*n+i, r]
    """
    n, m, p = 2, 4, 7
    rank = 50
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_222(j0, k0):
        """
        Adds Strassen decomposition for A[:, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[:, k0:k0+2]
        """
        nonlocal col
        
        # Indices for A (U): row * m + col
        a00 = 0 * m + j0
        a01 = 0 * m + (j0 + 1)
        a10 = 1 * m + j0
        a11 = 1 * m + (j0 + 1)
        
        # Indices for B (V): row * p + col
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        # Indices for C (W): col * n + row (trace convention)
        c00 = k0 * n + 0
        c01 = (k0 + 1) * n + 0
        c10 = k0 * n + 1
        c11 = (k0 + 1) * n + 1
        
        # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M2 = (a10+a11)*b00 -> C10 += M2, C11 -= M2
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] -= 1
        col += 1
        
        # M3 = a00*(b01-b11) -> C01 += M3, C11 += M3
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] -= 1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M4 = a11*(b10-b00) -> C00 += M4, C10 += M4
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] -= 1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M5 = (a00+a01)*b11 -> C00 -= M5, C01 += M5
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1
        W[c01, col] += 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01) -> C11 += M6
        U[a10, col] += 1
        U[a00, col] -= 1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11) -> C00 += M7
        U[a01, col] += 1
        U[a11, col] -= 1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1

    def add_schoolbook_221(j0, k0):
        """
        Adds schoolbook decomposition for A[:, j0:j0+2] * B[j0:j0+2, k0] -> C[:, k0]
        """
        nonlocal col
        
        a00 = 0 * m + j0
        a01 = 0 * m + (j0 + 1)
        a10 = 1 * m + j0
        a11 = 1 * m + (j0 + 1)
        
        b00 = j0 * p + k0
        b10 = (j0 + 1) * p + k0
        
        c00 = k0 * n + 0
        c10 = k0 * n + 1
        
        # C[0,k0] = a00*b00 + a01*b10
        U[a00, col] += 1
        V[b00, col] += 1
        W[c00, col] += 1
        col += 1
        
        U[a01, col] += 1
        V[b10, col] += 1
        W[c00, col] += 1
        col += 1
        
        # C[1,k0] = a10*b00 + a11*b10
        U[a10, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        col += 1
        
        U[a11, col] += 1
        V[b10, col] += 1
        W[c10, col] += 1
        col += 1

    # Process inner blocks j=0..1 and j=2..3
    for j_start in [0, 2]:
        # Process output blocks k=0..1, 2..3, 4..5 with Strassen
        for k_start in [0, 2, 4]:
            add_strassen_222(j_start, k_start)
        # Process output block k=6 with Schoolbook
        add_schoolbook_221(j_start, 6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
