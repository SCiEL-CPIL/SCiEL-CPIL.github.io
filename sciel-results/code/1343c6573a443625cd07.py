"""
Tensor decomposition for 2x2 matrix multiplication.
Implements Strassen's algorithm, which achieves the theoretical minimum rank of 7.
"""

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-7 tensor decomposition for 2x2 matrix multiplication (Strassen).
    
    The decomposition factors the multiplication tensor T into U, V, W such that:
    T[i, j, k] = sum_{r} U[i, r] * V[j, r] * W[k, r]
    
    Indexing Convention:
    U has shape (n*m, rank) = (4, 7). Row index = i*m + j for A[i, j].
    V has shape (m*p, rank) = (4, 7). Row index = j*p + k for B[j, k].
    W has shape (p*n, rank) = (4, 7). Row index = k*n + i for C[i, k] (Trace Convention).
    
    For n=m=p=2:
    A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    C indices (W[k*n+i]): 
       C[0,0] -> k=0,i=0 -> 0
       C[1,0] -> k=0,i=1 -> 1
       C[0,1] -> k=1,i=0 -> 2
       C[1,1] -> k=1,i=1 -> 3
    
    Strassen's 7 products and their contributions to C:
    M1 = (A00 + A11)(B00 + B11)  -> C00 += M1, C11 += M1
    M2 = (A10 + A11)B00          -> C10 += M2, C11 -= M2
    M3 = A00(B01 - B11)          -> C01 += M3, C11 += M3
    M4 = A11(B10 - B00)          -> C00 += M4, C10 += M4
    M5 = (A00 + A01)B11          -> C00 -= M5, C01 += M5
    M6 = (A10 - A00)(B00 + B01)  -> C11 += M6
    M7 = (A01 - A11)(B10 + B11)  -> C00 += M7
    
    We construct U, V, W such that for each r, U[:,r], V[:,r], W[:,r] are the coefficients.
    """
    
    # U: Coefficients for A elements in each product
    # Columns are the 7 products M1..M7
    # Rows are A00, A01, A10, A11
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # A00: M1, M3, M5, M6
        [0,  0,  0,  0,  1,  0,  1],  # A01: M5, M7
        [0,  1,  0,  0,  0,  1,  0],  # A10: M2, M6
        [1,  1,  0,  1,  0,  0, -1]   # A11: M1, M2, M4, M7
    ]
    
    # V: Coefficients for B elements in each product
    # Rows are B00, B01, B10, B11
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # B00: M1, M2, M4, M6
        [0,  0,  1,  0,  0,  1,  0],  # B01: M3, M6
        [0,  0,  0,  1,  0,  0,  1],  # B10: M4, M7
        [1,  0, -1,  0,  1,  0,  1]   # B11: M1, M3, M5, M7
    ]
    
    # W: Coefficients for C elements in each product
    # Rows are C00, C10, C01, C11 (based on W[k*n+i] indexing: 0:C00, 1:C10, 2:C01, 3:C11)
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # C00: M1, M4, M5, M7
        [0,  1,  0,  1,  0,  0,  0],  # C10: M2, M4
        [0,  0,  1,  0,  1,  0,  0],  # C01: M3, M5
        [1, -1,  1,  0,  0,  1,  0]   # C11: M1, M2, M3, M6
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
