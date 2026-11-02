import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _add_strassen_222(U, V, W, col, i_off, j_off, k_off):
    """
    Add a 2x2x2 Strassen block.
    U shape: (n*m, rank), indices i*m+j
    V shape: (m*p, rank), indices j*p+k
    W shape: (p*n, rank), indices k*n+i
    
    i_off, j_off, k_off are the starting indices for i, j, k.
    The block covers i in [i_off, i_off+1], j in [j_off, j_off+1], k in [k_off, k_off+1].
    """
    n, m, p = 2, 3, 5 # Dimensions for index calculation
    
    # M1: (A00+A11)(B00+B11)
    # A indices: (i_off, j_off), (i_off+1, j_off+1)
    # B indices: (j_off, k_off), (j_off+1, k_off+1)
    # C indices: (k_off, i_off), (k_off+1, i_off+1)
    U[i_off*m + j_off][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[j_off*p + k_off][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[k_off*n + i_off][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M2: (A10+A11)B00
    # A indices: (i_off+1, j_off), (i_off+1, j_off+1)
    # B indices: (j_off, k_off)
    # C indices: (k_off, i_off+1), (k_off+1, i_off+1) -- Wait, Strassen M2 contributes to C10 and C11?
    # Standard Strassen:
    # M1 = (A11 - A10)(B10 + B11) -> C10 += M1, C11 += M1? No.
    # Let's stick to the standard 7 multiplications:
    # M1 = A00(B00-B11)
    # M2 = (A00+A01)B11
    # M3 = (A10+A11)B00
    # M4 = A11(B01-B00)
    # M5 = (A00+A11)(B00+B11)
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    
    # C00 = M5 + M4 - M2 - M7
    # C01 = M1 + M2
    # C10 = M3 + M6
    # C11 = M5 - M1 + M3 + M7
    
    # Let's re-verify the previous code's mapping or use a known correct one.
    # The previous code used:
    # M1: (A00+A11)(B00+B11) -> C00 +=, C11 +=  (This is M5 in standard)
    # M2: (A10+A11)B00 -> C10 +=, C11 -= (This is M3 in standard, but signs differ?)
    # Standard M3 = (A10+A11)B00.
    # C10 = M3 + M6. C11 = M5 - M1 + M3 + M7.
    # If M2 is M3, then C10 += M2, C11 += M2.
    # The previous code had C11 -= M2. This might be where the error was or just a different variant.
    
    # Let's use the standard variant explicitly.
    # Reset col for this block? No, col is passed by value in Python, so I need to return the new col.
    
    # M1: A00(B00-B11)
    U[i_off*m + j_off][col] = 1
    V[j_off*p + k_off][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = -1
    W[(k_off+1)*n + i_off][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M2: (A00+A01)B11
    U[i_off*m + j_off][col] = 1
    U[i_off*m + (j_off+1)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[k_off*n + i_off][col] = -1
    W[(k_off+1)*n + i_off][col] = 1
    col += 1
    
    # M3: (A10+A11)B00
    U[(i_off+1)*m + j_off][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[j_off*p + k_off][col] = 1
    W[k_off*n + (i_off+1)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M4: A11(B01-B00)
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[j_off*p + (k_off+1)][col] = 1
    V[j_off*p + k_off][col] = -1
    W[k_off*n + i_off][col] = 1
    W[k_off*n + (i_off+1)][col] = 1
    col += 1
    
    # M5: (A00+A11)(B00+B11)
    U[i_off*m + j_off][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[j_off*p + k_off][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[k_off*n + i_off][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M6: (A01-A11)(B10+B11)
    U[i_off*m + (j_off+1)][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = -1
    V[(j_off+1)*p + k_off][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[k_off*n + (i_off+1)][col] = 1
    col += 1
    
    # M7: (A10-A00)(B00+B01)
    U[(i_off+1)*m + j_off][col] = 1
    U[i_off*m + j_off][col] = -1
    V[j_off*p + k_off][col] = 1
    V[j_off*p + (k_off+1)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    return col

def _add_schoolbook_221(U, V, W, col, i_off, j_off, k_off):
    """
    Add a 2x2x1 schoolbook block.
    i in [i_off, i_off+1], j in [j_off, j_off+1], k in [k_off, k_off]
    """
    n, m, p = 2, 3, 5
    for i in range(2):
        for j in range(2):
            U[(i_off+i)*m + (j_off+j)][col] = 1
            V[(j_off+j)*p + k_off][col] = 1
            W[k_off*n + (i_off+i)][col] = 1
            col += 1
    return col

def _add_schoolbook_215(U, V, W, col, i_off, j_off, k_off):
    """
    Add a 2x1x5 schoolbook block.
    i in [i_off, i_off+1], j in [j_off, j_off], k in [k_off, k_off+4]
    """
    n, m, p = 2, 3, 5
    for i in range(2):
        for k in range(5):
            U[(i_off+i)*m + j_off][col] = 1
            V[j_off*p + (k_off+k)][col] = 1
            W[(k_off+k)*n + (i_off+i)][col] = 1
            col += 1
    return col

def construct():
    n, m, p = 2, 3, 5
    # Total rank:
    # J1 (j=0,1):
    #   K1 (k=0,1): Strassen 2x2x2 -> 7
    #   K2 (k=2,3): Strassen 2x2x2 -> 7
    #   K3 (k=4):   Schoolbook 2x2x1 -> 4
    #   Sum: 18
    # J2 (j=2):
    #   K (k=0..4): Schoolbook 2x1x5 -> 10
    #   Sum: 10
    # Total: 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block J1 (j=0,1)
    # K1 (k=0,1)
    col = _add_strassen_222(U, V, W, col, 0, 0, 0)
    # K2 (k=2,3)
    col = _add_strassen_222(U, V, W, col, 0, 0, 2)
    # K3 (k=4)
    col = _add_schoolbook_221(U, V, W, col, 0, 0, 4)
    
    # Block J2 (j=2)
    col = _add_schoolbook_215(U, V, W, col, 0, 2, 0)
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    # Optional: Verify
    # U_np = np.array(U, dtype=float)
    # V_np = np.array(V, dtype=float)
    # W_np = np.array(W, dtype=float)
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             val = sum(U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r] for r in range(rank))
    #             if abs(val - 1.0) > 1e-9:
    #                 raise ValueError("Decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
