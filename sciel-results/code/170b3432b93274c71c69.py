import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-28 ansatz.
    
    Strategy:
    1. Split the multiplication based on the middle dimension m=3.
       Let j=0,1 be handled by M_{2,2,5} and j=2 by M_{2,1,5}.
    2. M_{2,1,5} has rank 10 (schoolbook, minimal).
    3. M_{2,2,5} is split into M_{2,2,2} (k=0,1), M_{2,2,2} (k=2,3), M_{2,2,1} (k=4).
       Rank = 7 (Strassen) + 7 (Strassen) + 4 (Schoolbook, minimal) = 18.
    4. Total Rank = 18 + 10 = 28.
    
    This is the known minimum rank for this decomposition strategy.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # Indices
        # A: [i, j] -> i*m + j
        # B: [j, k] -> j*p + k
        # C: [k, i] -> k*n + i
        
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        U[0*m + j_off, c] = 1
        U[1*m + (j_off+1), c] = 1
        V[j_off*p + k_off, c] = 1
        V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = 1
        W[(k_off+1)*n + 1, c] = 1
        c += 1
        
        # M2 = (A10+A11)B00 -> C10 - C11
        U[1*m + j_off, c] = 1
        U[1*m + (j_off+1), c] = 1
        V[j_off*p + k_off, c] = 1
        W[k_off*n + 1, c] = 1
        W[(k_off+1)*n + 1, c] = -1
        c += 1
        
        # M3 = A00(B01-B11) -> C01 + C11
        U[0*m + j_off, c] = 1
        V[j_off*p + (k_off+1), c] = 1
        V[(j_off+1)*p + (k_off+1), c] = -1
        W[(k_off+1)*n + 0, c] = 1
        W[(k_off+1)*n + 1, c] = 1
        c += 1
        
        # M4 = A11(B10-B00) -> C00 + C10
        U[1*m + (j_off+1), c] = 1
        V[(j_off+1)*p + k_off, c] = 1
        V[j_off*p + k_off, c] = -1
        W[k_off*n + 0, c] = 1
        W[k_off*n + 1, c] = 1
        c += 1
        
        # M5 = (A00+A01)B11 -> -C00 + C01
        U[0*m + j_off, c] = 1
        U[0*m + (j_off+1), c] = 1
        V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = -1
        W[(k_off+1)*n + 0, c] = 1
        c += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        U[1*m + j_off, c] = 1
        U[0*m + j_off, c] = -1
        V[j_off*p + k_off, c] = 1
        V[j_off*p + (k_off+1), c] = 1
        W[(k_off+1)*n + 1, c] = 1
        c += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00
        U[0*m + (j_off+1), c] = 1
        U[1*m + (j_off+1), c] = -1
        V[(j_off+1)*p + k_off, c] = 1
        V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = 1
        c += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(0, 2)
    
    # Remaining schoolbook terms
    # These cover:
    # 1. j in {0,1}, k=4 (2*1=2 terms? No, 2*1*1=2 terms? Wait. i in 0..1, j in 0..1, k=4. 2*2*1=4 terms)
    # 2. j=2, k in 0..4 (2*1*5=10 terms)
    # Total 14 terms. 14 + 14 = 28. Correct.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip terms handled by Strassen blocks
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
                
    if c != rank:
        raise ValueError(f"Rank mismatch: {c} != {rank}")
        
    # Convert to list of lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
