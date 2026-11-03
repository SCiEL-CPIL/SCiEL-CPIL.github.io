"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    rank = 36
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0

    def add_strassen_offset(j_base, k_base):
        nonlocal col
        
        def a(i, j): return i * m + (j_base + j)
        def b(j, k): return (j_base + j) * p + (k_base + k)
        def c(i, k): return (k_base + k) * n + i
        
        # M1 = (A00+A11)(B00+B11)
        U[a(0,0), col] = 1; V[b(0,0), col] = 1; W[c(0,0), col] = 1
        U[a(1,1), col] = 1; V[b(1,1), col] = 1; W[c(1,1), col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a(1,0), col] = 1; V[b(0,0), col] = 1; W[c(1,0), col] = 1
        U[a(1,1), col] = 1; V[b(0,0), col] = 1; W[c(1,1), col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a(0,0), col] = 1
        V[b(0,1), col] = 1
        V[b(1,1), col] = -1
        W[c(0,1), col] = 1
        W[c(1,1), col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a(1,1), col] = 1
        V[b(1,0), col] = 1
        V[b(0,0), col] = -1
        W[c(0,0), col] = 1
        W[c(1,0), col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a(0,0), col] = 1
        U[a(0,1), col] = 1
        V[b(1,1), col] = 1
        W[c(0,0), col] = -1
        W[c(0,1), col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a(1,0), col] = 1
        U[a(0,0), col] = -1
        V[b(0,0), col] = 1
        V[b(0,1), col] = 1
        W[c(1,0), col] = 1
        W[c(1,1), col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a(0,1), col] = 1
        U[a(1,1), col] = -1
        V[b(1,0), col] = 1
        V[b(1,1), col] = 1
        W[c(0,0), col] = 1
        col += 1

    def add_schoolbook_offset(j_base, k_base):
        nonlocal col
        # 2x2 * 2x1
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        def a(i, j): return i * m + (j_base + j)
        def b(j, k): return (j_base + j) * p + (k_base + k)
        def c(i, k): return (k_base + k) * n + i
        
        # Term 1: A00 B00 -> C00
        U[a(0,0), col] = 1
        V[b(0,0), col] = 1
        W[c(0,0), col] = 1
        col += 1
        
        # Term 2: A01 B10 -> C00
        U[a(0,1), col] = 1
        V[b(1,0), col] = 1
        W[c(0,0), col] = 1
        col += 1
        
        # Term 3: A10 B00 -> C10
        U[a(1,0), col] = 1
        V[b(0,0), col] = 1
        W[c(1,0), col] = 1
        col += 1
        
        # Term 4: A11 B10 -> C10
        U[a(1,1), col] = 1
        V[b(1,0), col] = 1
        W[c(1,0), col] = 1
        col += 1

    # Execute blocks
    # Block 1: j=0,1
    add_strassen_offset(0, 0)
    add_strassen_offset(0, 2)
    add_schoolbook_offset(0, 4)
    
    # Block 2: j=2,3
    add_strassen_offset(2, 0)
    add_strassen_offset(2, 2)
    add_schoolbook_offset(2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
