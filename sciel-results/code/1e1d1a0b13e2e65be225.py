import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def set_term(u_indices, v_indices, w_indices, values):
        nonlocal col
        # u_indices, v_indices, w_indices are lists of (index, value) pairs
        for idx, val in u_indices:
            U[idx, col] = val
        for idx, val in v_indices:
            V[idx, col] = val
        for idx, val in w_indices:
            W[idx, col] = val
        col += 1

    # Strassen Multiplications
    # M1 = (A00 + A11)(B00 + B11)
    # A00: idx 0, A11: idx 3
    # B00: idx 0 (j=0,k=0), B11: idx 3 (j=1,k=1)
    # C00: idx 0 (k=0,i=0), C11: idx 3 (k=1,i=1)
    set_term([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)])
    
    # M2 = (A10 + A11)B00
    # A10: idx 1, A11: idx 3
    # B00: idx 0
    # C10: idx 1 (k=0,i=1)
    set_term([(1,1), (3,1)], [(0,1)], [(1,1)])
    
    # M3 = A00(B01 - B11)
    # A00: idx 0
    # B01: idx 1 (j=0,k=1), B11: idx 3 (j=1,k=1)
    # C01: idx 2 (k=1,i=0)
    set_term([(0,1)], [(1,1), (3,-1)], [(2,1)])
    
    # M4 = A11(B10 - B00)
    # A11: idx 3
    # B10: idx 2 (j=1,k=0), B00: idx 0 (j=0,k=0)
    # C00: idx 0, C10: idx 1
    # C00 = ... + M4 ...
    # C10 = ... + M4 ...
    set_term([(3,1)], [(2,1), (0,-1)], [(0,1), (1,1)])
    
    # M5 = (A00 + A01)B11
    # A00: idx 0, A01: idx 2
    # B11: idx 3
    # C01: idx 2, C00: idx 0 (with -1)
    # C01 = ... + M5 ...
    # C00 = ... - M5 ...
    set_term([(0,1), (2,1)], [(3,1)], [(2,1), (0,-1)])
    
    # M6 = (A10 - A00)(B00 + B01)
    # A10: idx 1, A00: idx 0
    # B00: idx 0, B01: idx 1
    # C11: idx 3
    # C11 = ... + M6 ...
    set_term([(1,1), (0,-1)], [(0,1), (1,1)], [(3,1)])
    
    # M7 = (A01 - A11)(B10 + B11)
    # A01: idx 2, A11: idx 3
    # B10: idx 2, B11: idx 3
    # C00: idx 0, C11: idx 3 (with -1? No, C11 = M1 - M2 + M3 + M6. M7 is in C00?)
    # Let's check Strassen formulas again.
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # So M7 contributes +1 to C00.
    # C00 idx 0.
    set_term([(2,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
    
    # Trivial for k=2..6
    # For each k in 2..6:
    # j=0, i=0: A00 B0k -> Ck0
    # j=0, i=1: A01 B0k -> Ck1
    # j=1, i=0: A10 B1k -> Ck0
    # j=1, i=1: A11 B1k -> Ck1
    
    for k in range(2, p):
        # A00 (idx 0), B0k (idx 0*p + k = k), Ck0 (idx k*n + 0 = k*2)
        set_term([(0,1)], [(k,1)], [(k*2, 1)])
        
        # A01 (idx 1? No, A01 is i=0,j=1 -> idx 0*2+1=1), B0k (idx k), Ck1 (idx k*2+1)
        set_term([(1,1)], [(k,1)], [(k*2+1, 1)])
        
        # A10 (idx 2), B1k (idx 1*p + k = 7+k), Ck0 (idx k*2)
        set_term([(2,1)], [(7+k, 1)], [(k*2, 1)])
        
        # A11 (idx 3), B1k (idx 7+k), Ck1 (idx k*2+1)
        set_term([(3,1)], [(7+k, 1)], [(k*2+1, 1)])
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
