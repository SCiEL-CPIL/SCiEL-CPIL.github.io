import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    # Use a safe deadline to avoid timeout penalties
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    rank = 50
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        """
        Adds a 2x2x2 Strassen block.
        """
        nonlocal col
        if col >= rank:
            return False
            
        # A indices: i*m + j
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        # B indices: j*p + k
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        # C indices: k*n + i (Trace convention W[k*n+i, r])
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U[a, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        V[h, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10 += 1, C11 -= 1
        U[c, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1
        W[c11, col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01 += 1, C11 += 1
        U[a, col] += 1
        V[f, col] += 1
        V[h, col] -= 1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00 += 1, C10 += 1
        U[d, col] += 1
        V[g, col] += 1
        V[e, col] -= 1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M5 = (a+b)h -> C00 -= 1, C01 += 1
        U[a, col] += 1
        U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1
        W[c01, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11 += 1
        U[c, col] += 1
        U[a, col] -= 1
        V[e, col] += 1
        V[f, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00 += 1
        U[b, col] += 1
        U[d, col] -= 1
        V[g, col] += 1
        V[h, col] += 1
        W[c00, col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        """
        Adds a 2x2x1 schoolbook block.
        """
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a, col] += 1
        V[e, col] += 1
        W[c00, col] += 1
        col += 1
        
        # P2 = b*g
        U[b, col] += 1
        V[g, col] += 1
        W[c00, col] += 1
        col += 1
        
        # P3 = c*e
        U[c, col] += 1
        V[e, col] += 1
        W[c10, col] += 1
        col += 1
        
        # P4 = d*g
        U[d, col] += 1
        V[g, col] += 1
        W[c10, col] += 1
        col += 1
        
        return True

    # Structure:
    # A is 2x4. Split into A_left (cols 0,1) and A_right (cols 2,3).
    # B is 4x7. Split into B_top (rows 0,1) and B_bot (rows 2,3).
    # C = A_left @ B_top + A_right @ B_bot
    
    # 1. A_left (i0=0, j0=0) @ B_top (j0=0)
    # B_top cols 0-5 split into 3 blocks of 2. Col 6 is single.
    for k0 in [0, 2, 4]:
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        if not add_strassen_222(0, 0, k0):
            break
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    if not add_schoolbook_221(0, 0, 6):
        pass

    # 2. A_right (i0=0, j0=2) @ B_bot (j0=2)
    # B_bot cols 0-5 split into 3 blocks of 2. Col 6 is single.
    for k0 in [0, 2, 4]:
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        if not add_strassen_222(0, 2, k0):
            break
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    if not add_schoolbook_221(0, 2, 6):
        pass

    # The construction is deterministic and known to be correct.
    # We return the lists as required.
    return {"U": U, "V": V, "W": W}
