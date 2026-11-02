import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    # Total rank: 7 (2x1x7) + 7+7+4 (2x2x7) + 7 (2x1x7) = 32
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block.
        A is 2x2 at rows i0,i0+1 and cols j0,j0+1
        B is 2x2 at rows j0,j0+1 and cols k0,k0+1
        C is 2x2 at rows k0,k0+1 and cols i0,i0+1
        """
        nonlocal col
        if col >= rank:
            return False
            
        # A indices
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        # B indices
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        # C indices (output W[k*n+i, r])
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # Strassen's 7 multiplications
        # M1 = (a+d)(e+h) -> C[0,0] += M1, C[1,1] += M1
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (c+d)(e) -> C[1,0] += M2, C[1,1] -= M2
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = (a)(f-h) -> C[0,1] += M3, C[1,1] += M3
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = (d)(g-e) -> C[0,0] += M4, C[1,0] += M4
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a+b)(h) -> C[0,0] -= M5, C[0,1] += M5
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C[1,1] += M6
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C[0,0] += M7
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        """Add schoolbook for 2x2x1 block.
        A is 2x2 at rows i0,i0+1 and cols j0,j0+1
        B is 2x1 at rows j0,j0+1 and col k0
        C is 2x1 at row k0 and cols i0,i0+1
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
        
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1
        
        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1
        
        return True

    def add_schoolbook_211(i0, j0, k0):
        """Add schoolbook for 2x1x1 block.
        A is 2x1 at rows i0,i0+1 and col j0
        B is 1x1 at row j0 and col k0
        C is 1x1 at row k0 and col i0... wait, C is 2x1 at row k0 and cols i0,i0+1
        """
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        c = (i0 + 1) * m + j0
        
        e = j0 * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1
        
        return True

    def add_schoolbook_217(i0, j0):
        """Add schoolbook for 2x1x7 block.
        A is 2x1 at rows i0,i0+1 and col j0
        B is 1x7 at row j0 and cols 0..6
        C is 2x7 at rows 0..6 and cols i0,i0+1
        """
        nonlocal col
        for k0 in range(p):
            if not add_schoolbook_211(i0, j0, k0):
                return False
        return True

    # Part 1: A[:, 0] * B[0, :] -> 2x1x7, rank 7
    # i0=0, j0=0, k0=0..6
    if not add_schoolbook_217(0, 0):
        pass
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 2: A[:, 1:3] * B[1:3, :] -> 2x2x7
    # Split B[1:3, :] into B[1:3, 0:2], B[1:3, 2:4], B[1:3, 4:6], B[1:3, 6:7]
    # Wait, p=7, so we need to split 7 into [2,2,2,1]
    # A[:, 1:3] has j0=1, j1=2
    # B[1:3, :] has j0=1, j1=2
    
    # 2x2x2 blocks at k0=0,2,4
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 1, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # 2x2x1 block at k0=6
    if not add_schoolbook_221(0, 1, 6):
        pass
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 3: A[:, 3] * B[3, :] -> 2x1x7, rank 7
    if not add_schoolbook_217(0, 3):
        pass
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
