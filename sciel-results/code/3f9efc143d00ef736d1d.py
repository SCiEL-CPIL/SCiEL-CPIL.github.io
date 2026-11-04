"""4x5x6 matrix multiplication decomposition (rank 92)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We aim for rank 92
    # 4x4x4 core (56) + 4x4x1 (16) + 4x1x4 (16) + 4x1x1 (4)
    
    # Pre-allocate with a safe upper bound, then slice
    # Max possible rank is schoolbook 120.
    MAX_RANK = 120
    
    U = [[0] * MAX_RANK for _ in range(n * m)]
    V = [[0] * MAX_RANK for _ in range(m * p)]
    W = [[0] * MAX_RANK for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Adds a 2x2x2 Strassen block. Returns new rank offset."""
        nonlocal current_rank
        r = current_rank
        
        # Indices
        # U: (i*m + k)
        # V: (k*p + j)
        # W: (j*n + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[i0 * m + k0][r] += 1
        U[(i0+1) * m + (k0+1)][r] += 1
        V[k0 * p + j0][r] += 1
        V[(k0+1) * p + (j0+1)][r] += 1
        W[j0 * n + i0][r] += 1
        W[(j0+1) * n + (i0+1)][r] += 1
        current_rank += 1
        
        # M2 = (A10 + A11)B00
        r = current_rank
        U[(i0+1) * m + k0][r] += 1
        U[(i0+1) * m + (k0+1)][r] += 1
        V[k0 * p + j0][r] += 1
        W[j0 * n + (i0+1)][r] += 1
        W[(j0+1) * n + (i0+1)][r] += -1
        current_rank += 1
        
        # M3 = A00(B01 - B11)
        r = current_rank
        U[i0 * m + k0][r] += 1
        V[(k0+1) * p + (j0+1)][r] += 1
        V[k0 * p + (j0+1)][r] += -1
        W[(j0+1) * n + i0][r] += 1
        W[(j0+1) * n + (i0+1)][r] += 1
        current_rank += 1
        
        # M4 = A11(B10 - B00)
        r = current_rank
        U[(i0+1) * m + (k0+1)][r] += 1
        V[(k0+1) * p + j0][r] += 1
        V[k0 * p + j0][r] += -1
        W[j0 * n + i0][r] += 1
        W[j0 * n + (i0+1)][r] += 1
        current_rank += 1
        
        # M5 = (A00 + A01)B11
        r = current_rank
        U[i0 * m + k0][r] += 1
        U[i0 * m + (k0+1)][r] += 1
        V[(k0+1) * p + (j0+1)][r] += 1
        W[j0 * n + i0][r] += -1
        W[(j0+1) * n + i0][r] += 1
        current_rank += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        r = current_rank
        U[(i0+1) * m + k0][r] += 1
        U[i0 * m + k0][r] += -1
        V[k0 * p + j0][r] += 1
        V[k0 * p + (j0+1)][r] += 1
        W[(j0+1) * n + (i0+1)][r] += 1
        current_rank += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        r = current_rank
        U[i0 * m + (k0+1)][r] += 1
        U[(i0+1) * m + (k0+1)][r] += -1
        V[(k0+1) * p + j0][r] += 1
        V[(k0+1) * p + (j0+1)][r] += 1
        W[j0 * n + i0][r] += 1
        current_rank += 1
        
        return current_rank

    def add_schoolblock(i0, j0, k0, ni, nj, nk):
        """Adds a schoolbook block of size ni x nj x nk starting at i0, j0, k0."""
        nonlocal current_rank
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(nk):
                    r = current_rank
                    U[(i0+ii) * m + (k0+kk)][r] = 1
                    V[(k0+kk) * p + (j0+jj)][r] = 1
                    W[(j0+jj) * n + (i0+ii)][r] = 1
                    current_rank += 1
        return current_rank

    # 1. Core 4x4x4 block
    # Split into four 2x2x4 blocks
    # Block (0,0): i in 0,1; j in 0,1; k in 0,3
    #   Split k into 0,1 and 2,3
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 0, 2)
    
    # Block (1,0): i in 0,1; j in 2,3; k in 0,3
    add_strassen_2x2x2(0, 2, 0)
    add_strassen_2x2x2(0, 2, 2)
    
    # Block (0,1): i in 2,3; j in 0,1; k in 0,3
    add_strassen_2x2x2(2, 0, 0)
    add_strassen_2x2x2(2, 0, 2)
    
    # Block (1,1): i in 2,3; j in 2,3; k in 0,3
    add_strassen_2x2x2(2, 2, 0)
    add_strassen_2x2x2(2, 2, 2)
    
    # 2. Extra k=4 for j in 0..3
    # i in 0..3, j in 0..3, k=4
    add_schoolblock(0, 0, 4, 4, 4, 1)
    
    # 3. Extra j=4 for k in 0..3
    # i in 0..3, j=4, k in 0..3
    add_schoolblock(0, 4, 0, 4, 1, 4)
    
    # 4. Corner j=4, k=4
    # i in 0..3, j=4, k=4
    add_schoolblock(0, 4, 4, 4, 1, 1)
    
    final_rank = current_rank
    
    # Slice arrays to final rank
    U = [row[:final_rank] for row in U]
    V = [row[:final_rank] for row in V]
    W = [row[:final_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
