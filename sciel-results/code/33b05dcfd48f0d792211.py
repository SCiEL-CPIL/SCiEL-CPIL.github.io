import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Partitions
    # n=3: [2, 1]
    # m=5: [2, 2, 1]
    # p=7: [2, 2, 2, 1]
    part_n = [2, 1]
    part_m = [2, 2, 1]
    part_p = [2, 2, 2, 1]
    
    # Calculate offsets
    off_n = [0]
    for sz in part_n[:-1]:
        off_n.append(off_n[-1] + sz)
        
    off_m = [0]
    for sz in part_m[:-1]:
        off_m.append(off_m[-1] + sz)
        
    off_p = [0]
    for sz in part_p[:-1]:
        off_p.append(off_p[-1] + sz)
        
    # Pre-calculate total rank to allocate arrays
    # We can just use lists and append, or pre-allocate if we know rank.
    # Rank is 99.
    # Let's just use lists and append to be safe, then convert.
    # But for speed, pre-allocation is better.
    # Max rank is 105. 99 is close.
    
    # Let's count rank first
    rank = 0
    for i, n_b in enumerate(part_n):
        for j, m_b in enumerate(part_m):
            for k, p_b in enumerate(part_p):
                if n_b == 2 and m_b == 2 and p_b == 2:
                    rank += 7
                else:
                    rank += n_b * m_b * p_b
                    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for i, n_b in enumerate(part_n):
        i0 = off_n[i]
        for j, m_b in enumerate(part_m):
            j0 = off_m[j]
            for k, p_b in enumerate(part_p):
                k0 = off_p[k]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Strassen
                    # M1 = (A00 + A11)(B00 + B11)
                    U[i0 * m + j0][col] = 1
                    U[(i0+1) * m + (j0+1)][col] = 1
                    V[j0 * p + k0][col] = 1
                    V[(j0+1) * p + (k0+1)][col] = 1
                    W[k0 * n + i0][col] = 1
                    W[(k0+1) * n + (i0+1)][col] = 1
                    col += 1
                    
                    # M2 = (A10 + A11)B00
                    U[i0 * m + (j0+1)][col] = 1 # A10 is row i0+1, col j0? No.
                    # A is n x m. Row i, Col j.
                    # A10: Row i0+1, Col j0.
                    # A11: Row i0+1, Col j0+1.
                    # So:
                    U[i0 * m + (j0+1)][col] = 0 # Correcting:
                    # A10 is at (i0+1, j0). Index: (i0+1)*m + j0.
                    # A11 is at (i0+1, j0+1). Index: (i0+1)*m + (j0+1).
                    # B00 is at (j0, k0). Index: j0*p + k0.
                    # C10 is at (i0+1, k0). Index: k0*n + (i0+1).
                    # C11 is at (i0+1, k0+1). Index: (k0+1)*n + (i0+1).
                    
                    # Let's restart the Strassen mapping carefully.
                    # Local A: 2x2. Rows 0,1. Cols 0,1.
                    # Local B: 2x2. Rows 0,1. Cols 0,1.
                    # Local C: 2x2. Rows 0,1. Cols 0,1.
                    
                    # Global indices:
                    # A[r, c] -> U[(i0+r)*m + (j0+c)]
                    # B[r, c] -> V[(j0+r)*p + (k0+c)]
                    # C[r, c] -> W[(k0+c)*n + (i0+r)]
                    
                    # M1 = (A00 + A11)(B00 + B11)
                    # U: A00 (0,0), A11 (1,1)
                    # V: B00 (0,0), B11 (1,1)
                    # W: C00 (0,0), C11 (1,1)
                    U[i0 * m + j0][col] = 1
                    U[(i0+1) * m + (j0+1)][col] = 1
                    V[j0 * p + k0][col] = 1
                    V[(j0+1) * p + (k0+1)][col] = 1
                    W[k0 * n + i0][col] = 1
                    W[(k0+1) * n + (i0+1)][col] = 1
                    col += 1
                    
                    # M2 = (A10 + A11)B00
                    # A10 (1,0), A11 (1,1)
                    # B00 (0,0)
                    # C10 (1,0), C11 (1,1) -> Wait, C10 is row 1 col 0. C11 is row 1 col 1.
                    # Formula: C10 += M2, C11 += M2?
                    # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
                    # So M2 contributes to C10 and C11.
                    # C10 index: k0*n + (i0+1).
                    # C11 index: (k0+1)*n + (i0+1).
                    U[(i0+1) * m + j0][col] = 1
                    U[(i0+1) * m + (j0+1)][col] = 1
                    V[j0 * p + k0][col] = 1
                    W[k0 * n + (i0+1)][col] = 1
                    W[(k0+1) * n + (i0+1)][col] = 1
                    col += 1
                    
                    # M3 = A00(B10 - B11)
                    # A00 (0,0)
                    # B10 (1,0), B11 (1,1) -> -1 for B11
                    # C01 (0,1) -> C01 = M3 + M5.
                    # C01 index: (k0+1)*n + i0.
                    U[i0 * m + j0][col] = 1
                    V[(j0+1) * p + k0][col] = 1
                    V[(j0+1) * p + (k0+1)][col] = -1
                    W[(k0+1) * n + i0][col] = 1
                    col += 1
                    
                    # M4 = A11(B01 - B00)
                    # A11 (1,1)
                    # B01 (0,1), B00 (0,0) -> -1 for B00
                    # C00 (0,0) -> C00 = M1 + M4 ...
                    # C10 (1,0) -> C10 = ... + M4 ...
                    # C00 index: k0*n + i0.
                    # C10 index: k0*n + (i0+1).
                    U[(i0+1) * m + (j0+1)][col] = 1
                    V[j0 * p + (k0+1)][col] = 1
                    V[j0 * p + k0][col] = -1
                    W[k0 * n + i0][col] = 1
                    W[k0 * n + (i0+1)][col] = 1
                    col += 1
                    
                    # M5 = (A00 + A10)B11
                    # A00 (0,0), A10 (1,0)
                    # B11 (1,1)
                    # C00 (0,0) -> -M5
                    # C01 (0,1) -> +M5
                    # C00 index: k0*n + i0.
                    # C01 index: (k0+1)*n + i0.
                    U[i0 * m + j0][col] = 1
                    U[(i0+1) * m + j0][col] = 1
                    V[(j0+1) * p + (k0+1)][col] = 1
                    W[k0 * n + i0][col] = -1
                    W[(k0+1) * n + i0][col] = 1
                    col += 1
                    
                    # M6 = (A10 - A00)(B00 + B01)
                    # A10 (1,0), A00 (0,0) -> -1 for A00
                    # B00 (0,0), B01 (0,1)
                    # C11 (1,1) -> +M6
                    # C11 index: (k0+1)*n + (i0+1).
                    U[(i0+1) * m + j0][col] = 1
                    U[i0 * m + j0][col] = -1
                    V[j0 * p + k0][col] = 1
                    V[j0 * p + (k0+1)][col] = 1
                    W[(k0+1) * n + (i0+1)][col] = 1
                    col += 1
                    
                    # M7 = (A01 - A11)(B10 + B11)
                    # A01 (0,1), A11 (1,1) -> -1 for A11
                    # B10 (1,0), B11 (1,1)
                    # C00 (0,0) -> +M7
                    # C00 index: k0*n + i0.
                    U[i0 * m + (j0+1)][col] = 1
                    U[(i0+1) * m + (j0+1)][col] = -1
                    V[(j0+1) * p + k0][col] = 1
                    V[(j0+1) * p + (k0+1)][col] = 1
                    W[k0 * n + i0][col] = 1
                    col += 1
                    
                else:
                    # Schoolbook
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i0+u) * m + (j0+v)][col] = 1
                                V[(j0+v) * p + (k0+w)][col] = 1
                                W[(k0+w) * n + (i0+u)][col] = 1
                                col += 1
                                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
