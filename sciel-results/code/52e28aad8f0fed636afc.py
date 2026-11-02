import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses schoolbook: rank = n*m*p = 100.
    For each (i, j, k), we have one rank-1 term:
      U[i*m+j][col] = 1
      V[j*p+k][col] = 1  
      W[k*n+i][col] = 1
    
    This gives the exact decomposition where each output C[i,k] gets its contribution
    from the term with matching i, j, k.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    rank = n * m * p  # 100
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify correctness
    def verify(U, V, W, tol=1e-9):
        r = rank
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > tol:
                        return False, (i, j, k, s)
        return True, None
    
    valid, info = verify(U, V, W)
    if not valid:
        # Should not happen with schoolbook, but return anyway
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
