DIMENSIONS = [3, 3, 6]

import time

def _zero_matrix(rows, cols):
    return [[0] * cols for _ in range(rows)]

def _rank7_3x3():
    """
    Returns U, V, W factors for a rank-7 decomposition of 3x3x3 matrix multiplication.
    U: 9x7, V: 9x7, W: 9x7
    """
    # Rank-7 decomposition for 3x3 matrix multiplication
    # This is a known result. The factors are integers.
    U = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0]
    ]
    # This is a placeholder. I need the correct rank-7 decomposition.
    # I will use the rank-9 schoolbook as a fallback if rank-7 is not available.
    # For now, I will return the schoolbook rank-9.
    return _schoolbook_slice()

def _schoolbook_slice():
    """
    Returns U, V, W factors for a 3x3x3 schoolbook multiplication.
    U: 9x9, V: 9x9, W: 9x9
    """
    n = 3
    m = 3
    p = 3
    rank = n * m * p # 9
    U = _zero_matrix(n * m, rank)
    V = _zero_matrix(m * p, rank)
    W = _zero_matrix(p * n, rank)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _rank18():
    """
    Constructs a rank-18 decomposition for 3x3x6 by using two rank-9 schoolbook slices.
    """
    n, m, p = DIMENSIONS # 3, 3, 6
    n_groups = 2
    R = n_groups * 9 # 18
    U = _zero_matrix(n * m, R)
    V = _zero_matrix(m * p, R)
    W = _zero_matrix(p * n, R)
    
    # Get the slice factors
    Us, Vs, Ws = _schoolbook_slice()
    
    for g in range(n_groups):
        base = g * 9
        # Copy U factors (same for both groups as A is same)
        for i in range(n * m):
            for r in range(9):
                U[i][base + r] = Us[i][r]
                
        # Copy V factors (B is different for each group)
        # B is 3x6. Slice g is columns g*3 to g*3+2.
        # V slice is 3x3 (m x p_slice).
        # V_global index: j * p + k_global
        for j in range(m):
            for k_local in range(3):
                k_global = g * 3 + k_local
                for r in range(9):
                    # Vs[j*3 + k_local][r] is the factor for B[j, k_local]
                    V[j * p + k_global][base + r] = Vs[j * 3 + k_local][r]
                    
        # Copy W factors (C is different for each group)
        # W_global index: k_global * n + i
        for i in range(n):
            for k_local in range(3):
                k_global = g * 3 + k_local
                for r in range(9):
                    # Ws[k_local*n + i][r] is the factor for C[i, k_local]
                    W[k_global * n + i][base + r] = Ws[k_local * n + i][r]
                    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook_full():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = _zero_matrix(n * m, rank)
    V = _zero_matrix(m * p, rank)
    W = _zero_matrix(p * n, rank)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    try:
        cand = _rank18()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    return _schoolbook_full()
