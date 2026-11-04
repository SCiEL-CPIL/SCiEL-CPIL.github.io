import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Try to find a low rank decomposition for 2x3x9
    # Target rank 27 based on strategy
    best_result = search_low_rank(n, m, p, 27, deadline)
    
    if best_result is not None:
        return best_result

    # Fallback to schoolbook 54
    return construct_schoolbook(n, m, p)

def search_low_rank(n, m, p, target_rank, deadline):
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def calc_error(U, V, W):
        # U: (num_u, rank), V: (num_v, rank), W: (num_w, rank)
        # Res = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # We compute Res - T and sum squares
        # Efficient computation:
        # Res[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute this using einsum
        Res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = Res - T
        return np.sum(diff**2)

    random.seed(42)
    np.random.seed(42)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with random integers in {-1, 0, 1}
    # This is a good start for tensor rank problems
    U = np.random.randint(-1, 2, size=(num_u, target_rank)).astype(float)
    V = np.random.randint(-1, 2, size=(num_v, target_rank)).astype(float)
    W = np.random.randint(-1, 2, size=(num_w, target_rank)).astype(float)
    
    current_err = calc_error(U, V, W)
    
    if current_err < best_err:
        best_err = current_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
    # Simple Local Search / Hill Climbing
    # Since rank 27 is low, the solution space is small.
    # We try to change one element at a time by +/- 0.5 or +/- 1.
    
    deltas = [0.5, -0.5, 1.0, -1.0]
    
    # Time budget for search
    # We will run until deadline - 5s
    while time.time() < deadline - 5:
        # Pick a random matrix, row, col
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            mat = U
            rows = num_u
        elif mat_idx == 1:
            mat = V
            rows = num_v
        else:
            mat = W
            rows = num_w
            
        row = random.randint(0, rows - 1)
        col = random.randint(0, target_rank - 1)
        
        old_val = mat[row, col]
        delta = random.choice(deltas)
        new_val = old_val + delta
        
        mat[row, col] = new_val
        new_err = calc_error(U, V, W)
        
        if new_err < current_err:
            current_err = new_err
            if current_err < best_err:
                best_err = current_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_err < 1e-9:
                    # Snap to half-integers
                    bU = np.round(best_U * 2) / 2
                    bV = np.round(best_V * 2) / 2
                    bW = np.round(best_W * 2) / 2
                    snap_err = calc_error(bU, bV, bW)
                    if snap_err < 1e-9:
                        return {
                            "U": bU.tolist(),
                            "V": bV.tolist(),
                            "W": bW.tolist()
                        }
        else:
            mat[row, col] = old_val
            
    # If we found a good solution, try to snap
    if best_err < 1e-6:
        bU = np.round(best_U * 2) / 2
        bV = np.round(best_V * 2) / 2
        bW = np.round(best_W * 2) / 2
        snap_err = calc_error(bU, bV, bW)
        if snap_err < 1e-9:
            return {
                "U": bU.tolist(),
                "V": bV.tolist(),
                "W": bW.tolist()
            }

    return None

def construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
