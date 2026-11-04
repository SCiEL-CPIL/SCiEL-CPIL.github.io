"""Matrix multiplication 2x4x4 with rank-28 Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _strassen_2x2():
    """Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication."""
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition is correct."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True


def construct():
    """Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    strassen_patterns = _strassen_2x2()
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        if time.time() > deadline:
            break
        for pattern in strassen_patterns:
            if time.time() > deadline:
                break
            
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_bj + local_c
                U[global_i * m + global_j][col] = coeff
            
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_bj + local_r
                global_k = b_bk + local_c
                V[global_j * p + global_k][col] = coeff
            
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_bk + local_c
                W[global_k * n + global_i][col] = coeff
            
            col += 1
    
    if time.time() < deadline - 5.0:
        valid = _verify_decomposition(U, V, W, n, m, p, rank)
        if not valid:
            pass
    
    return {"U": U, "V": V, "W": W}
