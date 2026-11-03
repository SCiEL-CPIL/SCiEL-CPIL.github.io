import time

DIMENSIONS = [2, 4, 6]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
                i = u_idx // m
                j = u_idx % m
                j2 = v_idx // p
                k = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if i == i2 and j == j2 and k == k2:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    blocks = [
        ( [0, 1], [0, 1], [0, 1], [0, 1] ),
        ( [0, 1], [0, 1], [0, 1], [2, 3] ),
        ( [0, 1], [0, 1], [0, 1], [4, 5] ),
        ( [0, 1], [2, 3], [2, 3], [0, 1] ),
        ( [0, 1], [2, 3], [2, 3], [2, 3] ),
        ( [0, 1], [2, 3], [2, 3], [4, 5] ),
    ]
    
    strassen = [
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)] ),
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ),
        ( [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)] ),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx][col] += coeff
            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx][col] += coeff
            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx][col] += coeff
            col += 1
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition")
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
