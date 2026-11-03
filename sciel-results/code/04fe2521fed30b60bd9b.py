import time

DIMENSIONS = [2, 4, 6]
_DEADLINE = time.time() + 140

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
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
    
    # A11*B11: A rows 0-1, cols 0-1; B rows 0-1, cols 0-5
    # Split B cols into 3 groups of 2
    for b_col_offset in [0, 2, 4]:
        a_map = {
            0: 0 * m + 0,
            1: 0 * m + 1,
            2: 1 * m + 0,
            3: 1 * m + 1
        }
        b_map = {
            0: 0 * p + b_col_offset,
            1: 0 * p + b_col_offset + 1,
            2: 1 * p + b_col_offset,
            3: 1 * p + b_col_offset + 1
        }
        w_map = {
            0: b_col_offset * n + 0,
            1: (b_col_offset+1) * n + 0,
            2: b_col_offset * n + 1,
            3: (b_col_offset+1) * n + 1
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx]][col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx]][col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx]][col] += coeff
            col += 1
    
    # A12*B12: j=2, rank 1
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # A2*B_bottom: j=3, rank 1
    for i in range(n):
        U[i * m + 3][col] = 1
    for k in range(p):
        V[3 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition")
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
