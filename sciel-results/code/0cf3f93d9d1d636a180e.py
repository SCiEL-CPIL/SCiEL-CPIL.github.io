import time

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140.0

def _time_ok():
    return time.time() < _DEADLINE

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions: (a_rows, a_cols, b_rows, b_cols)
    # A is 2x4, B is 4x6, C is 2x6
    # U index: i*m + j (i in a_rows, j in a_cols)
    # V index: j*p + k (j in b_rows, k in b_cols)
    # W index: k*n + i (k in b_cols, i in a_rows)
    
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    # Strassen 7 components for 2x2x2
    # Local indices: A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #               B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #               C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen = [
        # M1 = (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00(B10-B11)
        ([(0, 1)], [(2, 1), (3, -1)], [(1, -1), (3, 1)]),
        # M4 = A11(B00-B01)
        ([(3, 1)], [(0, 1), (1, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00+A01)B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
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
                U[a_map[local_idx]][col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx]][col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx]][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
