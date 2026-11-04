"""Matrix multiplication 2x4x4 with Strassen-based rank-28 decomposition."""
import time

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    Each pattern is a dict with keys 'a', 'b', 'c' mapping local indices to coefficients.
    Local indices are 0..3 for 2x2 matrices (row-major).
    """
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

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    The output factor W follows the official trace convention W[k*n+i,r].
    """
    t0 = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2()
    
    # We decompose the 2x4 * 4x4 product into four 2x2 * 2x2 block products.
    # A is 2x4. B is 4x4.
    # Partition A into A1 (cols 0,1) and A2 (cols 2,3).
    # Partition B into B11 (rows 0,1, cols 0,1), B12 (rows 0,1, cols 2,3),
    #             B21 (rows 2,3, cols 0,1), B22 (rows 2,3, cols 2,3).
    # C1 = A1 B11 + A2 B21 (2x2 result)
    # C2 = A1 B12 + A2 B22 (2x2 result)
    # This is not quite right.
    # C is 2x4.
    # C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] B[0:2, 2:4] + A[:, 2:4] B[2:4, 2:4]
    
    # The four block products are:
    # 1. A1 * B11 -> contributes to C1
    # 2. A2 * B21 -> contributes to C1
    # 3. A1 * B12 -> contributes to C2
    # 4. A2 * B22 -> contributes to C2
    
    # We can use Strassen for each 2x2 multiplication.
    # However, Strassen computes P1..P7 and then combines them.
    # Here we just need to add the contributions to U, V, W.
    
    # Block definitions:
    # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
    # A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # B is 4x4. Rows 0,1,2,3. Cols 0,1,2,3.
    # C is 2x4. Rows 0,1. Cols 0,1,2,3.
    
    blocks = [
        # (a_r, a_c, b_r, b_c, c_r, c_c)
        # A1 (rows 0-1, cols 0-1) * B11 (rows 0-1, cols 0-1) -> C1 (rows 0-1, cols 0-1)
        (0, 0, 0, 0, 0, 0),
        # A2 (rows 0-1, cols 2-3) * B21 (rows 2-3, cols 0-1) -> C1 (rows 0-1, cols 0-1)
        (0, 2, 2, 0, 0, 0),
        # A1 (rows 0-1, cols 0-1) * B12 (rows 0-1, cols 2-3) -> C2 (rows 0-1, cols 2-3)
        (0, 0, 0, 2, 0, 2),
        # A2 (rows 0-1, cols 2-3) * B22 (rows 2-3, cols 2-3) -> C2 (rows 0-1, cols 2-3)
        (0, 2, 2, 2, 0, 2),
    ]
    
    col = 0
    for (a_r, a_c, b_r, b_c, c_r, c_c) in blocks:
        for pattern in strassen_patterns:
            # A part
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = a_r + local_r
                global_j = a_c + local_c
                # U index: i * m + j
                U[global_i * m + global_j][col] += coeff
            
            # B part
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_r + local_r
                global_k = b_c + local_c
                # V index: j * p + k
                V[global_j * p + global_k][col] += coeff
            
            # C part
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = c_r + local_r
                global_k = c_c + local_c
                # W index: k * n + i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
            if time.time() - t0 > deadline - 5:
                break
        if time.time() - t0 > deadline - 5:
            break
    
    return {"U": U, "V": V, "W": W}
