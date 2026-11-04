"""Matrix multiplication 2x4x4 with optimized tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
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


def _build_strassen_decomposition(n=2, m=4, p=4):
    """
    Build a rank-28 Strassen-based decomposition for 2x4 * 4x4 matrix multiplication.
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2()
    
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_i = local_a_idx // 2
                local_j = local_a_idx % 2
                global_i = local_i
                global_j = a_co + local_j
                U[global_i * m + global_j][col] = coeff
            
            for local_b_idx, coeff in pattern['b'].items():
                local_j = local_b_idx // 2
                local_k = local_b_idx % 2
                global_j = b_ro + local_j
                global_k = b_co + local_k
                V[global_j * p + global_k][col] = coeff
            
            for local_c_idx, coeff in pattern['c'].items():
                local_i = local_c_idx // 2
                local_k = local_c_idx % 2
                global_i = local_i
                global_k = c_co + local_k
                W[global_k * n + global_i][col] = coeff
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    """
    Verify that the decomposition correctly represents matrix multiplication.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    np.random.seed(42)
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            ua = np.dot(U_arr[:, r], A.flatten())
            vb = np.dot(V_arr[:, r], B.flatten())
            prod = ua * vb
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W_arr[k * n + i, r] * prod
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True


def construct():
    """
    Construct a low-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    t0 = time.time()
    deadline = 115.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: trivial rank-32 decomposition
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
