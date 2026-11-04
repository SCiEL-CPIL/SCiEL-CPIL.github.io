"""Matrix multiplication 2x4x4 with optimized tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]


def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    """
    patterns = [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
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


def _verify_decomposition(U, V, W, n=2, m=4, p=4, seed=42, num_tests=10):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(seed)
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            a_factor = np.dot(U_arr[:, r], A.flatten())
            b_factor = np.dot(V_arr[:, r], B.flatten())
            if a_factor != 0 and b_factor != 0:
                C_computed += W_arr[:, r].reshape(p, n).T * (a_factor * b_factor)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True


def _compute_slice_ranks(U, V, W, n=2, m=4, p=4):
    """
    Compute the slice ranks of the tensor to get lower bounds on the rank.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor T with shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    mode1_ranks = [np.linalg.matrix_rank(T[i, :, :]) for i in range(n * m)]
    mode1_max_rank = max(mode1_ranks)
    
    mode2_ranks = [np.linalg.matrix_rank(T[:, j, :]) for j in range(m * p)]
    mode2_max_rank = max(mode2_ranks)
    
    mode3_ranks = [np.linalg.matrix_rank(T[:, :, k]) for k in range(p * n)]
    mode3_max_rank = max(mode3_ranks)
    
    lower_bound = max(mode1_max_rank, mode2_max_rank, mode3_max_rank)
    
    return lower_bound, mode1_max_rank, mode2_max_rank, mode3_max_rank





def construct():
    """
    Construct a low-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Build the rank-28 Strassen decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    return {"U": U, "V": V, "W": W}
