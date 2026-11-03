"""Matrix multiplication 2x4x6 rank minimization via structured Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reproduces T."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    T_check = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_check, T, atol=atol)

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    """
    return [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]

def _construct_strassen_2x2x4(n, m, p):
    """
    Construct a rank-42 decomposition using Strassen on 2x2x2 blocks.
    """
    strassen_terms = _strassen_2x2x2_terms()
    
    rank = 6 * 7  # 42
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    
    col = 0
    for a_block in range(2):  # 0 or 1
        for b_block in range(3):  # 0, 1, or 2
            for term in strassen_terms:
                for u_idx, u_coeff in term['u']:
                    local_i = u_idx // 2
                    local_j = u_idx % 2
                    global_i = local_i
                    global_j = a_block * 2 + local_j
                    U[global_i * m + global_j, col] = u_coeff
                
                for v_idx, v_coeff in term['v']:
                    local_j = v_idx // 2
                    local_k = v_idx % 2
                    row_base = a_block * 2
                    global_row = row_base + local_j
                    global_col = b_block * 2 + local_k
                    V[global_row * p + global_col, col] = v_coeff
                
                for w_idx, w_coeff in term['w']:
                    local_i = w_idx // 2
                    local_k = w_idx % 2
                    global_i = local_i
                    global_k = b_block * 2 + local_k
                    W[global_k * n + global_i, col] = w_coeff
                
                col += 1
    
    return U, V, W, rank

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks as the primary approach.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor for verification
    T = _build_tensor(n, m, p)

    # Try Strassen rank-42 decomposition
    U, V, W, rank = _construct_strassen_2x2x4(n, m, p)
    
    # Verify the Strassen decomposition
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)

    # Final verification
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
