"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U_arr[:, r].reshape(n, m)
        v_r = V_arr[:, r].reshape(m, p)
        w_r = W_arr[:, r].reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_computed[i, k] += u_r[i, j] * v_r[j, k] * w_r[k, i] * A[i, j] * B[j, k]
    
    if not np.allclose(C_expected, C_computed, atol=1e-10):
        return False
    return True

def _build_rank33_decomposition():
    """Build the known rank-33 decomposition for 2x3 x 3x6."""
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    return U, V, W

def _try_rank22_decomposition():
    """
    Attempt to construct a rank-22 decomposition using two copies of a rank-11
    decomposition for 2x3 x 3x3.
    
    The idea is to split the 6 columns of B into two groups of 3:
    B = [B1 | B2] where B1, B2 are 3x3.
    Then C = A*B = A*B1 + A*B2.
    If we have a rank-11 decomposition for 2x3 x 3x3, we can use it for both
    A*B1 and A*B2, giving total rank 22.
    
    However, we need to be careful: the decomposition must be exact.
    We'll try to construct the rank-11 terms for 2x3 x 3x3.
    """
    n, m, p = 2, 3, 6
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We need a rank-11 decomposition for 2x3 x 3x3.
    # Let's try to derive or find one.
    # For 2x3 x 3x3, the tensor is T[i,j,k] = delta_{j,k} for j,k in {0,1,2}.
    
    # Known: rank of 2x2x2 is 7 (Strassen).
    # For 2x3x3, we can view it as 2x2x2 (for j,k in {0,1}) plus extra terms for k=2.
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0] + A[i,2]B[2,0]
    # C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1] + A[i,2]B[2,1]
    # C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2] + A[i,2]B[2,2]
    
    # This is complex. Let's try a different approach: use the fact that
    # the current rank-33 solution uses 3 blocks of 2 columns each, with 11 terms per block.
    # If we can merge two blocks (4 columns) into a single 2x3 x 3x4 block with rank 11,
    # then total rank would be 11 + 11 + 11 = 33 (no improvement) or 11 + 11 = 22 if we
    # can handle 6 columns with two blocks of 3 columns each.
    
    # Let's try to find a rank-11 decomposition for 2x3 x 3x3.
    # I'll attempt to construct one by solving the tensor equation.
    
    # Actually, let's just try to verify if a rank-22 solution exists by checking
    # if we can reuse the rank-11 terms from the 2x3x2 case but adapted for 2x3x3.
    
    # For now, return None to indicate failure, and we'll fall back to rank 33.
    # In a future iteration, we might find the actual rank-11 terms.
    
    # Let's try one specific candidate based on known results.
    # The rank of 2x3x3 is indeed 11. Here's a candidate set of terms:
    
    terms_2x3x3 = [
        # Term 1: (A00+A01)(B00+B11) - A01B11 - A00B00 + ... 
        # This is getting too complex to guess. Let's return None.
    ]
    
    # Since I don't have the exact terms, return None.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-22 solution
    result_22 = _try_rank22_decomposition()
    if result_22 is not None:
        U, V, W = result_22
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 33
    U, V, W = _build_rank33_decomposition()
    if not _verify_decomposition(U, V, W, n, m, p):
        # This should not happen, but just in case
        raise ValueError("Rank 33 decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
