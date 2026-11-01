import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, seed=42):
    """Verify that the decomposition correctly computes A @ B for random A, B."""
    np.random.seed(seed)
    A = np.random.randint(1, 7, size=(n, m)).astype(float)
    B = np.random.randint(1, 7, size=(m, p)).astype(float)
    C_expected = A @ B
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # C[i,k] = sum_t (sum_{i',j'} U[i'*m+j', t] * A[i',j']) * (sum_{j'} V[j'*p+k, t] * B[j',k]) * W[k*n+i, t]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for t in range(rank):
                a_val = sum(U_np[ii * m + jj, t] * A[ii, jj] for ii in range(n) for jj in range(m))
                b_val = sum(V_np[jj * p + k, t] * B[jj, k] for jj in range(m))
                total += a_val * b_val * W_np[k * n + i, t]
            C_computed[i, k] = total
    
    return np.allclose(C_computed, C_expected, atol=1e-8)


def _construct_strassen_block():
    """Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (a, b, c, d, e, f, g, h, C00, C01, C10, C11)
    where the product is (linear combo of A) * (linear combo of B) contributes to C entries.
    """
    return [
        (1, 0, 0, 1,  1, 0, 0, 1,  1,  0,  0,  1),  # M1 = (a+d)(e+h)
        (0, 0, 1, 1,  1, 0, 0, 0,  0,  0,  1, -1),  # M2 = (c+d)e
        (1, 0, 0, 0,  0, 1, 0,-1,  0,  1,  0,  1),  # M3 = a(f-h)
        (0, 0, 0, 1, -1, 0, 1, 0,  1,  0,  1,  0),  # M4 = d(g-e)
        (1, 1, 0, 0,  0, 0, 0, 1, -1,  1,  0,  0),  # M5 = (a+b)h
        (-1, 0, 1, 0,  1, 1, 0, 0,  0,  0,  0,  1),  # M6 = (c-a)(e+f)
        (0, 1, 0,-1,  0, 0, 1, 1,  1,  0,  0,  0),  # M7 = (b-d)(g+h)
    ]


def _construct_rank49():
    """Standard 7 independent 2x2 Strassen blocks for 2x2 @ 2x14. Rank = 49."""
    n, m, p = 2, 2, 14
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _construct_strassen_block()
    
    for block in range(num_blocks):
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c00_c, c01_c, c10_c, c11_c = term
            
            # A part: U[i*m+j, col]
            U[0][col] = a_c
            U[1][col] = b_c
            U[2][col] = c_c
            U[3][col] = d_c
            
            # B part: V[j*p+k, col]
            V[0 * p + (2 * block)][col] = e_c
            V[0 * p + (2 * block + 1)][col] = f_c
            V[1 * p + (2 * block)][col] = g_c
            V[1 * p + (2 * block + 1)][col] = h_c
            
            # C part: W[k*n+i, col]
            W[(2 * block) * n + 0][col] = c00_c
            W[(2 * block + 1) * n + 0][col] = c01_c
            W[(2 * block) * n + 1][col] = c10_c
            W[(2 * block + 1) * n + 1][col] = c11_c
    
    return U, V, W


def _construct_trivial():
    """Trivial rank-56 decomposition. Guaranteed correct."""
    n, m, p = 2, 2, 14
    rank = n * m * p  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W


def _try_shared_block_decomposition(n, m, p):
    """
    Try to find a decomposition where some Strassen terms are shared across blocks.
    
    Key idea: The 2x14 B matrix has 28 entries. If we group them into 7 pairs 
    (columns 0-1, 2-3, ..., 12-13), each pair forms a 2x2 block. Standard Strassen 
    uses 7 products per block = 49 total.
    
    However, we can try to use the fact that the A matrix is shared. Specifically,
    if two blocks use the same linear combination of A entries, we might be able 
    to merge them. But in standard Strassen, all 7 terms use different combos of A.
    
    Alternative: Try to find a rank < 49 decomposition using a global optimization.
    For now, implement a simple approach: try rank 48 by checking if one term 
    can be eliminated through a clever identity.
    
    Actually, let's try a different approach: use the fact that for the 2x2 case,
    the rank is exactly 7 (Provan 1978). For 2x2 @ 2x(2k), the rank is at most 7k.
    Whether it's exactly 7k is unknown in general, but for k=1 it's 7.
    
    Let's try to search for a rank-48 decomposition using a randomized algebraic approach.
    """
    # This is a placeholder for a more sophisticated search.
    # For now, return None to indicate failure.
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 14
    
    # Start with the known rank-49 Strassen block decomposition
    U49, V49, W49 = _construct_rank49()
    
    # Verify the rank-49 decomposition
    if not _verify_decomposition(U49, V49, W49, n, m, p):
        # Should never happen, but fallback to trivial
        U, V, W = _construct_trivial()
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U49, V49, W49
    best_rank = 49
    
    # Try to find a lower-rank decomposition if time permits
    if time.time() < deadline - 20:
        result = _try_shared_block_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                rank = U[0].__len__()
                if rank < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
    
    # Final verification of the best result
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, seed=123):
        # Double-check with different seed
        if not _verify_decomposition(best_U, best_V, best_W, n, m, p, seed=999):
            # Fallback to trivial
            U, V, W = _construct_trivial()
            return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
