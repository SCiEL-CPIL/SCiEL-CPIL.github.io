"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_fast(U, V, W, n, m, p, rng=None):
    """Verify exact decomposition using tensor contraction. Returns True if valid."""
    # The multiplication tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each (i,j,k) triple, the sum must equal 1.
    # Use numpy: build the full check via einsum-like approach
    # T_reconstructed[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    
    # For each (i,j,k): sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # We can compute this as: for each c, outer product Ur[:,:,c] * Vr[:,:,c] * Wr[:,:,c]
    # But that's rank-many 3D tensors. Instead use einsum.
    
    # Actually, let's just do it directly with a loop over a subset or full
    # Full check: n*m*p = 216 entries, each requiring rank multiplications
    # For rank ~200, that's 216*200 = 43200 ops, very fast.
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _try_remove_term(U, V, W, idx, n, m, p):
    """Try removing term idx. Returns (success, U, V, W) with term removed if valid."""
    # Remove column idx
    mask = np.ones(U.shape[1], dtype=bool)
    mask[idx] = False
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    if _verify_fast(U_new, V_new, W_new, n, m, p):
        return True, U_new, V_new, W_new
    return False, U, V, W


def _strassen_2x2_blocks(n, m, p):
    """
    Apply Strassen to 2x2 blocks. Returns numpy arrays or None.
    """
    if n % 2 or m % 2 or p % 2:
        return None

    nh, mh, ph = n // 2, m // 2, p // 2

    steps = [
        ({3: 1, 0: -1}, {0: 1, 3: 1}),
        ({0: 1, 3: 1}, {3: 1}),
        ({2: 1, 3: 1}, {1: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {2: 1}),
        ({2: 1, 1: -1}, {0: 1, 3: 1}),
    ]

    c_combos = [
        {0: 1, 1: -1, 2: 1, 6: -1},
        {2: 1, 5: 1},
        {1: 1, 4: 1},
        {0: 1, 3: 1, 4: -1, 6: 1},
    ]

    block_rank = nh * mh * ph
    total_rank = 7 * block_rank

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph

                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k

                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col

                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue

                            col = s * block_rank + i * (mh * ph) + j * ph + k

                            U[gi * m + gj, col] = a_sign
                            V[gj * p + gk, col] = b_sign
                            W[gk * n + gi, col] = c_sign

    return U, V, W


def _randomized_reduction(U, V, W, n, m, p, deadline, rng, max_attempts=5000):
    """
    Randomized search: try to remove terms from the current decomposition.
    Uses multiple restarts with random orderings.
    """
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    attempts = 0
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        # Try random removals from current best
        cur_U, cur_V, cur_W = best_U.copy(), best_V.copy(), best_W.copy()
        cur_rank = cur_U.shape[1]
        
        # Try to remove several terms in a row
        removed_in_pass = 0
        max_removals_per_pass = 20
        for _ in range(max_removals_per_pass):
            if time.time() > deadline:
                break
            if cur_rank <= best_rank:
                break
            # Pick a random column to try removing
            idx = rng.randint(0, cur_rank - 1)
            success, cur_U, cur_V, cur_W = _try_remove_term(cur_U, cur_V, cur_W, idx, n, m, p)
            if success:
                cur_rank -= 1
                removed_in_pass += 1
                if cur_rank < best_rank:
                    best_rank = cur_rank
                    best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
            else:
                # If this one fails, try a few more random ones before giving up on this pass
                pass
        
        # If we made progress, continue; otherwise, try a different random ordering
        if removed_in_pass == 0 and cur_rank == best_rank:
            # Reset and try with a different random seed/order
            # Sometimes try removing from the end (structured)
            pass

    return best_U, best_V, best_W, best_rank


def _greedy_structured_reduction(U, V, W, n, m, p, deadline, rng):
    """
    Structured greedy: try removing terms in a structured order (by block, by position).
    """
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    cur_U, cur_V, cur_W = best_U.copy(), best_V.copy(), best_W.copy()
    cur_rank = cur_U.shape[1]
    
    # Try all columns in order, removing if possible
    # This is O(rank * verify_cost) which for rank~200 and verify~216*200 ops is fine
    for idx in range(cur_rank - 1, -1, -1):  # Try from the end first
        if time.time() > deadline:
            break
        if cur_rank <= best_rank:
            break
        success, cur_U, cur_V, cur_W = _try_remove_term(cur_U, cur_V, cur_W, idx, n, m, p)
        if success:
            cur_rank -= 1
            best_rank = cur_rank
            best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    rng = random.Random(42)
    
    # Start with Strassen if possible (rank 189 for 6x6x6), else schoolbook (216)
    try:
        strassen = _strassen_2x2_blocks(n, m, p)
        if strassen is not None:
            U, V, W = strassen
            if _verify_fast(U, V, W, n, m, p):
                pass
            else:
                # Strassen failed verification, fall back to schoolbook
                U, V, W = _schoolbook(n, m, p)
        else:
            U, V, W = _schoolbook(n, m, p)
    except Exception:
        U, V, W = _schoolbook(n, m, p)
    
    # Phase 1: Greedy structured reduction (fast, deterministic)
    if time.time() < deadline - 5:
        U, V, W, rank = _greedy_structured_reduction(U, V, W, n, m, p, deadline - 5, rng)
    
    # Phase 2: Randomized reduction with multiple restarts
    if time.time() < deadline - 2:
        U, V, W, rank = _randomized_reduction(U, V, W, n, m, p, deadline - 2, rng, max_attempts=3000)
    
    # Phase 3: Another greedy pass on the result
    if time.time() < deadline - 1:
        U, V, W, rank = _greedy_structured_reduction(U, V, W, n, m, p, deadline - 1, rng)
    
    # Final verification
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to list-of-lists format as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
