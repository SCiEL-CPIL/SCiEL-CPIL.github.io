import time
import numpy as np
from itertools import permutations

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms(i0, j0, k0, n, m, p):
    """Generate 7 Strassen rank-1 terms for a 2x2x2 block at offset (i0,j0,k0)."""
    terms = []
    a00 = i0 * m + j0
    a01 = i0 * m + j0 + 1
    a10 = (i0 + 1) * m + j0
    a11 = (i0 + 1) * m + j0 + 1
    b00 = j0 * p + k0
    b01 = j0 * p + k0 + 1
    b10 = (j0 + 1) * p + k0
    b11 = (j0 + 1) * p + k0 + 1
    c00 = k0 * n + i0
    c01 = k0 * n + i0 + 1
    c10 = (k0 + 1) * n + i0
    c11 = (k0 + 1) * n + i0 + 1

    terms.append(([(a00,1),(a11,1)], [(b00,1),(b11,1)], [(c00,1),(c11,1)]))
    terms.append(([(a10,1),(a11,1)], [(b00,1)], [(c10,1),(c11,-1)]))
    terms.append(([(a00,1)], [(b10,1),(b00,-1)], [(c01,1),(c11,1)]))
    terms.append(([(a11,1)], [(b11,1),(b01,-1)], [(c00,1),(c10,1)]))
    terms.append(([(a00,1),(a01,1)], [(b11,1)], [(c00,-1),(c01,1)]))
    terms.append(([(a10,1),(a00,-1)], [(b00,1),(b01,1)], [(c11,1)]))
    terms.append(([(a01,1),(a11,-1)], [(b10,1),(b11,1)], [(c00,1)]))
    return terms


def _schoolbook_term(i, j, k, n, m, p):
    """Single schoolbook rank-1 term."""
    a = i * m + j
    b = j * p + k
    c = k * n + i
    return ([(a, 1)], [(b, 1)], [(c, 1)])


def _build_from_partition(m_parts, p_parts, n, m, p):
    """
    Build decomposition using block-Strassen with given partitions of m and p.
    n=2 is always treated as a 2-row block.
    Returns (terms_list, rank).
    """
    terms = []
    for j_start, j_size in zip(_cumsum(m_parts), m_parts):
        for k_start, k_size in zip(_cumsum(p_parts), p_parts):
            if j_size == 2 and k_size == 2:
                terms.extend(_strassen_2x2x2_terms(0, j_start, k_start, n, m, p))
            elif j_size == 2 and k_size == 1:
                for i in range(n):
                    for j in range(j_start, j_start + j_size):
                        terms.append(_schoolbook_term(i, j, k_start, n, m, p))
            elif j_size == 1 and k_size == 2:
                for i in range(n):
                    for k in range(k_start, k_start + k_size):
                        terms.append(_schoolbook_term(i, j_start, k, n, m, p))
            elif j_size == 1 and k_size == 1:
                for i in range(n):
                    terms.append(_schoolbook_term(i, j_start, k_start, n, m, p))
            else:
                for i in range(n):
                    for j in range(j_start, j_start + j_size):
                        for k in range(k_start, k_start + k_size):
                            terms.append(_schoolbook_term(i, j, k, n, m, p))
    return terms, len(terms)


def _cumsum(parts):
    """Return list of start indices for each part."""
    starts = []
    s = 0
    for x in parts:
        starts.append(s)
        s += x
    return starts


def _matrices_from_terms(terms, n, m, p):
    """Build U, V, W numpy arrays from a list of rank-1 terms."""
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_ent, v_ent, w_ent) in enumerate(terms):
        for idx, val in u_ent:
            U[idx, r] = val
        for idx, val in v_ent:
            V[idx, r] = val
        for idx, val in w_ent:
            W[idx, r] = val
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify decomposition against the matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    Rec = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T - Rec))
    return err < 1e-9, err


def _build_2x3x3_subblock(j_start, k_start, n, m, p):
    """
    Build a low-rank decomposition for the 2x3x3 sub-block
    (i in {0,1}, j in {j_start..j_start+2}, k in {k_start..k_start+2}).
    Uses a structured approach: treat as 2x2 Strassen on first two cols/rows
    plus schoolbook for the third, then try to merge.
    Returns list of terms.
    """
    terms = []
    # Use 2x2 Strassen for (j_start, j_start+1) x (k_start, k_start+1)
    terms.extend(_strassen_2x2x2_terms(0, j_start, k_start, n, m, p))
    # Schoolbook for remaining: j in {j_start, j_start+1, j_start+2}, k = k_start+2
    for i in range(n):
        for j in range(j_start, j_start + 3):
            terms.append(_schoolbook_term(i, j, k_start + 2, n, m, p))
    # Schoolbook for remaining: j = j_start+2, k in {k_start, k_start+1}
    for i in range(n):
        for k in range(k_start, k_start + 2):
            terms.append(_schoolbook_term(i, j_start + 2, k, n, m, p))
    return terms


def _search_best_partition(n, m, p):
    """
    Search over all partitions of m and p into parts of size 1 or 2.
    Return the best (terms, rank) found.
    """
    def gen_partitions(total, max_part=2):
        if total == 0:
            yield []
            return
        if total >= 1:
            for rest in gen_partitions(total - 1, max_part):
                yield [1] + rest
        if total >= 2:
            for rest in gen_partitions(total - 2, max_part):
                yield [2] + rest

    best_terms = None
    best_rank = m * p * n

    m_parts_list = list(gen_partitions(m))
    p_parts_list = list(gen_partitions(p))

    for mp in m_parts_list:
        for pp in p_parts_list:
            terms, rank = _build_from_partition(mp, pp, n, m, p)
            if rank < best_rank:
                best_rank = rank
                best_terms = terms

    return best_terms, best_rank


def _try_rank_reduction(U, V, W, T, n, m, p, max_iters=2000, tol=1e-10):
    """
    Try to reduce rank by zeroing out columns and adjusting via gradient descent.
    Returns (U_reduced, V_reduced, W_reduced, new_rank) or None if no improvement.
    """
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    best_err = np.max(np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T))
    
    # Try removing one column at a time and optimizing
    for col_to_remove in range(rank):
        if time.time() > 140:
            break
        # Create copy without column col_to_remove
        mask = [i for i in range(rank) if i != col_to_remove]
        U_t = U[:, mask].copy()
        V_t = V[:, mask].copy()
        W_t = W[:, mask].copy()
        new_rank = rank - 1
        
        # Gradient descent to minimize ||einsum(U,V,W) - T||^2
        lr = 0.01
        for it in range(max_iters):
            if time.time() > 140:
                break
            Rec = np.einsum('ar,br,cr->abc', U_t, V_t, W_t)
            diff = Rec - T
            # Gradients
            dU = np.einsum('abc,br,cr->ar', diff, V_t, W_t)
            dV = np.einsum('abc,ar,cr->br', diff, U_t, W_t)
            dW = np.einsum('abc,ar,br->cr', diff, U_t, V_t)
            U_t -= lr * dU
            V_t -= lr * dV
            W_t -= lr * dW
        
        err = np.max(np.abs(np.einsum('ar,br,cr->abc', U_t, V_t, W_t) - T))
        if err < tol and new_rank < best_rank:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = new_rank
            best_err = err
            # Continue trying to remove more columns
            U, V, W = U_t, V_t, W_t
            rank = new_rank
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None


def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses block-Strassen with partition search, then attempts rank reduction.
    """
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = DIMENSIONS

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Search for the best partition
    best_terms, best_rank = _search_best_partition(n, m, p)

    if time.time() > deadline - 30:
        # Not enough time for reduction; just use best partition
        U, V, W = _matrices_from_terms(best_terms, n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build matrices from best terms
    U, V, W = _matrices_from_terms(best_terms, n, m, p)

    # Verify
    valid, err = _verify(U, V, W, n, m, p)

    if not valid:
        # Fallback to schoolbook
        nmp = n * m * p
        U = np.zeros((n * m, nmp))
        V = np.zeros((m * p, nmp))
        W = np.zeros((p * n, nmp))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank reduction
    if time.time() < deadline - 5:
        result = _try_rank_reduction(U, V, W, T, n, m, p)
        if result is not None:
            U, V, W, new_rank = result
            # Verify reduced solution
            valid2, err2 = _verify(U, V, W, n, m, p)
            if valid2:
                pass  # Use reduced version
            else:
                # Revert to original
                U, V, W = _matrices_from_terms(best_terms, n, m, p)

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
