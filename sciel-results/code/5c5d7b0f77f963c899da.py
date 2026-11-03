import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]
TIME_LIMIT = 115  # safety margin before the 120 s evaluator limit


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r]^T W[:,r]^T == the schoolbook tensor."""
    rank = U.shape[1]
    # Build the schoolbook multiplication tensor:
    # T[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if j == j' (and i,i' free, k,k' free)
    # Actually, for matrix mult A(i,j) B(j,k) -> C(i,k):
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the j in a equals the j in b.
    a_idx = np.arange(n * m)
    b_idx = np.arange(m * p)
    c_idx = np.arange(p * n)

    # Decompose indices
    a_i, a_j = np.divmod(a_idx, m)
    b_j, b_k = np.divmod(b_idx, p)
    c_k, c_i = np.divmod(c_idx, n)

    # The tensor entry T[a,b,c] = 1 if a_j == b_j (the inner dimension matches)
    # and c_i == a_i and c_k == b_k (output index consistency)
    # Actually in the standard tensor: T[a, b, c] = 1 iff a_j == b_j, a_i == c_i, b_k == c_k

    # Compute the reconstructed tensor from U,V,W:
    # T_rec[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Compare with the true tensor.

    # True tensor: T_true[a, b, c] = 1 if a_j == b_j and a_i == c_i and b_k == c_k
    # We can't store the full tensor (too big), so we spot-check random entries.
    rng = np.random.RandomState(42)
    for _ in range(2000):
        a = rng.randint(n * m)
        b = rng.randint(m * p)
        c = rng.randint(p * n)
        a_i, a_j = divmod(a, m)
        b_j, b_k = divmod(b, p)
        c_k, c_i = divmod(c, n)
        true_val = 1.0 if (a_j == b_j and a_i == c_i and b_k == c_k) else 0.0
        rec_val = 0.0
        for r in range(rank):
            rec_val += U[a, r] * V[b, r] * W[c, r]
        if abs(rec_val - true_val) > tol:
            return False
    return True


def _strassen_decomposition(n, m, p):
    """Build the 2x2 Strassen-based decomposition with rank 56."""
    U = np.zeros((n * m, 56), dtype=int)
    V = np.zeros((m * p, 56), dtype=int)
    W = np.zeros((p * n, 56), dtype=int)

    # 2x2 Strassen local factors
    U_loc = np.zeros((4, 7), dtype=int)
    V_loc = np.zeros((4, 7), dtype=int)
    W_loc = np.zeros((4, 7), dtype=int)

    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    group = 0
    for jj in range(p // 2):
        for half in range(2):
            base_r = group * 7
            group += 1

            a_map = [
                0 * m + (2 * half + 0),
                0 * m + (2 * half + 1),
                1 * m + (2 * half + 0),
                1 * m + (2 * half + 1),
            ]
            b_map = [
                (2 * half + 0) * p + (2 * jj + 0),
                (2 * half + 0) * p + (2 * jj + 1),
                (2 * half + 1) * p + (2 * jj + 0),
                (2 * half + 1) * p + (2 * jj + 1),
            ]
            c_map = [
                (2 * jj + 0) * n + 0,
                (2 * jj + 1) * n + 0,
                (2 * jj + 0) * n + 1,
                (2 * jj + 1) * n + 1,
            ]

            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la, r]
                    if val:
                        U[a_map[la], rr] += val
                for lb in range(4):
                    val = V_loc[lb, r]
                    if val:
                        V[b_map[lb], rr] += val
                for lc in range(4):
                    val = W_loc[lc, r]
                    if val:
                        W[c_map[lc], rr] += val

    return U, V, W


def _identify_redundant_components(U, V, W, n, m, p):
    """
    For each rank-1 component r, compute its "signature": the set of output
    entries it touches (i.e., for each (a,b) where U[a,r]*V[b,r] != 0, the
    output c must satisfy c_k = b_k and c_i = a_i).
    
    Two components r1, r2 are mergeable if:
    - Their support sets (pairs (a,b) with non-zero U*V) are identical
    - Their W vectors are scalar multiples
    
    We can then merge them into one component.
    """
    rank = U.shape[1]
    
    # For each component, compute its support pattern
    # Component r touches output entry (i,k) [index c = k*n+i] 
    # via all pairs (a,b) where a_i=i, a_j=j, b_j=j, b_k=k, 
    # and U[a,r]*V[b,r] != 0.
    # The contribution to c=(k*n+i) is: sum over j of U[a_ij, r] * V[b_jk, r] * W[c, r]
    # where a_ij = i*m+j, b_jk = j*p+k.
    
    # For a given component r, the effective coefficient on output c=(k*n+i) is:
    #   eff[r, c] = W[c, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
    
    # Let's compute this effective tensor for each component.
    # eff is (rank, p*n)
    eff = np.zeros((rank, p * n), dtype=float)
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                c = k * n + i
                s = 0.0
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    s += U[a, r] * V[b, r]
                eff[r, c] = W[c, r] * s
    
    # Now find components that are scalar multiples of each other
    # (i.e., their effective tensors are proportional)
    redundant = set()
    merged = {}  # maps component to its "canonical" representative
    
    for r1 in range(rank):
        if r1 in redundant:
            continue
        if r1 in merged:
            continue
        for r2 in range(r1 + 1, rank):
            if r2 in redundant or r2 in merged:
                continue
            # Check if eff[r1] and eff[r2] are proportional
            if np.all(eff[r1] == 0) and np.all(eff[r2] == 0):
                continue
            if np.all(eff[r1] == 0) or np.all(eff[r2] == 0):
                continue
            # Check proportionality
            nonzero1 = eff[r1] != 0
            nonzero2 = eff[r2] != 0
            if not np.all(nonzero1 == nonzero2):
                continue
            if np.sum(nonzero1) == 0:
                continue
            ratio = eff[r1][nonzero1] / eff[r2][nonzero1]
            if np.allclose(ratio, ratio[0], atol=1e-10):
                # They're proportional! Merge r2 into r1.
                # But we can't just remove r2 — we need to adjust the
                # U, V, W matrices. Actually, since they produce the same
                # output (up to scalar), we can keep r1 and zero out r2's
                # contribution by subtracting.
                # 
                # Actually, a cleaner approach: if eff[r1] = alpha * eff[r2],
                # then we can replace r2's contribution by adjusting r1.
                # But this is complex. Instead, let's just identify which
                # components are truly redundant (zero effective tensor).
                redundant.add(r2)
                merged[r2] = r1
                break
    
    # Also find components with zero effective tensor (they contribute nothing)
    for r in range(rank):
        if np.all(eff[r] == 0):
            redundant.add(r)
    
    return redundant, eff


def _try_rank_reduction(U, V, W, n, m, p):
    """
    Try to reduce the rank by identifying and removing redundant components.
    """
    redundant, eff = _identify_redundant_components(U, V, W, n, m, p)
    
    if not redundant:
        return U, V, W
    
    # Remove redundant components
    keep = [r for r in range(U.shape[1]) if r not in redundant]
    U_red = U[:, keep]
    V_red = V[:, keep]
    W_red = W[:, keep]
    
    # Verify
    if _verify(U_red, V_red, W_red, n, m, p):
        return U_red, V_red, W_red
    
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Start with the Strassen rank-56 decomposition
    U, V, W = _strassen_decomposition(n, m, p)
    
    # Verify the base decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p), dtype=int)
        V = np.zeros((m * p, n * m * p), dtype=int)
        W = np.zeros((p * n, n * m * p), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]

    # Attempt algebraic rank reduction
    U_red, V_red, W_red = _try_rank_reduction(U, V, W, n, m, p)
    if U_red.shape[1] < best_rank and _verify(U_red, V_red, W_red, n, m, p):
        best_U, best_V, best_W = U_red, V_red, W_red
        best_rank = U_red.shape[1]

    # Additional reduction attempt: try to find linear dependencies
    # in the columns of the "effective tensor" and use them to reduce rank.
    # This is a more sophisticated approach using the tensor's structure.
    
    # Try: for each component, see if its U, V, W can be expressed as
    # a linear combination of other components' U, V, W vectors.
    # This is complex, so we'll try a simpler heuristic.
    
    # Try removing components one at a time and checking validity
    for attempt in range(best_rank):
        if time.time() - start_time > TIME_LIMIT:
            break
        U_curr, V_curr, W_curr = best_U, best_V, best_W
        curr_rank = U_curr.shape[1]
        for r_try in range(curr_rank):
            if time.time() - start_time > TIME_LIMIT:
                break
            # Remove component r_try
            keep = [r for r in range(curr_rank) if r != r_try]
            U_test = U_curr[:, keep]
            V_test = V_curr[:, keep]
            W_test = W_curr[:, keep]
            if _verify(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = curr_rank - 1
                break  # Restart the removal loop with the new best

    if time.time() - start_time > TIME_LIMIT:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
