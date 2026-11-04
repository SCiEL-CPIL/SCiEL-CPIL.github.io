"""Matrix multiplication tensor decomposition for 4x4 * 4x9.

We look for a rank-minimal exact decomposition of the tensor
  T[(i,j), (j,k), (k,i)] = 1   for 0<=i<4, 0<=j<4, 0<=k<9
  T = 0 otherwise
in the form sum_r U_r ⊗ V_r ⊗ W_r.

Strategy:
  1. Build the schoolbook decomposition (rank 144).
  2. Use a greedy "merge" heuristic: repeatedly try to replace two rank-1 terms
     by a single rank-1 term (or a small set) that reproduces their combined
     contribution while cancelling unwanted cross-terms via coefficient choice.
     The key insight: if two terms share the same U-support and same V-support
     but differ in W, or share U and W but differ in V, we can sometimes merge.
     More generally, we try to find linear combinations of pairs/triples that
     yield a valid rank-1 tensor.
  3. Use a local-search / coordinate-descent style approach: fix all but one
     term, and try to replace that term with a better one (or remove it if
     the remaining terms already cover its contribution).
  4. Additionally, exploit the block structure: split the 9 columns into groups
     (4+4+1) and use known low-rank decompositions for M_{4,4,4} if we can
     construct them algorithmically.

Given the time limit (120s), we focus on a fast greedy merge that can reduce
rank from 144 to something lower, and a verification step.
"""

import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    rank = U.shape[1]
    # Build the full tensor T from the decomposition
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We only need to check the non-zero entries and a sample of zero entries.
    # Non-zero: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check a sample of zero entries
    # The total number of (a,b,c) triples is n*m * m*p * p*n = 16*36*36 = 20736
    # We check all of them for safety (it's small enough)
    for a in range(n * m):
        i, j = divmod(a, m)
        for b in range(m * p):
            j2, k = divmod(b, p)
            if j != j2:
                # This is automatically zero in the target
                for c in range(p * n):
                    k2, i2 = divmod(c, n)
                    # In target: T[a,b,c] = 0 because j != j2
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
            else:
                # j == j2, need to check
                for c in range(p * n):
                    k2, i2 = divmod(c, n)
                    # Target: T = 1 if i==i2 and k==k2, else 0
                    target = 1.0 if (i == i2 and k == k2) else 0.0
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - target) > tol:
                        return False
    return True


def _try_merge_two(U, V, W, r1, r2, n, m, p):
    """
    Try to replace terms r1 and r2 with a single term.
    Returns (new_U_col, new_V_col, new_W_col) or None.
    
    The combined contribution of terms r1 and r2 is:
      U[:,r1] ⊗ V[:,r1] ⊗ W[:,r1] + U[:,r2] ⊗ V[:,r2] ⊗ W[:,r2]
    
    We want a single rank-1 tensor U' ⊗ V' ⊗ W' that equals this.
    This is generally impossible, but we can try:
    - If U[:,r1] is proportional to U[:,r2] and V[:,r1] is proportional to V[:,r2],
      then we can merge into a single term with W' = W[:,r1] + W[:,r2].
    - Similarly for other proportional pairs.
    """
    u1, v1, w1 = U[:, r1], V[:, r1], W[:, r1]
    u2, v2, w2 = U[:, r2], V[:, r2], W[:, r2]
    
    # Check if u1 is proportional to u2
    nz1 = np.nonzero(u1)[0]
    nz2 = np.nonzero(u2)[0]
    if len(nz1) == 0 or len(nz2) == 0:
        return None
    
    # Try to find scalar alpha such that u2 = alpha * u1
    # Use the first non-zero element
    a = nz1[0]
    if abs(u1[a]) < 1e-15:
        return None
    alpha = u2[a] / u1[a]
    if np.allclose(u2, alpha * u1, atol=1e-12):
        # U parts are proportional. Check V parts.
        nzv1 = np.nonzero(v1)[0]
        nzv2 = np.nonzero(v2)[0]
        if len(nzv1) == 0 or len(nzv2) == 0:
            return None
        b = nzv1[0]
        if abs(v1[b]) < 1e-15:
            return None
        beta = v2[b] / v1[b]
        if np.allclose(v2, beta * v1, atol=1e-12):
            # Both U and V are proportional. Merge: new term has U=u1, V=v1, W=w1+w2
            new_w = w1 + w2
            return (u1, v1, new_w)
    
    # Try if v1 proportional to v2 and w1 proportional to w2
    nzv1 = np.nonzero(v1)[0]
    nzv2 = np.nonzero(v2)[0]
    if len(nzv1) > 0 and len(nzv2) > 0:
        b = nzv1[0]
        if abs(v1[b]) > 1e-15:
            beta = v2[b] / v1[b]
            if np.allclose(v2, beta * v1, atol=1e-12):
                # V parts proportional. Check W.
                nzw1 = np.nonzero(w1)[0]
                nzw2 = np.nonzero(w2)[0]
                if len(nzw1) > 0 and len(nzw2) > 0:
                    c = nzw1[0]
                    if abs(w1[c]) > 1e-15:
                        gamma = w2[c] / w1[c]
                        if np.allclose(w2, gamma * w1, atol=1e-12):
                            new_u = u1 + u2
                            return (new_u, v1, w1)
    
    # Try if u1 proportional to u2 and w1 proportional to w2
    nz1 = np.nonzero(u1)[0]
    nz2 = np.nonzero(u2)[0]
    if len(nz1) > 0 and len(nz2) > 0:
        a = nz1[0]
        if abs(u1[a]) > 1e-15:
            alpha = u2[a] / u1[a]
            if np.allclose(u2, alpha * u1, atol=1e-12):
                nzw1 = np.nonzero(w1)[0]
                nzw2 = np.nonzero(w2)[0]
                if len(nzw1) > 0 and len(nzw2) > 0:
                    c = nzw1[0]
                    if abs(w1[c]) > 1e-15:
                        gamma = w2[c] / w1[c]
                        if np.allclose(w2, gamma * w1, atol=1e-12):
                            new_v = v1 + v2
                            return (u1, new_v, w1)
    
    return None


def _greedy_merge(U, V, W, n, m, p, deadline):
    """
    Greedy merge: repeatedly try to merge pairs of terms.
    """
    U, V, W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r1 in range(rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline:
                    break
                result = _try_merge_two(U, V, W, r1, r2, n, m, p)
                if result is not None:
                    new_u, new_v, new_w = result
                    # Replace r1 with the merged term, remove r2
                    U[:, r1] = new_u
                    V[:, r1] = new_v
                    W[:, r1] = new_w
                    U = np.delete(U, r2, axis=1)
                    V = np.delete(V, r2, axis=1)
                    W = np.delete(W, r2, axis=1)
                    rank -= 1
                    improved = True
                    break
            if improved:
                break
    return U, V, W


def _try_remove_term(U, V, W, r, n, m, p):
    """
    Check if term r can be removed (i.e., its contribution is already covered
    by the other terms).
    """
    rank = U.shape[1]
    if rank <= 1:
        return False
    # Compute the contribution of term r
    # T_r[a,b,c] = U[a,r] * V[b,r] * W[c,r]
    # Check if the sum of other terms already gives the correct values
    # where term r has non-zero contribution
    u = U[:, r]
    v = V[:, r]
    w = W[:, r]
    nz_u = np.nonzero(u)[0]
    nz_v = np.nonzero(v)[0]
    nz_w = np.nonzero(w)[0]
    
    for a in nz_u:
        for b in nz_v:
            for c in nz_w:
                # Target value at (a,b,c)
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                target = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                # Current contribution from all terms except r
                val = 0.0
                for r2 in range(rank):
                    if r2 == r:
                        continue
                    val += U[a, r2] * V[b, r2] * W[c, r2]
                # The contribution from term r is u[a]*v[b]*w[c]
                term_val = u[a] * v[b] * w[c]
                # Total should equal target
                if abs(val + term_val - target) > 1e-12:
                    # Can't remove
                    return False
    return True


def construct():
    """Construct a tensor decomposition for 4x4 * 4x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 105  # Safety margin
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but just in case
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try greedy merge
    U2, V2, W2 = _greedy_merge(U, V, W, n, m, p, deadline)
    if U2.shape[1] < best_rank:
        if _verify(U2, V2, W2, n, m, p):
            best_rank = U2.shape[1]
            best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
    
    # Try removing individual terms
    if time.time() < deadline:
        U3, V3, W3 = best_U.copy(), best_V.copy(), best_W.copy()
        rank = U3.shape[1]
        removed = True
        while removed and time.time() < deadline:
            removed = False
            for r in range(U3.shape[1]):
                if time.time() > deadline:
                    break
                if _try_remove_term(U3, V3, W3, r, n, m, p):
                    U3 = np.delete(U3, r, axis=1)
                    V3 = np.delete(V3, r, axis=1)
                    W3 = np.delete(W3, r, axis=1)
                    removed = True
                    break
        if U3.shape[1] < best_rank:
            if _verify(U3, V3, W3, n, m, p):
                best_rank = U3.shape[1]
                best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
    
    # Try another round of greedy merge on the reduced set
    if time.time() < deadline:
        U4, V4, W4 = _greedy_merge(best_U, best_V, best_W, n, m, p, deadline)
        if U4.shape[1] < best_rank:
            if _verify(U4, V4, W4, n, m, p):
                best_rank = U4.shape[1]
                best_U, best_V, best_W = U4.copy(), V4.copy(), W4.copy()
    
    # Convert to lists (the evaluator expects lists or arrays)
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
