"""Matrix multiplication 4x5 x 5x5 low-rank exact decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _matmul_tensor(n, m, p):
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _embed_block(T, n, m, p, i0, j0, k0, bi, bj, bk):
    """Return (n,m,p) tensor with T placed at block offset (i0,j0,k0)."""
    full = np.zeros((n, m, p))
    full[i0:i0 + bi, j0:j0 + bj, k0:k0 + bk] = T
    return full

def _build_factors(n, m, p, terms):
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j, c] += val
        for (j, k), val in v.items():
            V[j * p + k, c] += val
        for (k, i), val in w.items():
            W[k * n + i, c] += val
    return U, V, W

def _check(U, V, W, n, m, p):
    T = _matmul_tensor(n, m, p)
    for c in range(U.shape[1]):
        T -= np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(T)) < 1e-7

def _strassen_2x2():
    """7 terms for 2x2 * 2x2. Each term (u,v,w) with dict keys."""
    def D(a, b):
        return {(0, 0): a[0], (0, 1): b[0], (1, 0): a[1], (1, 1): b[1]}
    terms = []
    # M1 = (a00+a11)*(b00+b11)
    u = {(0, 0): 1, (1, 1): 1}
    v = {(0, 0): 1, (1, 1): 1}
    w = {(0, 0): 1, (1, 1): 1}
    terms.append((u, v, w))
    # M2 = (a10+a11)*b00
    u = {(1, 0): 1, (1, 1): 1}
    v = {(0, 0): 1}
    w = {(0, 0): 1, (1, 0): 1}
    terms.append((u, v, w))
    # M3 = a00*(b10-b11)
    u = {(0, 0): 1}
    v = {(1, 0): 1, (1, 1): -1}
    w = {(0, 1): 1, (1, 1): 1}
    terms.append((u, v, w))
    # M4 = a11*(b01-b00)
    u = {(1, 1): 1}
    v = {(0, 1): 1, (0, 0): -1}
    w = {(0, 0): 1, (1, 0): 1}
    terms.append((u, v, w))
    # M5 = (a00+a01)*b11
    u = {(0, 0): 1, (0, 1): 1}
    v = {(1, 1): 1}
    w = {(0, 1): 1, (1, 1): 1}
    terms.append((u, v, w))
    # M6 = (a10-a00)*(b00+b01)
    u = {(1, 0): 1, (0, 0): -1}
    v = {(0, 0): 1, (0, 1): 1}
    w = {(0, 0): 1, (1, 1): 1}
    terms.append((u, v, w))
    # M7 = (a01-a11)*(b10+b11)
    u = {(0, 1): 1, (1, 1): -1}
    v = {(1, 0): 1, (1, 1): 1}
    w = {(0, 1): 1, (1, 0): 1}
    terms.append((u, v, w))
    return terms

def _rect_block(n, m, p, method='schoolbook'):
    """Terms for n x m times m x p. Keys: u=(i,j), v=(j,k), w=(k,i)."""
    if method == 'strassen' and n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
    return terms

def _search_block(n, m, p, max_rank, deadline):
    """Brute-force search for exact rank decomposition of the n x m x p
    multiplication tensor using +/-1 entries. Returns list of terms or None."""
    # Build the target tensor.
    target = _matmul_tensor(n, m, p)
    flat = target.reshape(-1)
    # Try ranks from a lower bound up to max_rank.
    # Lower bound: at least max(n,m,p) and at least ceil(size/?) ... use simple.
    lb = max(n, m, p)
    for rank in range(lb, max_rank + 1):
        if time.time() > deadline:
            return None
        # Represent each factor column as a vector of +/-1 (or 0) entries.
        # Use a randomized / structured search: try all sign patterns is too large.
        # Instead use a greedy rank-1 peeling with exact residual check.
        terms = _greedy_peel(n, m, p, rank, deadline)
        if terms is not None:
            return terms
    return None

def _greedy_peel(n, m, p, target_rank, deadline):
    """Peel rank-1 components with +/-1 entries until residual is zero or rank exhausted."""
    residual = _matmul_tensor(n, m, p).copy()
    terms = []
    for _ in range(target_rank):
        if time.time() > deadline:
            return None
        if np.max(np.abs(residual)) < 1e-9:
            break
        # Find a rank-1 component (u,v,w) with entries in {-1,0,1} that is a
        # "face" of the residual: pick a nonzero entry and extend to a hyperplane.
        # Strategy: enumerate all sign patterns on the three slices is large;
        # instead try the "line" through a nonzero entry.
        nz = np.argwhere(residual != 0)
        if len(nz) == 0:
            break
        # Try each candidate generated from a random-ish subset of nonzero entries.
        # For small sizes, enumerate all entries.
        found = False
        for (i, j, k) in nz:
            sign = 1.0 if residual[i, j, k] > 0 else -1.0
            # Candidate: u has sign at (i,j); v has 1 at (j,k); w has 1 at (k,i).
            # This is the schoolbook line; it reduces exactly one entry but may
            # overshoot others. Use it only if it doesn't create new nonzeros
            # beyond existing support (keeps residual within support).
            u = np.zeros((n, m)); v = np.zeros((m, p)); w = np.zeros((p, n))
            u[i, j] = sign
            v[j, k] = 1.0
            w[k, i] = 1.0
            comp = np.einsum('ij,jk,ki->ijk', u, v, w)
            newres = residual - comp
            # Accept if support doesn't grow.
            if np.count_nonzero(newres) <= np.count_nonzero(residual):
                residual = newres
                terms.append((u, v, w))
                found = True
                break
        if not found:
            return None
    if np.max(np.abs(residual)) < 1e-9:
        # Convert arrays to dicts.
        out = []
        for (u, v, w) in terms:
            ud = {(a, b): float(u[a, b]) for (a, b) in np.argwhere(u != 0)}
            vd = {(a, b): float(v[a, b]) for (a, b) in np.argwhere(v != 0)}
            wd = {(a, b): float(w[a, b]) for (a, b) in np.argwhere(w != 0)}
            out.append((ud, vd, wd))
        return out
    return None

def _full_decompose(n, m, p, deadline):
    """Build a full 4x5 x 5x5 decomposition using block partitioning 5 = 2+3."""
    # Partition: rows i in [0..3], cols j in [0..4] split into jA=[0,1] (size2) and jB=[2,3,4] (size3).
    # k in [0..4] split into kA=[0,1] (size2) and kB=[2,3,4] (size3).
    # Blocks (i-size, j-size, k-size):
    #   (4,2,2): A[:,jA] * B[jA,kA]  -> rank via search
    #   (4,2,3): A[:,jA] * B[jA,kB]  -> rank via search
    #   (4,3,2): A[:,jB] * B[jB,kA]  -> rank via search
    #   (4,3,3): A[:,jB] * B[jB,kB]  -> rank via search
    terms_all = []
    blocks = [
        (4, 2, 2, 0, 0, 0),
        (4, 2, 3, 0, 0, 2),
        (4, 3, 2, 0, 2, 0),
        (4, 3, 3, 0, 2, 2),
    ]
    for (bi, bj, bk, i0, j0, k0) in blocks:
        if time.time() > deadline:
            break
        # Determine max rank to search: schoolbook for this block.
        school = bi * bj * bk
        # Try to find a decomposition of the bi x bj x bk multiplication tensor.
        found = _search_block(bi, bj, bk, school, deadline)
        if found is None:
            found = _rect_block(bi, bj, bk)
        # Embed into full (n,m,p)
        for (u, v, w) in found:
            u_full = {}
            for ((ii, jj), val) in u.items():
                u_full[(i0 + ii, j0 + jj)] = val
            v_full = {}
            for ((jj, kk), val) in v.items():
                v_full[(j0 + jj, k0 + kk)] = val
            w_full = {}
            for ((kk, ii), val) in w.items():
                w_full[(k0 + kk, i0 + ii)] = val
            terms_all.append((u_full, v_full, w_full))
    return terms_all

def _schoolbook(n, m, p):
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
    return terms

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0  # safety margin below 150s kill
    best_terms = None
    best_rank = n * m * p
    # 1) Try the structured block decomposition with search.
    try:
        terms = _full_decompose(n, m, p, deadline)
        if terms is not None:
            U, V, W = _build_factors(n, m, p, terms)
            if _check(U, V, W, n, m, p):
                best_terms = terms
                best_rank = len(terms)
    except Exception:
        pass
    # 2) Fallback: schoolbook (always valid).
    if best_terms is None:
        best_terms = _schoolbook(n, m, p)
        best_rank = len(best_terms)
    U, V, W = _build_factors(n, m, p, best_terms)
    # Convert to lists of numbers for the interface.
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
