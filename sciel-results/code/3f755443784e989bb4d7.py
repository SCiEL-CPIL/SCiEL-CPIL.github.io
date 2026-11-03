"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11
import time

# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _generate_candidates():
    """All nonzero {−1,0,1} vectors of weight 1 and weight 2 in R^11."""
    vecs = []
    # weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            vecs.append(v)
    # weight 2
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def _build_conflict_matrix(cands):
    """Return boolean matrix M where M[i,j] = True iff cands[i],cands[j] conflict
    (i.e. |u·v| > max_norm^2, where max_norm^2 = 4 for weight-2 pool)."""
    n = len(cands)
    # inner product matrix
    G = cands @ cands.T  # (n, n)
    max_n2 = 4
    conflict = np.abs(G) > max_n2
    np.fill_diagonal(conflict, False)
    return conflict


def _spectral_clusters(conflict, n_clusters=8, max_iter=200):
    """Spectral clustering on the compatibility graph (non-conflict edges).
    Returns cluster labels array of length n."""
    n = conflict.shape[0]
    # Compatibility adjacency: edge if not in conflict
    comp = (~conflict).astype(np.float64)
    np.fill_diagonal(comp, 0.0)
    A = sp.csr_matrix(comp)
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.csr_matrix(np.eye(n)) - A
    # Solve (L - lambda I) x = 0 via eigenvalue problem
    try:
        vals, vecs = spla.eigsh(L, k=n_clusters, which='SM')
    except Exception:
        try:
            vals, vecs = spla.eigsh(L, k=max(2, n_clusters // 2), which='SM')
        except Exception:
            # fallback: random labels
            return np.zeros(n, dtype=int)
    # take leading eigenvectors (smallest eigenvalues)
    order = np.argsort(vals)
    emb = vecs[:, order[:n_clusters]]
    # normalize rows
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    # simple k-means
    labels = _kmeans(emb, n_clusters, max_iter=max_iter, seed=42)
    return labels


def _kmeans(X, k, max_iter=200, seed=0):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    # init: pick k random points
    idx = rng.choice(n, k, replace=False)
    C = X[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # assign
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        # update centers
        for j in range(k):
            mask = new_labels == j
            if mask.any():
                C[j] = X[mask].mean(axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels


def _largest_cluster(conflict, labels):
    """Return indices of the largest cluster that is internally conflict-free."""
    best = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        # check internal conflicts
        sub = conflict[np.ix_(idx, idx)]
        if not sub.any():
            if len(idx) > len(best):
                best = idx.tolist()
    return best


def _greedy_min_degree(cands, conflict, start_idx=None):
    """Min-degree greedy: repeatedly add the vertex with fewest conflicts
    to the current set."""
    n = len(cands)
    in_set = np.zeros(n, dtype=bool)
    chosen = []
    # start with a vertex
    if start_idx is not None and not in_set[start_idx]:
        in_set[start_idx] = True
        chosen.append(start_idx)
    # work with remaining
    remaining = [i for i in range(n) if not in_set[i]]
    while remaining:
        # for each remaining vertex, count conflicts with chosen
        best_v = -1
        best_deg = n + 1
        # vectorized: conflict rows for remaining vs chosen
        if chosen:
            ch = np.array(chosen)
            degs = conflict[np.ix_(remaining, ch)].sum(axis=1)
        else:
            degs = np.zeros(len(remaining), dtype=int)
        # pick min degree, tie-break by index
        order = np.argsort(degs)
        # try candidates in order until one has zero conflicts with chosen
        for o in order:
            v = remaining[o]
            if chosen and conflict[v, chosen].any():
                continue
            best_v = v
            break
        if best_v < 0:
            break
        in_set[best_v] = True
        chosen.append(best_v)
        remaining.remove(best_v)
    return chosen


def _local_search_1for2(cands, conflict, chosen_set):
    """1-for-2 local search: try replacing one chosen vector with two new ones
    if it increases the set size."""
    chosen = list(chosen_set)
    n = len(cands)
    in_set = np.zeros(n, dtype=bool)
    for v in chosen:
        in_set[v] = True
    improved = True
    iters = 0
    max_iters = 2000
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i, a in enumerate(chosen):
            # try removing a and adding two new vectors b, c
            in_set[a] = False
            # candidates that don't conflict with current (without a)
            cur_conflict_mask = np.zeros(n, dtype=bool)
            for x in chosen:
                cur_conflict_mask |= conflict[x]
            cur_conflict_mask[a] = False
            candidates = np.where(~in_set & ~cur_conflict_mask)[0]
            # try pairs
            found = False
            for bi in range(len(candidates)):
                b = candidates[bi]
                for ci in range(bi + 1, len(candidates)):
                    c = candidates[ci]
                    if conflict[b, c]:
                        continue
                    # add b and c
                    chosen_new = [x for x in chosen if x != a]
                    chosen_new.append(b)
                    chosen_new.append(c)
                    # verify no conflicts
                    ok = True
                    for x in chosen_new:
                        for y in chosen_new:
                            if x != y and conflict[x, y]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        in_set[a] = False
                        in_set[b] = True
                        in_set[c] = True
                        chosen = chosen_new
                        improved = True
                        found = True
                        break
                if found:
                    break
            if not found:
                in_set[a] = True
    return chosen


def _verify(cands, chosen):
    """Verify chosen set has min pairwise distance >= max norm."""
    if not chosen:
        return True
    V = cands[np.array(chosen)]
    G = V @ V.T
    norms2 = np.diag(G)
    max_n2 = norms2.max()
    off = G.copy()
    np.fill_diagonal(off, 0)
    # min squared distance = min over pairs of |u-v|^2 = |u|^2 + |v|^2 - 2 u·v
    # we need |u-v|^2 >= max_n2 for all pairs
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            d2 = norms2[i] + norms2[j] - 2 * G[i, j]
            if d2 < max_n2 - 1e-9:
                return False
    return True


def construct():
    deadline = time.time() + 140.0
    cands = _generate_candidates()
    n = len(cands)
    conflict = _build_conflict_matrix(cands)

    best_set = []

    # Phase 1: spectral clustering to find dense clusters
    try:
        labels = _spectral_clusters(conflict, n_clusters=8)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            sub = conflict[np.ix_(idx, idx)]
            if not sub.any():
                if len(idx) > len(best_set):
                    best_set = idx.tolist()
    except Exception:
        pass

    # Phase 2: min-degree greedy from multiple starts
    starts = [0, 1, 2, n // 2, n - 1]
    # also try the best spectral cluster's first vertex
    if best_set:
        starts.append(best_set[0])
    for s in starts:
        if time.time() > deadline - 30:
            break
        chosen = _greedy_min_degree(cands, conflict, start_idx=s)
        if len(chosen) > len(best_set):
            best_set = chosen

    # Phase 3: local search refinement on the best set
    if best_set and time.time() < deadline - 20:
        refined = _local_search_1for2(cands, conflict, best_set)
        if len(refined) > len(best_set):
            best_set = refined

    # Final verification
    if not _verify(cands, best_set):
        # fallback to a trivial valid set
        best_set = [0]

    return cands[np.array(best_set, dtype=int)]
# EVOLVE-BLOCK-END
