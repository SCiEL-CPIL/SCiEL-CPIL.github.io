"""Maximum Cut: spectral init + vectorized greedy + SA + multi-flip + restarts + polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Cut size via matrix ops: sum_v labels[v]*deg[v] - labels^T A labels."""
    lf = labels.astype(np.float64)
    deg = adj.sum(axis=1).astype(np.float64)
    return float(lf @ deg - lf @ adj @ lf)


def _compute_gains(adj, labels):
    """gain[v] = delta in cut if we flip v = deg[v] - 2*same[v]."""
    same_mask = (labels[:, None] == labels[None, :])
    same_counts = (adj & same_mask).sum(axis=1)
    deg = adj.sum(axis=1)
    return deg - 2 * same_counts


def _apply_flip(adj, labels, gains, v):
    """Flip v in-place; update gains of v and its neighbors (vectorized)."""
    nv = labels[v]
    labels[v] = 1 - nv
    nbrs = adj[v]
    gains[nbrs] -= 2 * (labels[nbrs] == labels[v]).astype(np.int64)
    gains[nbrs] += 2 * (labels[nbrs] != labels[v]).astype(np.int64)
    gains[v] = -gains[v]


def _greedy_improve(adj, labels, deadline):
    """Vectorized multi-pass greedy: flip every vertex with positive gain."""
    n = len(labels)
    labels = labels.copy()
    gains = _compute_gains(adj, labels)
    for _ in range(200):
        if time.time() >= deadline:
            break
        flip = gains > 0
        if not flip.any():
            break
        idx = np.where(flip)[0]
        for v in idx:
            _apply_flip(adj, labels, gains, v)
    return labels, gains


def _sa(adj, labels, gains, deadline, start_temp, cooling):
    """Simulated annealing with vectorized flips and periodic gain refresh."""
    n = len(labels)
    labels = labels.copy()
    gains = gains.copy()
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    temp = start_temp
    it = 0
    while time.time() < deadline and temp > 0.05:
        v = random.randint(0, n - 1)
        delta = gains[v]
        if delta > 0 or random.random() < np.exp(delta / temp):
            _apply_flip(adj, labels, gains, v)
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        temp *= cooling
        it += 1
        if it % 2000 == 0:
            gains = _compute_gains(adj, labels)  # guard against drift
    return best_labels


def _two_flip(adj, labels, gains, deadline, tries):
    """Exact 2-flip moves (delta computed analytically, no O(n) recompute)."""
    n = len(labels)
    labels = labels.copy()
    gains = gains.copy()
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    for _ in range(tries):
        if time.time() >= deadline:
            break
        i, j = random.sample(range(n), 2)
        if i == j:
            continue
        delta = gains[i] + gains[j]
        if adj[i, j]:
            delta += 4 if labels[i] == labels[j] else -4
        if delta > 0:
            _apply_flip(adj, labels, gains, i)
            _apply_flip(adj, labels, gains, j)
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    return best_labels, best_cut


def _three_flip(adj, labels, gains, deadline, tries):
    """3-flip moves: exact delta via direct edge-counting (O(deg) per try)."""
    n = len(labels)
    labels = labels.copy()
    gains = gains.copy()
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    for _ in range(tries):
        if time.time() >= deadline:
            break
        i, j, k = random.sample(range(n), 3)
        s = labels[i] + labels[j] + labels[k]
        delta = 0
        for a, b in ((i, j), (i, k), (j, k)):
            if adj[a, b]:
                delta += -4 if labels[a] == labels[b] else 4
        for a in (i, j, k):
            for u in np.where(adj[a])[0]:
                if u in (i, j, k):
                    continue
                delta += -2 if labels[a] == labels[u] else 2
        if delta > 0:
            for a in (i, j, k):
                _apply_flip(adj, labels, gains, a)
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    return best_labels, best_cut


def _polish(adj, labels, deadline):
    """Final exact 1-flip local-optima pass (fresh gains, no drift)."""
    labels = labels.copy()
    for _ in range(100):
        if time.time() >= deadline:
            break
        gains = _compute_gains(adj, labels)
        flip = gains > 0
        if not flip.any():
            break
        for v in np.where(flip)[0]:
            _apply_flip(adj, labels, gains, v)
    return labels


def _spectral_init(adj):
    """Fiedler vector split."""
    n = len(adj)
    af = adj.astype(np.float64)
    deg = af.sum(axis=1)
    L = np.diag(deg) - af
    try:
        _, V = np.linalg.eigh(L)
        fiedler = V[:, 1]
    except Exception:
        fiedler = np.random.randn(n)
    thr = np.mean(fiedler)
    labels = (fiedler > thr).astype(np.int64)
    if labels.sum() == 0 or labels.sum() == n:
        thr = np.median(fiedler)
        labels = (fiedler > thr).astype(np.int64)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, size=n).astype(np.int64)
    return labels


def _random_spectral_init(adj):
    """Random subspace spectral projection split (diverse high-quality seed)."""
    n = len(adj)
    af = adj.astype(np.float64)
    deg = af.sum(axis=1)
    L = np.diag(deg) - af
    try:
        _, V = np.linalg.eigh(L)
    except Exception:
        return np.random.randint(0, 2, size=n).astype(np.int64)
    m = min(5, n - 1)
    cols = list(range(1, 2 + m))
    cols += [random.randint(2, n - 1) for _ in range(3)]
    Z = V[:, cols]
    proj = Z @ np.random.randn(len(cols))
    thr = np.mean(proj)
    labels = (proj > thr).astype(np.int64)
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, size=n).astype(np.int64)
    return labels


def partition(adjacency):
    start_time = time.time()
    DEADLINE = start_time + 115.0

    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    adj = np.array(adjacency, dtype=bool)

    best_labels = None
    best_cut = -1

    def consider(labels, cut):
        nonlocal best_labels, best_cut
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Phase 1: diverse spectral seeds + greedy
    seeds = [_spectral_init(adj)] + [_random_spectral_init(adj) for _ in range(3)]
    for s in seeds:
        if time.time() >= DEADLINE - 5:
            break
        ref, _ = _greedy_improve(adj, s, DEADLINE - 5)
        consider(ref, _cut_size(adj, ref))

    # Phase 2: SA + 2-flip + 3-flip from best
    if time.time() < DEADLINE - 10 and best_labels is not None:
        gains = _compute_gains(adj, best_labels)
        sa_l = _sa(adj, best_labels, gains, DEADLINE - 5, 8.0, 0.9995)
        consider(sa_l, _cut_size(adj, sa_l))
        g2 = _compute_gains(adj, best_labels)
        tf_l, _ = _two_flip(adj, best_labels, g2, DEADLINE - 5, 4000)
        consider(tf_l, _cut_size(adj, tf_l))
        g3 = _compute_gains(adj, best_labels)
        tf3_l, _ = _three_flip(adj, best_labels, g3, DEADLINE - 5, 1500)
        consider(tf3_l, _cut_size(adj, tf3_l))

    # Phase 3: random restarts with greedy + short SA
    restarts = 0
    while time.time() < DEADLINE - 15 and restarts < 40:
        restarts += 1
        labels = np.random.randint(0, 2, size=n).astype(np.int64)
        ref, _ = _greedy_improve(adj, labels, DEADLINE - 5)
        consider(ref, _cut_size(adj, ref))
        if time.time() < DEADLINE - 15:
            gains = _compute_gains(adj, ref)
            sa_l = _sa(adj, ref, gains, DEADLINE - 5, 5.0, 0.999)
            consider(sa_l, _cut_size(adj, sa_l))

    # Phase 4: final polish on best
    if best_labels is not None and time.time() < DEADLINE - 2:
        polished = _polish(adj, best_labels, DEADLINE - 1)
        consider(polished, _cut_size(adj, polished))

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
