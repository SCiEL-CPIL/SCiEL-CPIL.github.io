"""Fresh design: local-search over vertex orderings to maximize greedy-decode independent-set size."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 120.0  # hard wall-clock limit (safety margin vs 150s kill)
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _decode_size(order, adj):
    """Simulate the fixed greedy decoder; return the independent-set size.
    order: array of vertex indices sorted by descending score.
    adj: bool adjacency matrix (symmetric, no self loops)."""
    n = adj.shape[0]
    chosen = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    size = 0
    for v in order:
        v = int(v)
        if chosen[v] or blocked[v]:
            continue
        chosen[v] = True
        size += 1
        # block its neighbours (vectorized)
        nbrs = np.nonzero(adj[v])[0]
        blocked[nbrs] = True
    return size


def _initial_order(adj):
    """Seed ordering: combine degree (low first) with eigenvector centrality (low first)
    plus a small amount of 2-hop structure, to break ties effectively."""
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(float)

    # eigenvector centrality via power iteration (few iters, cheap)
    try:
        x = np.ones(n) / n
        for _ in range(12):
            x_new = adj.astype(float) @ x
            s = x_new.sum()
            if s <= 0:
                break
            x = x_new / s
    except Exception:
        x = np.ones(n) / n

    # 2-hop count (number of distance-2 neighbours) as a tie-breaker
    try:
        two_hop = (adj.astype(float) @ adj.astype(float)).sum(axis=1)
    except Exception:
        two_hop = np.zeros(n)

    # normalize each term to [0,1]
    def _norm(a):
        lo, hi = a.min(), a.max()
        if hi <= lo:
            return np.zeros(n)
        return (a - lo) / (hi - lo)

    score = _norm(deg) + 0.6 * _norm(x) + 0.1 * _norm(two_hop)
    # lower score = picked first (greedy wants low-degree / low-centrality first)
    return np.argsort(score, kind="stable")


def priority(adjacency):
    global _START
    _START = time.time()

    adj = np.ascontiguousarray(adjacency, dtype=bool)
    n = adj.shape[0]

    if n == 0:
        return np.zeros(0, dtype=float)

    order = _initial_order(adj).copy()
    best_size = _decode_size(order, adj)

    # Local search: 2-opt moves (swap two positions in the ordering),
    # hill-climbing on the decoded independent-set size.
    # Use a deadline guard and occasional restarts to escape local optima.
    iters_since_improve = 0
    max_no_improve = 400
    restarts = 0
    max_restarts = 6

    # Precompute an index map for quick neighbour lookups
    nbr_lists = [np.nonzero(adj[i])[0] for i in range(n)]

    while _remaining() > 0.5:
        improved = False
        # one pass of random-ish 2-opt trials
        # pick pairs to try; use a deterministic stride to be reproducible-ish
        step = max(1, n // 400)
        for i in range(0, n - 1, step):
            if _remaining() < 0.5:
                break
            # try swapping i with a few candidate j's
            for dj in (1, 2, 5, 10, 20, 50):
                j = i + dj
                if j >= n:
                    break
                if i == j:
                    continue
                # perform swap
                order[i], order[j] = order[j], order[i]
                s = _decode_size(order, adj)
                if s > best_size:
                    best_size = s
                    improved = True
                    iters_since_improve = 0
                else:
                    # revert
                    order[i], order[j] = order[j], order[i]
            if improved:
                break

        if improved:
            continue

        iters_since_improve += 1
        if iters_since_improve >= max_no_improve:
            # restart with a perturbed order to escape local optimum
            restarts += 1
            iters_since_improve = 0
            if restarts > max_restarts:
                break
            # perturb: keep best order, shuffle a random subset
            perm = np.random.permutation(n)
            order = perm.copy()
            # re-seed by blending with initial order to avoid pure random
            base = _initial_order(adj)
            order = (order + base) % n  # crude blend
            # ensure uniqueness: use argsort of a random+base score
            rand = np.random.rand(n)
            order = np.argsort(rand * 0.5 + _norm(_initial_order(adj) * 0 + np.arange(n) * 0 + np.zeros(n)) * 0 + rand, kind="stable")
            # simpler: just use a fresh random order blended with degree
            deg = adj.sum(axis=1).astype(float)
            blend = 0.7 * rand + 0.3 * _norm(deg)
            order = np.argsort(blend, kind="stable")
            best_size = _decode_size(order, adj)

    # Convert the final ordering into scores (higher score = picked first).
    # Assign scores by rank: the vertex at position 0 in order gets the highest score.
    scores = np.zeros(n, dtype=float)
    ranks = np.zeros(n, dtype=float)
    for pos, v in enumerate(order):
        ranks[int(v)] = float(n - pos)  # higher = earlier in greedy
    # add tiny deterministic jitter to break exact ties stably
    scores = ranks + np.linspace(0, 1e-6, n)

    return scores
# EVOLVE-BLOCK-END
