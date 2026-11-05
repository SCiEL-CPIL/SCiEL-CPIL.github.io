"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _evaluate_fast(nbrs, scores):
    """Fast evaluation using precomputed neighbor lists."""
    n = len(nbrs)
    if n == 0:
        return 0
    indices = np.argsort(-scores, kind='stable')
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in indices:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            if len(nbrs[v]) > 0:
                blocked[nbrs[v]] = True
    return count

def _compute_sim_order(n, nbrs):
    """
    Simulate a parallel discrete-time exclusion process.
    Vertices with the fewest remaining neighbors are selected in each round.
    Returns an array of step indices (lower is earlier).
    """
    if n == 0:
        return np.array([])
    
    remaining = np.ones(n, dtype=bool)
    rem_deg = np.zeros(n, dtype=int)
    for i in range(n):
        rem_deg[i] = len(nbrs[i])
        
    sim_order = np.full(n, n, dtype=int)
    t = 0
    while np.any(remaining):
        degs = np.where(remaining, rem_deg, np.inf)
        min_deg = np.min(degs)
        candidates = np.where(remaining & (rem_deg == min_deg))[0]
        for v in candidates:
            sim_order[v] = t
        for v in candidates:
            remaining[v] = False
            for u in nbrs[v]:
                if remaining[u]:
                    rem_deg[u] -= 1
        t += 1
    return sim_order

def _fiedler_fast(L, n, iters=200):
    """Approximate Fiedler vector via power iteration on shifted Laplacian."""
    max_deg = int(np.max(np.sum(L, axis=1))) if n > 0 else 0
    c = 2.0 * max_deg + 1.0
    M = c * np.eye(n) - L
    ones = np.ones(n) / np.sqrt(n)
    v = np.random.randn(n)
    v = v - np.dot(v, ones) * ones
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        v = np.random.randn(n)
        v = v - np.dot(v, ones) * ones
        norm_v = np.linalg.norm(v)
    v /= norm_v
    for _ in range(iters):
        w = M @ v
        w = w - np.dot(w, ones) * ones
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-10:
            break
        v = w / norm_w
    return v

def _compute_indep_nbrs(n, nbrs, adjacency):
    """For each vertex, count how many of its neighbors are NOT adjacent to each other.
    A vertex with many mutually-independent neighbors is 'safe' to pick early."""
    indep = np.zeros(n, dtype=float)
    for i in range(n):
        nb = nbrs[i]
        if len(nb) == 0:
            indep[i] = 0
        elif len(nb) == 1:
            indep[i] = 1.0
        else:
            # Count edges among neighbors
            sub_adj = adjacency[np.ix_(nb, nb)]
            edges_among = np.sum(sub_adj) / 2.0
            # Max independent set among neighbors is at least len(nb) - edges_among
            # Use a simpler proxy: len(nb) - 2*edges_among (each edge removes at most one)
            indep[i] = max(1, len(nb) - edges_among)
    return indep

def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    """
    deadline = time.time() + 138
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    nbrs = [np.where(adjacency[i])[0] for i in range(n)]
    degree = np.array([len(nbrs[i]) for i in range(n)], dtype=float)

    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(degree)

    mean_nd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            mean_nd[i] = np.mean(degree[nbrs[i]])
        else:
            mean_nd[i] = 0
    MeanND_n = norm(mean_nd)

    snd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            snd[i] = np.sum(degree[nbrs[i]])
    SND_n = norm(snd)

    A = adjacency.astype(float)
    A_sq = A @ A
    tri = np.diag(A_sq) / 2.0
    Tri_n = norm(tri)

    clus = np.zeros(n)
    mask = degree > 1
    clus[mask] = 2.0 * tri[mask] / (degree[mask] * (degree[mask] - 1.0))
    Clus_n = norm(clus)

    try:
        D_mat = np.diag(degree)
        L = D_mat - A
        fiedler = _fiedler_fast(L, n, iters=300)
        abs_fiedler = np.abs(fiedler)
        AbsFiedler_n = norm(abs_fiedler)
    except Exception:
        AbsFiedler_n = np.zeros(n)

    core = np.zeros(n, dtype=int)
    deg_copy = degree.astype(int).copy()
    remaining = np.ones(n, dtype=bool)
    max_deg = int(deg_copy.max()) if n > 0 else 0
    for k in range(max_deg, -1, -1):
        for i in range(n):
            if remaining[i] and deg_copy[i] >= k:
                core[i] = k
                remaining[i] = False
                for j in nbrs[i]:
                    if remaining[j]:
                        deg_copy[j] -= 1
    Core_n = norm(core.astype(float))

    sim_order = _compute_sim_order(n, nbrs)
    SimOrder_n = 1.0 - norm(sim_order.astype(float))

    # New feature: independence of neighbor set
    indep_nbrs = _compute_indep_nbrs(n, nbrs, adjacency)
    IndepNbrs_n = norm(indep_nbrs)

    # Features: D, MeanND, SND, Tri, Clus, Core, Fiedler, SimOrder, IndepNbrs
    # Indices:    0      1        2     3     4      5      6         7            8
    
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 1.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 1.0, 0.0, 0.0),
        (-0.8, -0.3, 0.0, 0.0, 0.0, -0.3, 1.0, 0.0, 0.0),
        (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
        (-1.0, -0.3, -0.2, 0.0, 0.0, -0.3, 0.5, 0.0, 0.0),
        (-0.9, -0.4, -0.3, -0.2, -0.2, -0.4, 0.8, 0.0, 0.0),
        (-0.8, -0.4, -0.4, -0.3, -0.3, -0.4, 0.7, 0.0, 0.0),
        (-1.0, -0.6, 0.0, 0.0, 0.0, -0.6, 0.6, 0.0, 0.0),
        (-0.9, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.5, 0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0),
        (-0.8, -0.2, 0.0, 0.0, 0.0, -0.2, 0.3, 0.7, 0.0),
        (-0.9, -0.3, 0.0, 0.0, 0.0, -0.3, 0.4, 0.6, 0.0),
        # With IndepNbrs
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.5, 0.5, 0.5),
        (-0.8, -0.3, 0.0, 0.0, 0.0, -0.3, 0.5, 0.5, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.7),
        (-0.9, -0.3, 0.0, 0.0, 0.0, -0.3, 0.4, 0.4, 0.6),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.5),
    ]

    def compute_scores(w):
        return (w[0] * D_n + w[1] * MeanND_n + w[2] * SND_n +
                w[3] * Tri_n + w[4] * Clus_n + w[5] * Core_n +
                w[6] * AbsFiedler_n + w[7] * SimOrder_n + w[8] * IndepNbrs_n)

    best_count = 0
    best_scores = -D_n.copy()
    best_weights = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for weights in candidates:
        if time.time() > deadline:
            break
        w = np.array(weights)
        scores = compute_scores(w)
        count = _evaluate_fast(nbrs, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()
            best_weights = w.copy()

    # Hill Climbing with adaptive step size
    if time.time() < deadline - 10:
        base_weights = best_weights.copy()
        step_size = 0.15
        steps = 0
        no_improve_count = 0
        while time.time() < deadline - 10:
            steps += 1
            if steps > 800:
                break
            perturbation = np.random.normal(0, step_size, 9)
            new_weights = base_weights + perturbation
            scores = compute_scores(new_weights)
            count = _evaluate_fast(nbrs, scores)
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                best_weights = new_weights.copy()
                base_weights = new_weights.copy()
                no_improve_count = 0
                step_size = min(0.3, step_size * 1.1)
            else:
                no_improve_count += 1
                if no_improve_count > 30:
                    step_size *= 0.7
                    base_weights = best_weights.copy()
                    no_improve_count = 0

    # Efficient local search: try swapping pairs of vertices in the ordering
    if time.time() < deadline - 3:
        order = np.argsort(-best_scores, kind='stable')
        order_list = order.tolist()
        n_order = len(order_list)
        pos_of = {v: i for i, v in enumerate(order_list)}
        
        # Build a score array from order
        def order_to_scores(ord_list):
            sc = np.empty(n, dtype=float)
            for pos, v in enumerate(ord_list):
                sc[v] = float(n_order - pos)
            return sc
        
        current_scores = order_to_scores(order_list)
        current_count = _evaluate_fast(nbrs, current_scores)
        if current_count > best_count:
            best_count = current_count
            best_scores = current_scores.copy()
        
        # Try adjacent swaps and small-range swaps
        improved = True
        max_sweeps = 30
        sweep = 0
        while improved and sweep < max_sweeps and time.time() < deadline - 3:
            improved = False
            sweep += 1
            for i in range(n_order - 1):
                if time.time() > deadline - 3:
                    break
                vi = order_list[i]
                # Try swapping with next few positions
                for j in range(i + 1, min(i + 6, n_order)):
                    if time.time() > deadline - 3:
                        break
                    vj = order_list[j]
                    order_list[i], order_list[j] = vj, vi
                    new_scores = order_to_scores(order_list)
                    count = _evaluate_fast(nbrs, new_scores)
                    if count > best_count:
                        best_count = count
                        best_scores = new_scores.copy()
                        improved = True
                    else:
                        order_list[i], order_list[j] = vi, vj

        # Try moving a vertex to a better position (limited range)
        if time.time() < deadline - 3:
            for _ in range(20):
                if time.time() > deadline - 3:
                    break
                any_improve = False
                for i in range(n_order):
                    if time.time() > deadline - 3:
                        break
                    vi = order_list[i]
                    # Try moving to positions within ±10
                    for j in range(max(0, i - 10), min(n_order, i + 11)):
                        if i == j:
                            continue
                        order_list.pop(i)
                        order_list.insert(j, vi)
                        new_scores = order_to_scores(order_list)
                        count = _evaluate_fast(nbrs, new_scores)
                        if count > best_count:
                            best_count = count
                            best_scores = new_scores.copy()
                            any_improve = True
                        else:
                            order_list.pop(j)
                            order_list.insert(i, vi)
                if not any_improve:
                    break

    return best_scores
# EVOLVE-BLOCK-END
