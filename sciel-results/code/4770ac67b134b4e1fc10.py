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
        
    sim_order = np.full(n, n, dtype=int) # Default to last
    t = 0
    while np.any(remaining):
        # Find min degree among remaining
        degs = np.where(remaining, rem_deg, np.inf)
        min_deg = np.min(degs)
        
        # Find candidates with min degree
        candidates = np.where(remaining & (rem_deg == min_deg))[0]
        
        for v in candidates:
            sim_order[v] = t
            
        # Remove candidates and update degrees of their neighbors
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

def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    """
    deadline = time.time() + 145
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
        fiedler = _fiedler_fast(L, n, iters=200)
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

    # New Feature: Simulation Order
    sim_order = _compute_sim_order(n, nbrs)
    SimOrder_n = 1.0 - norm(sim_order.astype(float))

    # Features: D, MeanND, SND, Tri, Clus, Core, Fiedler, SimOrder
    # Indices:    0      1        2     3     4      5      6         7
    
    # Precompute feature matrix for fast linear combination
    feat_mat = np.column_stack([D_n, MeanND_n, SND_n, Tri_n, Clus_n, Core_n, AbsFiedler_n, SimOrder_n])
    
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 1.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 1.0, 0.0),
        (-0.8, -0.3, 0.0, 0.0, 0.0, -0.3, 1.0, 0.0),
        (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
        (-1.0, -0.3, -0.2, 0.0, 0.0, -0.3, 0.5, 0.0),
        (-0.9, -0.4, -0.3, -0.2, -0.2, -0.4, 0.8, 0.0),
        (-0.8, -0.4, -0.4, -0.3, -0.3, -0.4, 0.7, 0.0),
        (-1.0, -0.6, 0.0, 0.0, 0.0, -0.6, 0.6, 0.0),
        (-0.9, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.5, 0.5),
        (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 1.0),
        (-0.8, -0.2, 0.0, 0.0, 0.0, -0.2, 0.3, 0.7),
        (-0.9, -0.3, 0.0, 0.0, 0.0, -0.3, 0.4, 0.6),
        (-1.0, -0.4, -0.2, 0.0, 0.0, -0.2, 0.3, 0.7),
        (-0.7, -0.3, -0.2, 0.0, -0.2, -0.3, 0.4, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.3, 0.7),
        (-1.0, -0.5, -0.3, -0.2, -0.1, -0.4, 0.5, 0.3),
        (-0.8, -0.2, 0.0, -0.2, 0.0, -0.2, 0.4, 0.6),
        (-1.0, -0.6, -0.4, -0.3, -0.3, -0.5, 0.6, 0.2),
        # Additional candidates emphasizing low-degree and low-core
        (-1.2, -0.3, 0.0, 0.0, 0.0, -0.5, 0.3, 0.5),
        (-1.0, -0.4, -0.3, -0.2, -0.2, -0.6, 0.4, 0.4),
        (-0.9, -0.5, -0.4, -0.3, -0.3, -0.5, 0.5, 0.3),
        (-1.1, -0.2, -0.2, 0.0, 0.0, -0.4, 0.2, 0.6),
    ]

    best_count = 0
    best_scores = -D_n.copy()
    best_weights = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for weights in candidates:
        if time.time() > deadline:
            break
        w = np.array(weights)
        scores = feat_mat @ w
        count = _evaluate_fast(nbrs, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()
            best_weights = w.copy()

    # Improved Hill Climbing with momentum and adaptive step size
    if time.time() < deadline - 8:
        base_weights = best_weights.copy()
        step_size = 0.08
        momentum = 0.0
        steps = 0
        no_improve_count = 0
        max_steps = 2000
        while time.time() < deadline - 8 and steps < max_steps:
            steps += 1
            # Adaptive step size: shrink on success, expand on stagnation
            if no_improve_count > 10:
                step_size = min(0.3, step_size * 1.3)
            elif no_improve_count == 0:
                step_size = max(0.02, step_size * 0.85)
            
            # Momentum-based perturbation
            perturbation = np.random.normal(0, step_size, 8)
            new_weights = base_weights + momentum + perturbation
            
            scores = feat_mat @ new_weights
            count = _evaluate_fast(nbrs, scores)
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                best_weights = new_weights.copy()
                # Update momentum toward successful direction
                momentum = 0.5 * (new_weights - base_weights)
                base_weights = new_weights.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Decay momentum
                momentum *= 0.7
                # Reset to best if stuck
                if no_improve_count > 20 or np.random.random() < 0.02:
                    base_weights = best_weights.copy()
                    momentum = 0.0
                    no_improve_count = 0

    # Final Polish: Multi-round local improvement
    if time.time() < deadline - 2:
        scores = best_scores.copy()
        
        # Precompute adjacency set for quick lookup
        adj_set = set()
        for i in range(n):
            for j in nbrs[i]:
                adj_set.add((min(i,j), max(i,j)))
        
        # Round 1: Targeted pair swaps
        improved_any = True
        max_sweeps = 30
        sweep = 0
        while improved_any and sweep < max_sweeps and time.time() < deadline - 2:
            improved_any = False
            sweep += 1
            n_try = min(n, 60)
            try_indices = np.random.choice(n, size=n_try, replace=False)
            for i_idx in try_indices:
                if time.time() > deadline - 2:
                    break
                i = int(i_idx)
                for _ in range(4):
                    j = int(np.random.randint(n))
                    if i == j:
                        continue
                    if (min(i,j), max(i,j)) in adj_set:
                        continue
                    scores[i], scores[j] = scores[j], scores[i]
                    count = _evaluate_fast(nbrs, scores)
                    if count > best_count:
                        best_count = count
                        best_scores = scores.copy()
                        improved_any = True
                    else:
                        scores[i], scores[j] = scores[j], scores[i]
        
        scores = best_scores.copy()
        
        # Round 2: Local neighborhood score adjustments
        if time.time() < deadline - 1:
            for sweep in range(10):
                if time.time() > deadline - 1:
                    break
                improved = False
                n_try = min(n, 80)
                try_indices = np.random.choice(n, size=n_try, replace=False)
                for i_idx in try_indices:
                    i = int(i_idx)
                    # Try small score adjustments
                    for delta in [0.01, -0.01, 0.02, -0.02]:
                        old_val = scores[i]
                        scores[i] = old_val + delta
                        count = _evaluate_fast(nbrs, scores)
                        if count > best_count:
                            best_count = count
                            best_scores = scores.copy()
                            improved = True
                        else:
                            scores[i] = old_val
                if not improved:
                    break

    return best_scores
# EVOLVE-BLOCK-END
