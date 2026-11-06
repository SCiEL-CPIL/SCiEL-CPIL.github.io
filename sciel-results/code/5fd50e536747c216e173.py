"""Rewritten program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency, order):
    """Compute independent set size given a vertex order (indices sorted by priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            # Vectorized removal: mark v and all its neighbors as unavailable
            available[v] = False
            available[adjacency[v]] = False
    return count


def _score_order(adjacency, priority_vec):
    """Evaluate a priority vector by running the greedy decoder."""
    n = len(priority_vec)
    order = np.argsort(-priority_vec, kind='stable')
    return _greedy_size(adjacency, order)


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute degrees
    deg = A.sum(axis=1).astype(float)
    
    # Base score: negative degree (lower degree = higher priority)
    base_score = -deg.copy()
    
    # Compute neighborhood edge counts for each vertex (vectorized):
    # C[v] = (1/2) * sum_{u,w} A[v,u] * A[u,w] * A[w,v] = (1/2) * (A @ (A @ A))[v,v]
    # This is O(n^3) but done in BLAS; for n=200 it is ~1-2 ms vs ~100+ ms in a Python loop.
    A2 = A.astype(np.int64)
    C = (A2 @ (A2 @ A2)).diagonal().astype(float) * 0.5
    # Guard: ensure non-negative (numerical safety)
    C = np.maximum(C, 0.0)
    # Zero out rows where degree < 2 (no pairs of neighbors to form edges)
    C[deg < 2] = 0.0
    
    # Normalize C to [0, 1] range if possible
    max_C = C.max()
    if max_C > 0:
        C_norm = C / max_C
    else:
        C_norm = C
    
    # Strategy 1: Base + neighborhood sparsity bonus
    # Prefer low degree AND sparse neighborhood
    score1 = base_score - 0.5 * C_norm
    
    # Strategy 2: Pure degree
    score2 = base_score.copy()
    
    # Strategy 3: Degree adjusted by neighborhood density
    # density_v = 2*C[v] / (deg[v]*(deg[v]-1)) if deg[v] >= 2
    density = np.zeros(n)
    mask = deg >= 2
    density[mask] = 2.0 * C[mask] / (deg[mask] * (deg[mask] - 1.0))
    score3 = base_score - density
    
    # Strategy 4: Inverse of (degree + neighborhood_edges)
    score4 = -1.0 / (deg + C + 1e-9)
    
    # Evaluate each strategy
    candidates = [score1, score2, score3, score4]
    
    best_score = -1
    best_vec = base_score.copy()
    
    for s in candidates:
        if time.time() > deadline:
            break
        size = _score_order(adjacency, s)
        if size > best_score:
            best_score = size
            best_vec = s.copy()
    
    # Try informed perturbations: blend base heuristic with random tie-breaking
    # This explores orderings that differ mainly in how ties/low-degree vertices are ordered
    rng = np.random.default_rng(42)
    
    for _ in range(80):
        if time.time() > deadline:
            break
        # Blend: 70% base heuristic + 30% random (scaled to same magnitude)
        w = 0.3 + 0.4 * rng.random()  # weight on random component
        rand_scores = rng.random(n)
        # Scale random to match magnitude of base
        base_mag = np.abs(base_score).max() + 1e-9
        blended = (1 - w) * base_score + w * (rand_scores - 0.5) * 2 * base_mag
        size = _score_order(adjacency, blended)
        if size > best_score:
            best_score = size
            best_vec = blended.copy()
    
    # Local search phase 1: random swaps (as before)
    if time.time() < deadline - 5.0:
        # Get current order
        order = np.argsort(-best_vec, kind='stable')
        current_size = best_score
        
        # Try random swaps
        for _ in range(600):
            if time.time() > deadline:
                break
            
            # Pick two random positions in the order
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            
            # Swap in the order
            order[i], order[j] = order[j], order[i]
            
            # Evaluate
            new_size = _greedy_size(adjacency, order)
            
            if new_size > current_size:
                current_size = new_size
                # Update best_vec to reflect this order
                # Reconstruct priority from order: earlier = higher priority
                new_priority = np.zeros(n)
                for rank, v in enumerate(order):
                    new_priority[v] = n - rank
                best_vec = new_priority
            else:
                # Revert
                order[i], order[j] = order[j], order[i]
    
    # Local search phase 2: use the time saved by the vectorized C computation
    # to run additional swaps with a fresh, fixed-seed RNG for reproducibility.
    # This is the same search family (random-swap local search), just with more
    # iterations enabled by the faster initialization.
    if time.time() < deadline - 2.0:
        rng2 = np.random.default_rng(12345)
        order2 = np.argsort(-best_vec, kind='stable')
        current_size2 = best_score
        # Run until deadline or a generous cap (whichever comes first)
        for _ in range(2000):
            if time.time() > deadline - 1.0:
                break
            i, j = rng2.integers(0, n, size=2)
            if i == j:
                continue
            order2[i], order2[j] = order2[j], order2[i]
            new_size = _greedy_size(adjacency, order2)
            if new_size > current_size2:
                current_size2 = new_size
                new_priority = np.zeros(n)
                for rank, v in enumerate(order2):
                    new_priority[v] = n - rank
                best_vec = new_priority
            else:
                order2[i], order2[j] = order2[j], order2[i]
    
    return best_vec.astype(float)
# EVOLVE-BLOCK-END
