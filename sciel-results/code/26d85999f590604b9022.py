"""Combined program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency, order, neighbor_lists=None):
    """Compute independent set size given a vertex order (indices sorted by priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            if neighbor_lists is not None:
                for u in neighbor_lists[v]:
                    available[u] = False
            else:
                neighbors = adjacency[v].astype(bool)
                available[neighbors] = False
    return count


def _score_order(adjacency, priority_vec, neighbor_lists=None):
    """Evaluate a priority vector by running the greedy decoder."""
    n = len(priority_vec)
    order = np.argsort(-priority_vec, kind='stable')
    return _greedy_size(adjacency, order, neighbor_lists)


def _local_search(adjacency, order, neighbor_lists, rng, max_iter, deadline):
    """Local search via adjacent transpositions and random swaps."""
    n = len(order)
    current_size = _greedy_size(adjacency, order, neighbor_lists)
    
    # Adjacent transposition: swap order[i] and order[i+1]
    # This is a minimal change, so we can quickly evaluate
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Mix of adjacent and random swaps
        if rng.random() < 0.7:
            # Adjacent transposition
            i = rng.integers(0, n - 1)
            j = i + 1
        else:
            # Random swap
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
        
        # Swap
        order[i], order[j] = order[j], order[i]
        
        new_size = _greedy_size(adjacency, order, neighbor_lists)
        
        if new_size > current_size:
            current_size = new_size
        else:
            # Revert
            order[i], order[j] = order[j], order[i]
    
    return current_size, order


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n = adjacency.shape[0]
    A = adjacency.astype(bool)

    # Precompute degrees
    deg = A.sum(axis=1).astype(float)

    # Precompute neighbor lists for faster greedy evaluation
    neighbor_lists = [np.where(A[v])[0] for v in range(n)]

    # Compute neighborhood edge counts for each vertex
    C = np.zeros(n)
    for v in range(n):
        if deg[v] < 2:
            C[v] = 0.0
        else:
            nbrs = np.where(A[v])[0]
            if len(nbrs) > 0:
                sub = A[np.ix_(nbrs, nbrs)]
                C[v] = sub.sum() / 2.0

    # Normalize C
    max_C = C.max()
    if max_C > 0:
        C_norm = C / max_C
    else:
        C_norm = C

    # Density: 2*C / (d*(d-1))
    density = np.zeros(n)
    mask = deg >= 2
    density[mask] = 2.0 * C[mask] / (deg[mask] * (deg[mask] - 1.0))

    # Mean neighbor degree
    neighbor_deg_sum = A @ deg
    mean_neighbor_deg = np.zeros(n)
    mask_pos = deg > 0
    mean_neighbor_deg[mask_pos] = neighbor_deg_sum[mask_pos] / deg[mask_pos]

    # Build multiple candidate scoring strategies
    base_score = -deg.copy()

    score1 = base_score - 0.5 * C_norm
    score2 = base_score.copy()
    score3 = base_score - density
    score4 = -1.0 / (deg + C + 1e-9)
    score5 = -deg + 1.5 * density
    score6 = -deg + 0.1 * mean_neighbor_deg
    score7 = -deg + 1.5 * density + 0.1 * mean_neighbor_deg
    score8 = -deg + 2.0 * density
    score9 = -deg + 1.0 * density
    score10 = -deg - 0.3 * C_norm + 0.1 * mean_neighbor_deg

    candidates = [score1, score2, score3, score4, score5, score6, score7, score8, score9, score10]

    # Evaluate all candidates
    best_size = -1
    best_order = None

    for s in candidates:
        if time.time() > deadline:
            break
        order = np.argsort(-s, kind='stable')
        size = _greedy_size(adjacency, order, neighbor_lists)
        if size > best_size:
            best_size = size
            best_order = order.copy()

    # Local search from the best initial order
    rng = np.random.default_rng(42)
    
    # Phase 1: Local search from best candidate
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Allocate most time to local search
        max_iter = min(500, int(remaining_time * 20))  # ~20 iters/sec estimate
        best_size, best_order = _local_search(
            adjacency, best_order, neighbor_lists, rng, max_iter, deadline
        )

    # Phase 2: Try a few more initial orderings with shorter local search
    if time.time() < deadline - 15.0:
        # Try different random seeds for initial orderings
        for seed in [1, 7, 13]:
            if time.time() > deadline - 10.0:
                break
            rng2 = np.random.default_rng(seed)
            # Create a random-ish ordering biased by degree
            noise = rng2.normal(0, 0.5, size=n)
            perturbed = base_score + noise
            order = np.argsort(-perturbed, kind='stable')
            
            # Short local search
            remaining = deadline - time.time() - 10.0
            if remaining < 3.0:
                break
            max_iter = min(100, int(remaining * 10))
            size, order = _local_search(
                adjacency, order, neighbor_lists, rng2, max_iter, deadline - 5.0
            )
            
            if size > best_size:
                best_size = size
                best_order = order

    # Reconstruct priority vector from best order
    best_vec = np.zeros(n)
    for rank, v in enumerate(best_order):
        best_vec[v] = n - rank

    return best_vec.astype(float)
# EVOLVE-BLOCK-END
