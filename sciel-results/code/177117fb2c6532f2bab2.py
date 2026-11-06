"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_count(order, adj, n):
    """Count independent set size given an ordering (descending priority)."""
    selected = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not selected[v]:
            selected[v] = True
            count += 1
            # Mark neighbors
            if adj[v].any():
                selected[adj[v]] = True
    return count


def _local_search(order, adj, n, time_limit, start_time):
    """Improve ordering via swaps. Returns (best_order, best_count)."""
    best_count = _greedy_count(order, adj, n)
    best_order = list(order)
    current_count = best_count
    
    # Precompute adjacency as list of neighbor sets for speed
    adj_list = [np.where(adj[i])[0] for i in range(n)]
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.1:
        return best_order, best_count
    
    # Try random swaps
    rng = np.random.default_rng(42)
    n_swaps = 0
    
    while time.time() - start_time < time_limit:
        # Pick two random positions to swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        i, j = int(i), int(j)
        
        # Swap
        order[i], order[j] = order[j], order[i]
        new_count = _greedy_count(order, adj, n)
        
        if new_count > current_count:
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_order = list(order)
        else:
            # Undo swap
            order[i], order[j] = order[j], order[i]
        
        n_swaps += 1
        # Check time more frequently
        if n_swaps % 1000 == 0:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
    
    return best_order, best_count


def priority(adjacency):
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Convert to boolean numpy array if needed
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    degrees = adj.sum(axis=1).astype(float)
    
    # Compute effective degree: sum of neighbor degrees
    eff_degrees = (adj @ degrees).astype(float)
    
    # Compute 2-hop neighborhood size
    two_hop = (adj.astype(np.int8) @ adj.astype(np.int8)).astype(float)
    two_hop_degrees = two_hop.sum(axis=1)
    
    # Generate multiple candidate score functions
    candidates = []
    
    # 1. Lowest degree
    candidates.append(-degrees)
    
    # 2. Lowest effective degree
    candidates.append(-eff_degrees)
    
    # 3. Combination: degree + effective degree
    candidates.append(-(degrees + eff_degrees))
    
    # 4. Degree normalized by 2-hop neighborhood
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = degrees / (two_hop_degrees + 1e-9)
    candidates.append(-ratio)
    
    # 5. Effective degree normalized by 2-hop
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio2 = eff_degrees / (two_hop_degrees + 1e-9)
    candidates.append(-ratio2)
    
    # 6. Random perturbation of degree
    rng = np.random.default_rng(42)
    candidates.append(-degrees + rng.normal(0, degrees.std() if degrees.std() > 0 else 1.0, n))
    
    # 7. Random perturbation of effective degree
    candidates.append(-eff_degrees + rng.normal(0, eff_degrees.std() if eff_degrees.std() > 0 else 1.0, n))
    
    # Evaluate candidates and find best
    best_order = None
    best_count = -1
    
    remaining_time = time_limit - (time.time() - start_time)
    time_per_candidate = remaining_time / max(1, len(candidates))
    
    for idx, scores in enumerate(candidates):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 3.0:
            break
        
        # Convert scores to ordering (descending score)
        order = np.argsort(-scores, kind='stable')
        
        # Local search with time budget for this candidate
        ls_time_limit = min(time_per_candidate, remaining_time - 2.0)
        if ls_time_limit < 0.5:
            ls_time_limit = 0.5
        
        order_ls, count_ls = _local_search(order, adj, n, ls_time_limit, start_time + (time.time() - start_time))
        
        if count_ls > best_count:
            best_count = count_ls
            best_order = list(order_ls)
        
        # Check if we're running out of time
        if time.time() - start_time > time_limit - 2.0:
            break
    
    # Convert best ordering to scores
    if best_order is not None:
        scores = np.zeros(n, dtype=float)
        for rank, v in enumerate(best_order):
            scores[v] = -rank
        return scores
    else:
        # Fallback: lowest degree first
        return -degrees
# EVOLVE-BLOCK-END
