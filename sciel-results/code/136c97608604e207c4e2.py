"""Improved program for Maximum Independent Set greedy scoring.
COMBINED approach: 
- Structural framework from current program (spectral scoring, multi-start local search, final polish)
- Refinement from inspiration (simpler, more focused heuristic generation, tighter local search)
Key improvements:
1. Faster local search with incremental evaluation
2. Better initial candidates including 2-hop neighborhood awareness
3. Adaptive local search with multiple move types
4. Time-aware budget management
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _greedy_size(adjacency_bool, order):
    """Compute independent set size given a vertex order (indices sorted by priority, ascending)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency_bool[v]] = False
    return count


def _greedy_size_incremental(adjacency_bool, order, pos_i, pos_j, current_size, current_available):
    """Fast evaluation: only check if swapping positions i and j changes the result.
    This is an approximation - we recompute from the earlier position.
    """
    n = len(order)
    # We need to recompute from min(pos_i, pos_j) onwards
    start = min(pos_i, pos_j)
    
    # Rebuild available state up to start
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order[:start]:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency_bool[v]] = False
    
    # Create new order with swap
    new_order = order.copy()
    new_order[pos_i], new_order[pos_j] = order[pos_j], order[pos_i]
    
    # Continue from start
    for v in new_order[start:]:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency_bool[v]] = False
    
    return count


def _spectral_scores(A, n, rng, num_iters=40):
    """Generate scores using power iteration on a modified adjacency matrix."""
    alpha = 0.8
    M = np.eye(n) - alpha * A.astype(float)
    
    x = rng.random(n)
    x = x / (np.linalg.norm(x) + 1e-10)
    
    for _ in range(num_iters):
        x_new = M @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            break
        x = x_new / norm
    
    deg = A.sum(axis=1).astype(float)
    deg_norm = deg / (deg.max() + 1e-10)
    
    x_mean = x.mean()
    x_std = x.std() + 1e-10
    x_norm = (x - x_mean) / x_std
    
    combined = deg_norm - 0.3 * x_norm
    return combined


def _two_hop_score(A, deg, n):
    """Score based on 2-hop neighborhood: prefer vertices whose neighbors have high degree."""
    score = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            # Sum of degrees of neighbors
            nbr_deg_sum = deg[nbrs].sum()
            score[v] = deg[v] + 0.5 * nbr_deg_sum
        else:
            score[v] = deg[v]
    return score


def _core_order(A, deg, n):
    """Generate order based on core decomposition (peel lowest degree first)."""
    order = np.zeros(n, dtype=int)
    remaining = set(range(n))
    deg_copy = deg.copy().astype(int)
    
    for pos in range(n):
        if not remaining:
            break
        min_deg = min(deg_copy[v] for v in remaining)
        candidates = [v for v in remaining if deg_copy[v] == min_deg]
        v = random.choice(candidates)
        order[pos] = v
        remaining.remove(v)
        for u in np.where(A[v])[0]:
            if u in remaining:
                deg_copy[u] -= 1
    
    return order


def _generate_heuristic_orders(A, deg, n, rng):
    """Generate a diverse set of candidate orders using different heuristics."""
    orders = []
    
    # 1. Ascending degree (classic greedy for MIS)
    orders.append(np.argsort(deg, kind='stable'))
    
    # 2. Ascending degree with random tie-breaking
    noise = rng.random(n) * 0.5
    orders.append(np.argsort(deg + noise, kind='stable'))
    
    # 3. Pure random
    orders.append(rng.permutation(n))
    
    # 4. Based on average neighbor degree
    if n > 0:
        avg_nbr_deg = np.zeros(n)
        for v in range(n):
            nbrs = np.where(A[v])[0]
            if len(nbrs) > 0:
                avg_nbr_deg[v] = deg[nbrs].mean()
            else:
                avg_nbr_deg[v] = 0.0
        combined = deg + 0.5 * avg_nbr_deg
        orders.append(np.argsort(combined, kind='stable'))
    
    # 5. Core number approximation
    core_order = _core_order(A, deg, n)
    orders.append(core_order)
    
    # 6. Weighted random with degree bias
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    orders.append(perm)
    
    # 7. Two-hop neighborhood score
    two_hop = _two_hop_score(A, deg, n)
    orders.append(np.argsort(two_hop, kind='stable'))
    
    # 8. Multiple random perturbations of degree order
    base_order = np.argsort(deg, kind='stable')
    for _ in range(2):
        perturbed = base_order.copy()
        for _ in range(n // 10):
            i, j = rng.integers(0, n, size=2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        orders.append(perturbed)
    
    # 9. Spectral-based ordering
    spectral_scores = _spectral_scores(A, n, rng, num_iters=25)
    orders.append(np.argsort(spectral_scores, kind='stable'))
    
    # 10. Boost low-degree vertices with high-degree neighbors
    nbr_max_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            nbr_max_deg[v] = deg[nbrs].max()
        else:
            nbr_max_deg[v] = 0.0
    boost_score = deg - 0.3 * nbr_max_deg
    orders.append(np.argsort(boost_score, kind='stable'))
    
    return orders


def _local_search(A, start_order, n, rng, deadline):
    """Perform local search starting from start_order. Returns best order and size."""
    current_order = start_order.copy()
    current_size = _greedy_size(A, current_order)
    best_order = current_order.copy()
    best_size = current_size
    no_improve = 0
    max_no_improve = 300
    
    # Precompute neighbor sets for faster access
    neighbor_sets = [np.where(A[v])[0] for v in range(n)]
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        # Choose move type: swap, insert, or reverse
        move_type = rng.integers(0, 3)
        
        if move_type == 0:  # Swap
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            new_order = current_order.copy()
            new_order[i], new_order[j] = current_order[j], current_order[i]
            
        elif move_type == 1:  # Move vertex to new position
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            new_order = list(current_order)
            v = new_order.pop(i)
            new_order.insert(j, v)
            new_order = np.array(new_order)
            
        else:  # Reverse a segment
            i = rng.integers(0, n - 1)
            j = rng.integers(i + 1, n)
            new_order = current_order.copy()
            new_order[i:j+1] = new_order[i:j+1][::-1]
        
        new_size = _greedy_size(A, new_order)
        
        if new_size > current_size:
            current_order = new_order
            current_size = new_size
            no_improve = 0
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            no_improve += 1
    
    return best_order, best_size


def _targeted_polish(A, best_order, n, rng, deadline):
    """Try targeted swaps around the best solution to find small improvements."""
    current_order = best_order.copy()
    current_size = _greedy_size(A, current_order)
    best_order = current_order.copy()
    best_size = current_size
    
    while time.time() < deadline:
        # Try swapping adjacent pairs
        i = rng.integers(0, n - 1)
        new_order = current_order.copy()
        new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
        
        new_size = _greedy_size(A, new_order)
        
        if new_size > current_size:
            current_order = new_order
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            # Try a random long-range swap
            j = rng.integers(0, n)
            new_order = current_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_size = _greedy_size(A, new_order)
            if new_size > current_size:
                current_order = new_order
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
    
    return best_order, best_size


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute degrees
    deg = A.sum(axis=1).astype(float)
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse candidate orders
    candidates = _generate_heuristic_orders(A, deg, n, rng)
    
    best_size = -1
    best_order = None
    
    # Evaluate all initial candidates
    for order in candidates:
        if time.time() > deadline - 30:
            break
        size = _greedy_size(A, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
    
    if best_order is None:
        best_order = np.argsort(deg, kind='stable')
        best_size = _greedy_size(A, best_order)
    
    # Local search from best initial candidate
    ls_deadline = deadline - 10.0
    best_order, best_size = _local_search(A, best_order, n, rng, ls_deadline)
    
    # If time remains, do additional restarts from other candidates
    if time.time() < ls_deadline - 15:
        other_starts = [candidates[0], candidates[2], candidates[6]] if len(candidates) >= 7 else [candidates[0]]
        for start in other_starts:
            if time.time() > ls_deadline - 5:
                break
            o, s = _local_search(A, start, n, rng, ls_deadline)
            if s > best_size:
                best_size = s
                best_order = o
    
    # Final polish
    if time.time() < deadline - 3:
        best_order, best_size = _targeted_polish(A, best_order, n, rng, deadline - 1)
    
    # Convert best_order to priority vector
    # Higher priority means selected earlier.
    # Vertex at index 0 in order gets highest priority.
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
    
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
