"""Combined program for Maximum Independent Set greedy scoring.
Uses dynamic heuristics and spectral methods for initial orders,
with fast incremental local search (swap and insertion moves).
Incorporates multi-start hill climbing with diverse restart strategies
and Fiedler vector spectral signals from inspiration.
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

def _greedy_size_incremental(A, order, pos1, pos2):
    """Fast evaluation of a single swap at positions pos1 and pos2 in the order.
    Only re-simulates from min(pos1, pos2) onward.
    """
    n = len(order)
    lo = min(pos1, pos2)
    if lo >= n:
        return -1

    available = np.ones(n, dtype=bool)
    count = 0
    for i in range(lo):
        v = order[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False

    v1, v2 = order[pos1], order[pos2]
    for i in range(lo, n):
        if i == pos1:
            v = v2
        elif i == pos2:
            v = v1
        else:
            v = order[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False
    return count

def _greedy_size_insertion(A, order, old_pos, new_pos):
    """Evaluate moving element from old_pos to new_pos.
    Re-simulates from min(old_pos, new_pos) onward.
    """
    n = len(order)
    lo = min(old_pos, new_pos)
    if lo >= n:
        return -1

    # Build the modified order
    elem = order[old_pos]
    modified = list(order)
    del modified[old_pos]
    modified.insert(new_pos, elem)

    available = np.ones(n, dtype=bool)
    count = 0
    for i in range(lo):
        v = modified[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False

    for i in range(lo, n):
        v = modified[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False
    return count

def _dynamic_min_degree_order(A, n):
    """Generate order using dynamic min-degree heuristic."""
    available = np.ones(n, dtype=bool)
    order = np.zeros(n, dtype=int)
    deg = A.sum(axis=1).astype(int)
    
    for pos in range(n):
        if not available.any():
            break
        cur_deg = np.where(available, deg, np.inf)
        v = np.argmin(cur_deg)
        order[pos] = v
        available[v] = False
        neighbors = np.where(A[v])[0]
        deg[neighbors] -= 1
        
    return order

def _spectral_priority(A, n):
    """
    Generate spectral scores using power iteration on the adjacency matrix.
    Vertices with high principal eigenvector values are in dense regions.
    For MIS greedy, we want to pick peripheral (low centrality) vertices first.
    """
    A_float = A.astype(float)
    rng = np.random.default_rng(0)
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    for _ in range(80):
        new_v = A_float @ v
        norm = np.linalg.norm(new_v)
        if norm < 1e-12:
            break
        new_v /= norm
        if np.abs(np.dot(new_v, v)) > 1 - 1e-8:
            v = new_v
            break
        v = new_v
    principal = np.abs(v)
    
    p_min = principal.min()
    p_max = principal.max()
    if p_max > p_min:
        principal_norm = (principal - p_min) / (p_max - p_min)
    else:
        principal_norm = np.zeros(n)
    
    return principal_norm

def _laplacian_fiedler(A, n):
    """
    Compute the Fiedler vector (second smallest eigenvalue eigenvector of L).
    """
    deg = A.sum(axis=1).astype(float)
    L = np.diag(deg) - A.astype(float)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eig(L)
        idx = np.argsort(eigenvalues.real)
        fiedler = np.abs(eigenvectors[:, idx[1]])
    except:
        fiedler = np.ones(n) / n
    
    f_min = fiedler.min()
    f_max = fiedler.max()
    if f_max > f_min:
        fiedler_norm = (fiedler - f_min) / (f_max - f_min)
    else:
        fiedler_norm = np.zeros(n)
    
    return fiedler_norm

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
    """Generate a diverse set of candidate orders."""
    orders = []
    
    # 1. Static Ascending Degree
    orders.append(np.argsort(deg, kind='stable'))
    
    # 2. Static Ascending Degree with Noise
    noise = rng.random(n) * 0.5
    orders.append(np.argsort(deg + noise, kind='stable'))
    
    # 3. Dynamic Min-Degree
    orders.append(_dynamic_min_degree_order(A, n))
    
    # 4. Spectral (Low Centrality first)
    spectral = _spectral_priority(A, n)
    score_h5 = deg * 0.5 + spectral * 0.5
    orders.append(np.argsort(score_h5, kind='stable'))
    
    # 5. Fiedler-based - vertices with extreme Fiedler values
    fiedler = _laplacian_fiedler(A, n)
    score_h6 = deg * 0.5 + fiedler * 0.5
    orders.append(np.argsort(score_h6, kind='stable'))
    
    # 6. Random
    orders.append(rng.permutation(n))
    
    # 7. Core Number Order
    orders.append(_core_order(A, deg, n))
    
    # 8. Combined - degree, spectral, fiedler
    score_h7 = deg * 0.4 + spectral * 0.3 + fiedler * 0.3
    orders.append(np.argsort(score_h7, kind='stable'))
    
    # 9. 2-hop neighborhood size
    two_hop = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) == 0:
            two_hop[v] = 0
        else:
            two_hop_nbrs = set()
            for u in nbrs:
                two_hop_nbrs.update(np.where(A[u])[0])
            two_hop_nbrs.discard(v)
            two_hop_nbrs.difference_update(nbrs)
            two_hop[v] = len(two_hop_nbrs)
    orders.append(np.argsort(two_hop, kind='stable'))
    
    # 10. Weighted random with degree bias
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    orders.append(perm)
    
    # 11. Perturbed degree order
    base_order = np.argsort(deg, kind='stable')
    perturbed = base_order.copy()
    for _ in range(n // 10):
        i, j = rng.integers(0, n, size=2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
    orders.append(perturbed)
    
    # 12. Neighbor clustering score: prefer vertices whose neighbors are densely connected
    neighbor_clustering = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) < 2:
            neighbor_clustering[v] = 0
        else:
            sub_A = A[np.ix_(nbrs, nbrs)]
            edges = sub_A.sum() / 2
            possible = len(nbrs) * (len(nbrs) - 1) / 2
            neighbor_clustering[v] = edges / possible if possible > 0 else 0
    
    # Combine: low degree + high neighbor clustering is good
    deg_norm = deg / (deg.max() + 1e-10)
    cluster_norm = neighbor_clustering / (neighbor_clustering.max() + 1e-10)
    combined_score = deg_norm - 0.5 * cluster_norm
    orders.append(np.argsort(combined_score, kind='stable'))
    
    # 13. Mean neighbor degree signal
    mean_nbr_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            mean_nbr_deg[v] = deg[nbrs].mean()
        else:
            mean_nbr_deg[v] = 0
    
    mean_nbr_norm = mean_nbr_deg / (mean_nbr_deg.max() + 1e-10)
    deg_norm2 = deg / (deg.max() + 1e-10)
    combined2 = deg_norm2 + 0.3 * mean_nbr_norm
    orders.append(np.argsort(combined2, kind='stable'))
    
    # 14. Inverse neighbor degree heuristic (local expected-loss)
    # Prioritize vertices whose neighbors are high-degree (less likely to be in MIS)
    inv_nbr_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            inv_nbr_deg[v] = np.sum(1.0 / (deg[nbrs] + 1e-10))
        else:
            inv_nbr_deg[v] = 0
    
    inv_norm = inv_nbr_deg / (inv_nbr_deg.max() + 1e-10)
    deg_norm3 = deg / (deg.max() + 1e-10)
    combined3 = deg_norm3 - 0.4 * inv_norm
    orders.append(np.argsort(combined3, kind='stable'))
    
    return orders

def _local_search(A, start_order, n, rng, deadline):
    """Perform local search with swap and insertion moves using incremental evaluation.
    Uses adaptive move ordering and early termination.
    """
    current_order = start_order.copy()
    current_size = _greedy_size(A, current_order)
    best_order = current_order.copy()
    best_size = current_size
    no_improve = 0
    max_no_improve = 200  # Reduced to allow more restarts
    
    # Track recent moves for bias
    recent_positions = {}
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        # Choose move type: swap (65%) or insertion (35%)
        if rng.random() < 0.65:
            # Swap move with position bias toward recent active positions
            if recent_positions and rng.random() < 0.3:
                # Sample from recent positions
                pos_list = list(recent_positions.keys())
                i = pos_list[rng.integers(0, len(pos_list))]
                j = rng.integers(0, n)
            else:
                i = rng.integers(0, n)
                j = rng.integers(0, n)
            
            if i == j:
                continue
            
            new_size = _greedy_size_incremental(A, current_order, i, j)
            
            if new_size > current_size:
                current_order[i], current_order[j] = current_order[j], current_order[i]
                current_size = new_size
                no_improve = 0
                # Update recent positions
                recent_positions[i] = time.time()
                recent_positions[j] = time.time()
                # Prune old entries
                cutoff = time.time() - 2.0
                for k in list(recent_positions.keys()):
                    if recent_positions[k] < cutoff:
                        del recent_positions[k]
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                no_improve += 1
        else:
            # Insertion move
            old_pos = rng.integers(0, n)
            new_pos = rng.integers(0, n)
            if old_pos == new_pos:
                continue
            
            new_size = _greedy_size_insertion(A, current_order, old_pos, new_pos)
            
            if new_size > current_size:
                # Apply the move
                elem = current_order[old_pos]
                current_order = np.delete(current_order, old_pos)
                current_order = np.insert(current_order, new_pos, elem)
                current_size = new_size
                no_improve = 0
                recent_positions[old_pos] = time.time()
                recent_positions[new_pos] = time.time()
                cutoff = time.time() - 2.0
                for k in list(recent_positions.keys()):
                    if recent_positions[k] < cutoff:
                        del recent_positions[k]
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                no_improve += 1

    return best_order, best_size

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    deg = A.sum(axis=1).astype(float)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse candidate orders
    candidates = _generate_heuristic_orders(A, deg, n, rng)
    
    best_size = -1
    best_order = None
    top_candidates = []
    
    # Evaluate all initial candidates
    for order in candidates:
        if time.time() > deadline - 40:
            break
        size = _greedy_size(A, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
        top_candidates.append((size, order))
        
    if best_order is None:
        best_order = np.argsort(deg, kind='stable')
        best_size = _greedy_size(A, best_order)

    # Sort candidates by size to pick the best ones for local search
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Local search from the best candidates
    ls_deadline = deadline - 5.0
    
    # 1. Local search from the absolute best
    best_order, best_size = _local_search(A, best_order, n, rng, ls_deadline)
    
    # 2. Multi-start hill climbing with diverse restart strategies
    num_restarts = 20
    max_iters_per_restart = 2000
    
    for r in range(num_restarts):
        if time.time() > ls_deadline - 10:
            break
            
        # Initialize current order for this restart
        if r % 4 == 0:
            # Start from best with small perturbation
            current_order = best_order.copy()
            for _ in range(5):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        elif r % 4 == 1:
            # Perturb the best order found so far more aggressively
            current_order = best_order.copy()
            for _ in range(15):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        elif r % 4 == 2:
            # Start from a random heuristic
            current_order = candidates[r % len(candidates)].copy()
            for _ in range(8):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        else:
            # Start from a new random order
            current_order = rng.permutation(n)
                
        # Run local search from this starting point
        restart_deadline = min(ls_deadline - 5, time.time() + 5.0)
        o_r, s_r = _local_search(A, current_order, n, rng, restart_deadline)
        
        if s_r > best_size:
            best_size = s_r
            best_order = o_r

    # Convert best_order to priority vector
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
        
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
