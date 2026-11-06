"""Improved program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size_fast(adj_mask, order):
    """
    Fast greedy independent set size using integer bitmasks.
    adj_mask: list of int bitmasks, adj_mask[v] has bit u set if v-u is an edge.
    order: array of vertex indices in selection order.
    """
    n = len(order)
    available = (1 << n) - 1  # all vertices available
    count = 0
    for v in order:
        bit = 1 << v
        if available & bit:
            count += 1
            # Remove v and all its neighbors from available
            available &= ~(bit | adj_mask[v])
    return count


def _greedy_size_incremental(adj_mask, order, start_idx):
    """
    Re-simulate greedy from start_idx, assuming all vertices before start_idx
    are already processed. Returns (count_from_start, final_available_mask).
    """
    n = len(order)
    available = (1 << n) - 1
    count = 0
    # First process vertices before start_idx to get the available state
    for i in range(start_idx):
        v = order[i]
        bit = 1 << v
        if available & bit:
            count += 1
            available &= ~(bit | adj_mask[v])
    # Now process from start_idx
    for i in range(start_idx, n):
        v = order[i]
        bit = 1 << v
        if available & bit:
            count += 1
            available &= ~(bit | adj_mask[v])
    return count


def _power_iteration_eigenvector(M, n, iters=50, tol=1e-8):
    """
    Fast power iteration for the principal eigenvector of a symmetric matrix.
    Returns the eigenvector (normalized to unit L2 norm).
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        new_v = M @ v
        norm = np.linalg.norm(new_v)
        if norm < 1e-12:
            break
        new_v /= norm
        if np.abs(np.dot(new_v, v)) > 1 - tol:
            v = new_v
            break
        v = new_v
    return v


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Build bitmask representation for fast greedy
    adj_mask = np.zeros(n, dtype=object)
    for v in range(n):
        mask = 0
        neighbors = np.where(A[v])[0]
        for u in neighbors:
            mask |= (1 << u)
        adj_mask[v] = mask
    adj_mask_list = adj_mask.tolist()
    
    # Precompute degrees
    deg = A.sum(axis=1).astype(float)
    
    # Compute spectral scores using power iteration on adjacency matrix
    A_float = A.astype(float)
    principal = _power_iteration_eigenvector(A_float, n, iters=50)
    principal = np.abs(principal)
    
    # Normalize spectral scores
    p_min = principal.min()
    p_max = principal.max()
    if p_max > p_min:
        spectral_norm = (principal - p_min) / (p_max - p_min)
    else:
        spectral_norm = np.zeros(n)
        
    # Generate candidate orders
    rng = np.random.default_rng(42)
    
    # Heuristic 1: Ascending Degree
    order1 = np.argsort(deg, kind='stable')
    
    # Heuristic 2: Spectral (low spectral score first, i.e., peripheral)
    order2 = np.argsort(spectral_norm, kind='stable')
    
    # Heuristic 3: Combined Degree and Spectral
    score_h3 = deg / (deg.max() + 1e-6) * 0.5 + spectral_norm * 0.5
    order3 = np.argsort(score_h3, kind='stable')
    
    # Heuristic 4: Random
    order4 = rng.permutation(n)
    
    # Heuristic 5: Degree with noise
    noise = rng.normal(0, 0.5, size=n)
    score_h5 = deg + noise
    order5 = np.argsort(score_h5, kind='stable')
    
    # Heuristic 6: Effective degree (degree / (1 + avg neighbor degree))
    avg_neighbor_deg = np.zeros(n)
    for v in range(n):
        neighbors = np.where(A[v])[0]
        if len(neighbors) > 0:
            avg_neighbor_deg[v] = deg[neighbors].mean()
    eff_deg = deg / (1.0 + avg_neighbor_deg)
    order6 = np.argsort(eff_deg, kind='stable')
    
    # Heuristic 7: 2-hop neighborhood size
    two_hop = np.zeros(n)
    for v in range(n):
        neighbors = np.where(A[v])[0]
        two_hop[v] = len(np.unique(np.concatenate([neighbors, np.where(A[neighbors].any(axis=1))[0]])) if len(neighbors) > 0 else [])
    order7 = np.argsort(two_hop, kind='stable')
    
    candidates = [order1, order2, order3, order4, order5, order6, order7]
    
    best_size = -1
    best_order = None
    
    # Evaluate initial candidates
    for order in candidates:
        if time.time() > deadline:
            break
        size = _greedy_size_fast(adj_mask_list, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
            
    # Local Search: Multi-start Hill Climbing with incremental evaluation
    num_restarts = 60
    max_iters_per_restart = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize current order for this restart
        if r == 0:
            current_order = best_order.copy()
        elif r % 4 == 1:
            # Perturb the best order found so far
            current_order = best_order.copy()
            for _ in range(8):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        elif r % 4 == 2:
            # Start from a random heuristic candidate
            current_order = candidates[r % len(candidates)].copy()
            for _ in range(5):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        elif r % 4 == 3:
            # Start from a new random order
            current_order = rng.permutation(n)
        else:
            # Start from best with more perturbation
            current_order = best_order.copy()
            for _ in range(20):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
        current_size = _greedy_size_fast(adj_mask_list, current_order)
        
        # If this restart starts better than global best, update
        if current_size > best_size:
            best_size = current_size
            best_order = current_order.copy()
            
        no_improve_count = 0
        max_no_improve = 100
        
        for i in range(max_iters_per_restart):
            if time.time() > deadline - 3:
                break
                
            if no_improve_count > max_no_improve:
                break
                
            # Move: Swap two adjacent positions (fast incremental eval)
            pos = rng.integers(0, n - 1)
            
            # Try swapping positions pos and pos+1
            current_order[pos], current_order[pos+1] = current_order[pos+1], current_order[pos]
            
            new_size = _greedy_size_fast(adj_mask_list, current_order)
            
            if new_size > current_size:
                current_size = new_size
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                # Undo the swap
                current_order[pos], current_order[pos+1] = current_order[pos+1], current_order[pos]
                no_improve_count += 1
                
            # Also try a random insertion move occasionally (every 5th iteration)
            if i % 5 == 4 and time.time() < deadline - 3:
                v_idx = rng.integers(0, n)
                target_pos = rng.integers(0, n)
                
                if v_idx != target_pos:
                    v = current_order[v_idx]
                    
                    # Remove v from its position
                    temp = current_order.copy()
                    temp = np.delete(temp, v_idx)
                    if v_idx < target_pos:
                        insert_idx = target_pos - 1
                    else:
                        insert_idx = target_pos
                    temp = np.insert(temp, insert_idx, v)
                    
                    new_size = _greedy_size_fast(adj_mask_list, temp)
                    
                    if new_size > current_size:
                        current_order = temp
                        current_size = new_size
                        no_improve_count = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_order = current_order.copy()
                    else:
                        no_improve_count += 1
                
    # Final polish: try adjacent swaps around the best order
    if time.time() < deadline - 1:
        polish_order = best_order.copy()
        polish_size = _greedy_size_fast(adj_mask_list, polish_order)
        for _ in range(2000):
            if time.time() > deadline - 0.5:
                break
            i = rng.integers(0, n - 1)
            polish_order[i], polish_order[i+1] = polish_order[i+1], polish_order[i]
            swap_size = _greedy_size_fast(adj_mask_list, polish_order)
            if swap_size > polish_size:
                polish_size = swap_size
                if polish_size > best_size:
                    best_size = polish_size
                    best_order = polish_order.copy()
            else:
                polish_order[i], polish_order[i+1] = polish_order[i+1], polish_order[i]
                
    # Convert best_order to priority vector
    # Higher priority means selected earlier.
    # So vertex at index 0 in order gets highest priority.
    # priority[v] = n - index(v)
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
        
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
