"""Improved program for Maximum Independent Set greedy scoring.
Uses dynamic heuristics (Min-Degree) and spectral methods to generate
better initial priority orders, followed by local search refinement.
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

def _dynamic_min_degree_order(A, n):
    """Generate order using dynamic min-degree heuristic."""
    available = np.ones(n, dtype=bool)
    order = np.zeros(n, dtype=int)
    deg = A.sum(axis=1).astype(int)
    
    for pos in range(n):
        if not available.any():
            break
        # Mask degrees: set unavailable to inf
        cur_deg = np.where(available, deg, np.inf)
        v = np.argmin(cur_deg)
        order[pos] = v
        available[v] = False
        # Update degrees of neighbors
        neighbors = np.where(A[v])[0]
        deg[neighbors] -= 1
        
    return order

def _dynamic_max_neighbor_degree_order(A, n):
    """Generate order preferring vertices with low-degree neighbors (indirectly)."""
    available = np.ones(n, dtype=bool)
    order = np.zeros(n, dtype=int)
    deg = A.sum(axis=1).astype(int)
    
    for pos in range(n):
        if not available.any():
            break
        # Score: degree of vertex + 0.5 * avg degree of neighbors
        scores = np.full(n, np.inf)
        avail_idx = np.where(available)[0]
        for v in avail_idx:
            nbrs = np.where(A[v])[0]
            nbrs_avail = nbrs[available[nbrs]]
            if len(nbrs_avail) > 0:
                avg_nbr_deg = deg[nbrs_avail].mean()
            else:
                avg_nbr_deg = 0.0
            scores[v] = deg[v] + 0.5 * avg_nbr_deg
            
        v = np.argmin(scores)
        order[pos] = v
        available[v] = False
        neighbors = np.where(A[v])[0]
        deg[neighbors] -= 1
        
    return order

def _spectral_order(A, n, reverse=False):
    """Generate order based on spectral centrality."""
    try:
        eigvals, eigvecs = np.linalg.eig(A)
        # Largest eigenvalue
        idx = np.argmax(eigvals)
        vec = np.real(eigvecs[:, idx])
        # Normalize
        vec = vec - vec.min()
        vec = vec / (vec.max() + 1e-9)
        
        if reverse:
            # Pick low centrality first
            return np.argsort(vec, kind='stable')
        else:
            # Pick high centrality first
            return np.argsort(-vec, kind='stable')
    except:
        return np.arange(n)

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
    
    # 4. Dynamic Max-Neighbor-Degree (Low score = low deg + low nbr deg)
    orders.append(_dynamic_max_neighbor_degree_order(A, n))
    
    # 5. Spectral (Low Centrality first)
    orders.append(_spectral_order(A, n, reverse=True))
    
    # 6. Spectral (High Centrality first)
    orders.append(_spectral_order(A, n, reverse=False))
    
    # 7. Random
    orders.append(rng.permutation(n))
    
    # 8. Core Number Order (approximate)
    # Simple version: peel lowest degree
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
    orders.append(order)

    return orders

def _local_search(A, start_order, n, rng, deadline):
    """Perform local search starting from start_order."""
    current_order = start_order.copy()
    current_size = _greedy_size(A, current_order)
    best_order = current_order.copy()
    best_size = current_size
    no_improve = 0
    max_no_improve = 200  # Increase patience slightly
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        # Random swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        new_order = current_order.copy()
        new_order[i], new_order[j] = current_order[j], current_order[i]
        
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
            
        # Occasionally try an adjacent swap (smaller step)
        if rng.random() < 0.2:
             i = rng.integers(0, n-1)
             j = i + 1
             new_order2 = current_order.copy()
             new_order2[i], new_order2[j] = current_order[j], current_order[i]
             new_size2 = _greedy_size(A, new_order2)
             if new_size2 > current_size:
                 current_order = new_order2
                 current_size = new_size2
                 no_improve = 0
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
        if time.time() > deadline - 30:
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
    
    # Local search from the best 3 candidates
    ls_deadline = deadline - 10.0
    
    # 1. Local search from the absolute best
    best_order, best_size = _local_search(A, best_order, n, rng, ls_deadline)
    
    # 2. If time permits, local search from the 2nd and 3rd best
    if len(top_candidates) > 1 and time.time() < ls_deadline - 20:
        _, second_best = top_candidates[1]
        o2, s2 = _local_search(A, second_best, n, rng, ls_deadline)
        if s2 > best_size:
            best_size = s2
            best_order = o2

    if len(top_candidates) > 2 and time.time() < ls_deadline - 10:
        _, third_best = top_candidates[2]
        o3, s3 = _local_search(A, third_best, n, rng, ls_deadline)
        if s3 > best_size:
            best_size = s3
            best_order = o3

    # Convert best_order to priority vector
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
        
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
