import numpy as np
import time

def _greedy_mis(adjacency, order):
    """Run greedy MIS with given vertex order (descending priority)."""
    n = len(order)
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        count += 1
        # Remove neighbors
        available[adjacency[v]] = False
        available[v] = False
    return count

def priority(adjacency):
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Initial order: lowest degree first
    initial_order = np.argsort(-degrees)  # descending priority = ascending degree
    # Wait, we need descending score. If score = -degree, then descending score = ascending degree.
    # So order should be vertices sorted by -degree descending = degree ascending.
    initial_order = np.argsort(degrees)  # ascending degree = highest priority first
    
    deadline = time.time() + 120  # 120s total, leave margin
    
    best_order = initial_order.copy()
    best_size = _greedy_mis(adjacency, best_order)
    
    # Local search: try swapping adjacent elements
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 5:
                break
            # Swap i and i+1
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_size = _greedy_mis(adjacency, new_order)
            if new_size > best_size:
                best_size = new_size
                best_order = new_order
                improved = True
    
    # Convert best_order to scores: earlier in order = higher score
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank  # higher score = picked earlier
    
    return scores
