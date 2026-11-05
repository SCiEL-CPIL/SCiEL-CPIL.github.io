import numpy as np
import time

def _greedy_size(adj, order):
    """
    Simulate greedy MIS: pick vertices in 'order', skip neighbors of picked.
    Returns size of independent set.
    """
    n = len(order)
    picked = np.zeros(n, dtype=bool)
    size = 0
    for v in order:
        if not picked[v]:
            picked[v] = True
            size += 1
            # Mark neighbors as picked
            nbrs = np.where(adj[v])[0]
            picked[nbrs] = True
    return size

def _initial_scores(adj):
    """
    Generate initial scores. We use negative degree as a starting point,
    but we'll optimize from here.
    """
    deg = adj.sum(axis=1)
    return -deg.copy()

def _swap_scores(scores, i, j):
    """
    Swap the scores of vertices i and j.
    Returns the new order implied by the scores.
    """
    s = scores.copy()
    s[i], s[j] = s[j], s[i]
    return s

def _get_order(scores):
    """
    Get the vertex order from scores (descending).
    """
    return np.argsort(-scores)

def _evaluate_order(adj, order):
    """
    Evaluate the greedy MIS size for a given order.
    """
    return _greedy_size(adj, order)

def _local_search(adj, scores, time_limit):
    """
    Local search: try swapping pairs of vertices in the order to improve MIS size.
    We use a simulated annealing-like approach or simple hill climbing.
    """
    start_time = time.time()
    n = len(scores)
    
    # Get initial order and size
    order = _get_order(scores)
    current_size = _evaluate_order(adj, order)
    best_size = current_size
    best_scores = scores.copy()
    
    # Precompute neighbor sets for fast access
    neighbor_sets = [set(np.where(adj[i])[0].tolist()) for i in range(n)]
    
    # Hill climbing: try all swaps, keep the best one
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_swap = None
        best_swap_size = current_size
        
        # Try all pairs (i, j) where i < j
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i+1, n):
                # Swap i and j in the order
                new_order = order.copy()
                pos_i = np.where(new_order == i)[0][0]
                pos_j = np.where(new_order == j)[0][0]
                new_order[pos_i], new_order[pos_j] = j, i
                
                # Evaluate
                new_size = _evaluate_order(adj, new_order)
                
                if new_size > best_swap_size:
                    best_swap_size = new_size
                    best_swap = (i, j)
                    improved = True
        
        if best_swap is not None:
            # Apply the best swap
            i, j = best_swap
            scores[i], scores[j] = scores[j], scores[i]
            order = _get_order(scores)
            current_size = best_swap_size
            if current_size > best_size:
                best_size = current_size
                best_scores = scores.copy()
        
        if (time.time() - start_time) > time_limit:
            break
    
    return best_scores, best_size

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Start with negative degree scores.
    2. Use local search to optimize the vertex ordering by directly evaluating
       the greedy MIS decoder's output.
    3. Return the scores that yield the best independent set size.
    """
    start_time = time.time()
    DEADLINE = 100.0  # Safety margin before 120s limit
    
    try:
        adj = np.asarray(adjacency, dtype=bool)
        n = adj.shape[0]
        
        if n < 2:
            return np.zeros(n)
        
        # Initial scores: negative degree
        scores = _initial_scores(adj)
        
        # Run local search
        time_limit = DEADLINE - 5.0  # Leave 5s safety margin
        scores, best_size = _local_search(adj, scores, time_limit)
        
        if time.time() - start_time > DEADLINE:
            return -adj.sum(axis=1)
            
        return scores
        
    except Exception:
        # Fallback to simple degree score
        try:
            deg = np.sum(adjacency, axis=1)
            return -deg
        except:
            return np.zeros(adjacency.shape[0])
