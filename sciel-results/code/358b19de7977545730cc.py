import numpy as np
import time
import random

def _greedy_set_size_from_order(adj, order):
    """
    Run the fixed greedy decoder given a specific order of vertices.
    Returns the size of the independent set.
    """
    n = adj.shape[0]
    chosen = set()
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        v = int(v)
        if not blocked[v]:
            chosen.add(v)
            count += 1
            # Block v and its neighbors
            blocked[v] = True
            nbrs = np.nonzero(adj[v])[0]
            if len(nbrs) > 0:
                blocked[nbrs] = True
    return count

def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists as arrays for faster access if needed, 
    # but nonzero on bool matrix is quite fast for n=200.
    
    # 1. Initial Heuristic: 
    # Start with vertices of low degree first.
    # In the greedy decoder, we pick the first available vertex.
    # So we want low-degree vertices to have high scores (early in order).
    deg = adj.sum(axis=1)
    # Sort by degree ascending. Ties broken randomly.
    # We want the permutation such that low degree comes first.
    # Score = -degree. 
    # But we work with permutations directly.
    # Initial permutation: sorted by degree, then random.
    
    # To allow SA, we need a state that is a permutation of 0..n-1.
    # Let's create an initial permutation based on degree.
    # Lower degree -> earlier in permutation -> higher priority.
    # We assign "scores" later. For now, just the order.
    
    # Randomize ties in degree to avoid deterministic bad start.
    # We can just sort indices by (degree, random_key).
    keys = np.random.rand(n)
    # Combine: primary sort degree asc, secondary sort random key.
    # np.lexsort sorts by the last key first.
    # So keys=[random, degree] -> sorts by degree, then random? 
    # No, lexsort((key1, key2)) sorts by key2 then key1.
    # We want primary: degree. Secondary: random.
    # So lexsort((random_keys, degrees)).
    init_order = np.lexsort((keys, deg))
    
    # Convert order to scores for the final return.
    # But for optimization, we keep the permutation.
    
    best_order = init_order.copy()
    best_size = _greedy_set_size_from_order(adj, best_order)
    
    current_order = init_order.copy()
    current_size = best_size
    
    # Simulated Annealing Parameters
    # Time budget: 110 seconds to be safe.
    deadline = start_time + 110.0
    
    T = 1.0
    T_min = 0.01
    # We need to estimate how many steps we can do.
    # Each step: 1 swap + 1 greedy run.
    # Greedy run is ~O(n^2) worst case, but usually less.
    # For n=200, it's very fast. Let's say 1000 steps/sec? Maybe more.
    
    it = 0
    no_improve = 0
    max_no_improve = 5000  # Stop if stuck for a long time
    
    while time.time() < deadline:
        # Generate move: swap two random positions in the permutation
        i, j = random.sample(range(n), 2)
        
        # Apply swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Evaluate
        new_size = _greedy_set_size_from_order(adj, current_order)
        
        # Acceptance
        if new_size > current_size:
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = current_order.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Accept with probability exp((new_size - current_size) / T)
            # Note: new_size - current_size is <= 0.
            # Since size is integer, diff is 0, -1, -2...
            # If diff is 0, prob is 1.
            # If diff is -1, prob is exp(-1/T).
            diff = new_size - current_size
            if diff < 0:
                if T > T_min:
                    prob = np.exp(diff / T)
                    if random.random() < prob:
                        current_size = new_size
                        no_improve += 1
                    else:
                        # Reject
                        current_order[i], current_order[j] = current_order[j], current_order[i]
                        no_improve += 1
                else:
                    # Reject
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    no_improve += 1
            else:
                # diff == 0, accept to explore
                no_improve += 1
        
        # Cool down
        # Simple exponential cooling.
        # We don't know total steps, so use time-based or step-based.
        # Let's use step-based with a fixed rate, but ensure we don't cool too fast.
        T *= 0.9999 
        if T < T_min:
            T = T_min
        
        it += 1
        
        if no_improve > max_no_improve:
            break

    # Convert best_order to scores.
    # The greedy decoder sorts by score descending.
    # So vertex at index 0 in best_order (first to be considered) should have the highest score.
    # Vertex at index n-1 should have the lowest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        # Higher score = earlier in order.
        # Score = n - rank.
        scores[v] = n - rank
        
    return scores
