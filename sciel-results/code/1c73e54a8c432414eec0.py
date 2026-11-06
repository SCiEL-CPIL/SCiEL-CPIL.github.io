import numpy as np
import time

def _greedy_mis(adjacency, order):
    """Run greedy MIS with given vertex order (descending priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency[v]] = False
    return count

def _fast_greedy_size(adjacency, available_mask):
    """
    Estimate MIS size of the subgraph defined by available_mask 
    using a simple Min-Degree heuristic.
    """
    n = adjacency.shape[0]
    avail = available_mask.copy()
    avail_indices = np.where(avail)[0]
    if len(avail_indices) == 0:
        return 0
        
    # Compute degrees in subgraph
    degs = np.sum(adjacency[:, avail], axis=1)
    degs = degs[avail_indices]
    
    # Sort available indices by degree (ascending)
    sorted_indices = avail_indices[np.argsort(degs)]
    
    # Greedy simulation on this order
    avail_sim = np.zeros(n, dtype=bool)
    avail_sim[avail] = True
    
    cnt = 0
    for v in sorted_indices:
        if avail_sim[v]:
            cnt += 1
            avail_sim[v] = False
            avail_sim[adjacency[v]] = False
            
    return cnt

def priority(adjacency):
    n = adjacency.shape[0]
    deadline = time.time() + 115 # Safety margin (150s limit, 35s buffer)
    
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Candidate 1: Min Degree (Standard)
    score_min_deg = -degrees
    
    # Candidate 2: Min Degree + Neighbor Sum
    neighbor_deg_sum = adjacency.dot(degrees)
    
    candidates = []
    
    # Add Min Degree
    candidates.append(score_min_deg)
    
    # Add variations of the Neighbor Sum heuristic
    for lam in [0.1, 0.5, 1.0, 2.0]:
        score = -degrees + lam * neighbor_deg_sum
        candidates.append(score)
        
    # Add variations that penalize neighbor sum
    for lam in [-0.5, -1.0]:
        score = -degrees + lam * neighbor_deg_sum
        candidates.append(score)

    # Candidate: 1-Step Lookahead
    if time.time() < deadline - 20:
        lookahead_scores = np.zeros(n)
        for v in range(n):
            mask = np.ones(n, dtype=bool)
            mask[v] = False
            mask[adjacency[v]] = False
            est_size = _fast_greedy_size(adjacency, mask)
            lookahead_scores[v] = 1 + est_size
        candidates.append(lookahead_scores)

    # Evaluate all candidates to find the best score vector
    best_size = -1
    best_score = candidates[0]
    
    for score in candidates:
        if time.time() > deadline - 10:
            break
        order = np.argsort(-score)
        size = _greedy_mis(adjacency, order)
        if size > best_size:
            best_size = size
            best_score = score

    # Local Search Refinement
    # Convert best_score to an initial order
    best_order = np.argsort(-best_score)
    
    # Improved Local Search with Random Kicks
    # We use a combination of adjacent swaps and random perturbations (kicks)
    # to escape local minima.
    
    # First, do a few passes of pure adjacent swaps to settle
    passes = 0
    max_swaps_passes = 5
    while passes < max_swaps_passes and time.time() < deadline - 5:
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
        if not improved:
            break
        passes += 1
        
    # Now, use random kicks to explore further
    # If we still have time, try random swaps of distant elements
    if time.time() < deadline - 2:
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in local search? 
        # Actually, using a fixed seed might be bad if we want diversity across runs?
        # The task says "five generated Erdos-Renyi graphs", so the graph is fixed per eval?
        # No, the function is called once per graph.
        # Using a fixed seed inside the function makes the local search deterministic for a given input.
        # This is usually good for stability.
        
        kick_attempts = 0
        max_kicks = 1000 # Arbitrary limit, will be cut by time
        
        while time.time() < deadline - 1 and kick_attempts < max_kicks:
            kick_attempts += 1
            
            # Perform a random kick: swap two random positions
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            i, j = int(i), int(j)
            
            new_order = best_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            # After a kick, do a few adjacent swap passes to settle
            local_best_order = new_order
            local_best_size = _greedy_mis(adjacency, local_best_order)
            
            # Settle with 1 pass of adjacent swaps
            for _ in range(1):
                improved_local = False
                for k in range(n - 1):
                    if time.time() > deadline - 1:
                        break
                    temp_order = local_best_order.copy()
                    temp_order[k], temp_order[k+1] = temp_order[k+1], temp_order[k]
                    temp_size = _greedy_mis(adjacency, temp_order)
                    if temp_size > local_best_size:
                        local_best_size = temp_size
                        local_best_order = temp_order
                        improved_local = True
                if not improved_local:
                    break
            
            # If the kicked version is better than the global best, update
            if local_best_size > best_size:
                best_size = local_best_size
                best_order = local_best_order

    # Convert best_order back to scores
    # The greedy decoder picks vertices in descending score order.
    # So vertex at index 0 in best_order should have the highest score.
    # Vertex at index 1 should have the next highest, etc.
    # We can assign scores as n-1, n-2, ..., 0 or simply 1/n, 1/(n+1)... 
    # Using rank is sufficient.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
