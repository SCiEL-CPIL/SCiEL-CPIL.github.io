import numpy as np
import time

def priority(adjacency):
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    
    def simulate(order):
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count

    # Heuristic Initialization
    deg = adj.sum(axis=1)
    nbr_deg_sum = adj @ deg
    
    # Sort by degree, then by nbr_deg_sum
    # We want low degree first
    # score for sorting: deg * 1000 + nbr_deg_sum
    sort_keys = deg.astype(float) + 0.001 * nbr_deg_sum.astype(float)
    initial_order = np.argsort(sort_keys, kind='stable')
    
    best_val = simulate(initial_order)
    best_order = initial_order.copy()
    
    start_time = time.time()
    max_time = 145.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    current_order = initial_order.copy()
    
    # Iterated Local Search
    while time.time() - start_time < max_time:
        # Local Search
        current_val = simulate(current_order)
        no_improve = 0
        max_no_improve = 5000 # Tune this
        
        while time.time() - start_time < max_time - 1 and no_improve < max_no_improve:
            i, j = rng.integers(0, n, size=2)
            if i == j: continue
            
            # Swap
            current_order[i], current_order[j] = current_order[j], current_order[i]
            new_val = simulate(current_order)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_order = current_order.copy()
            else:
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
                no_improve += 1
                
        # Perturb for next restart
        # Swap a few random pairs
        perturbations = rng.integers(3, 10)
        for _ in range(perturbations):
            i, j = rng.integers(0, n, size=2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        # If we are stuck for a long time, re-initialize from best?
        # The perturbation from current (which is local best) is standard ILS.
        
    # Convert best_order to scores
    # The score should reflect the order.
    # Higher score = picked earlier.
    # So rank 0 in order (first) gets highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
