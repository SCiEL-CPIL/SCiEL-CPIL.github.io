import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # Precompute neighbors as a list of lists for fast indexing in the loop
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = np.sum(adjacency, axis=1)
    
    # Vectorized Greedy MIS Size Calculator
    def greedy_mis_size(order):
        blocked = np.zeros(n, dtype=bool)
        count = 0
        b = blocked
        nb = neighbors
        for v in order:
            if not b[v]:
                count += 1
                b[v] = True
                b[nb[v]] = True
        return count

    # Compute weighted degree: sum of neighbor degrees
    weighted_deg = np.zeros(n, dtype=float)
    for i in range(n):
        if len(neighbors[i]) > 0:
            weighted_deg[i] = degrees[neighbors[i]].sum()
    
    rng = np.random.default_rng(42)
    
    # Initial Order:
    # Heuristic: Pick vertices with low degree first. 
    # Secondary: Low weighted degree (peripheral).
    sort_keys = degrees.astype(float) * 1000.0 + weighted_deg.astype(float) + rng.random(n) * 0.1
    initial_order = np.argsort(sort_keys) # Low degree first
    
    # Evaluate initial
    current_size = greedy_mis_size(initial_order)
    best_order = initial_order.copy()
    best_size = current_size
    
    # Local Search: Simulated Annealing
    num_restarts = 5
    base_steps = 100000 # Increased steps due to speedup
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        if r == 0:
            order = initial_order.copy()
        else:
            order = best_order.copy()
            # Perturb
            num_swaps = 50
            for _ in range(num_swaps):
                i, j = rng.integers(0, n, 2)
                order[i], order[j] = order[j], order[i]
        
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order.copy()
            
        temp = 5.0
        cooling = 0.9999
        min_temp = 0.01
        
        # Estimate steps
        elapsed = time.time() - start_time
        remaining_time = deadline - 10 - elapsed
        if remaining_time <= 0:
            break
            
        steps_this_restart = base_steps
        
        for step in range(steps_this_restart):
            if time.time() > deadline - 5:
                break
                
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            delta = current_size - new_size
            
            if delta < 0:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order.copy()
            elif delta == 0:
                current_size = new_size
            else:
                # Accept worse move with probability exp(-delta/temp)
                # delta > 0
                if rng.random() < np.exp(-delta / temp):
                    current_size = new_size
                else:
                    order[i], order[j] = order[j], order[i]
                    
            temp *= cooling
            if temp < min_temp:
                break
    
    # Convert best_order to scores
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
