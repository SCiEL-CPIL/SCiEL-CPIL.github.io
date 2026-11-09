import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # Convert adjacency matrix to list of neighbor lists for fast access
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = np.array([len(nei) for nei in neighbors])
    
    # k-core decomposition to get structural priority
    # Vertices removed later in k-core peeling are more "central"
    # For MIS, we want to pick low-degree / peripheral vertices first
    # So vertices removed EARLY (low core number) should have HIGH priority
    
    core_numbers = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    deg = degrees.copy()
    
    # Simple k-core peeling
    for k in range(1, int(degrees.max()) + 2):
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if remaining[i] and deg[i] < k:
                    remaining[i] = False
                    core_numbers[i] = k - 1
                    changed = True
                    for u in neighbors[i]:
                        if remaining[u]:
                            deg[u] -= 1
        if not remaining.any():
            break
    
    # Assign scores based on core number (lower core = higher priority)
    # Also use degree as tiebreaker (lower degree = higher priority)
    # Score = -core_number * 1000 - degree
    # Higher score = picked first
    
    scores = -core_numbers.astype(float) * 1000.0 - degrees.astype(float)
    
    # Now refine with local search on the order
    order = list(np.argsort(-scores))  # Highest score first
    
    def greedy_mis_size(order):
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                for u in neighbors[v]:
                    selected[u] = True
        return count
    
    current_size = greedy_mis_size(order)
    best_order = order[:]
    best_size = current_size
    
    rng = np.random.default_rng(42)
    
    # Local search with more aggressive steps
    num_restarts = 8
    steps_per_restart = 30000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            order = order[:]
        else:
            order = best_order[:]
            # More perturbations to escape local optima
            num_swaps = min(20, n // 5)
            for _ in range(num_swaps):
                i, j = rng.integers(0, n, 2)
                order[i], order[j] = order[j], order[i]
            
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order[:]
            
        temp = 3.0
        cooling = 0.9998
        
        for step in range(steps_per_restart):
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
                    best_order = order[:]
            elif delta == 0:
                current_size = new_size
            else:
                if rng.random() < np.exp(-delta / temp):
                    current_size = new_size
                else:
                    order[i], order[j] = order[j], order[i]
                    
            temp *= cooling
            if temp < 0.1:
                break
                
    # Convert best_order to scores
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
