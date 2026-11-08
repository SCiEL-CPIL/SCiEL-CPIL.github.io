import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists as numpy arrays for fast indexing
    # adjacency is a bool matrix
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = adjacency.sum(axis=1)
    
    # Fast greedy MIS evaluation
    # order: array of vertex indices in consideration order
    # Returns the size of the independent set
    def greedy_mis_size(order):
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as blocked
                if len(neighbors[v]) > 0:
                    selected[neighbors[v]] = True
        return count
    
    # We want to find a good ordering (permutation) of vertices.
    # The greedy decoder picks vertices in descending score order.
    # So the first vertex in the order has the highest score.
    
    # Strategy:
    # 1. Start with several heuristic orderings.
    # 2. Improve each using local search (swaps) with simulated annealing.
    # 3. Return the best ordering found as scores.
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_order = None
    
    def update_best(order, size):
        nonlocal best_size, best_order
        if size > best_size:
            best_size = size
            best_order = order[:]
    
    # Generate initial orderings
    initial_orders = []
    
    # 1. Minimum degree (ascending degree)
    initial_orders.append(np.argsort(degrees))
    
    # 2. Maximum degree (descending degree)
    initial_orders.append(np.argsort(-degrees))
    
    # 3. Random
    initial_orders.append(rng.permutation(n))
    
    # 4. Core number based ordering (approximated by degree for simplicity, or compute core number)
    # Let's compute a simple core number-like score: 
    # Actually, let's just use noisy degree
    noisy_deg = degrees + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_deg))
    
    # 5. Another random
    initial_orders.append(rng.permutation(n))
    
    # 6. Noisy min degree
    noisy_min_deg = -degrees + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_min_deg))
    
    # 7. Noisy max degree
    noisy_max_deg = degrees + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(-noisy_max_deg))
    
    # Evaluate initial orders
    for order in initial_orders:
        size = greedy_mis_size(order)
        update_best(order, size)
    
    # Local search: Simulated Annealing with swaps
    # We'll use a time-based loop to ensure we finish in time.
    # For each initial order, we'll do local search.
    
    num_restarts = len(initial_orders)
    steps_per_restart = 8000
    
    # If we have time, we can do more restarts or steps.
    # Let's use a global time budget.
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        order = initial_orders[r].copy()
        current_size = greedy_mis_size(order)
        update_best(order, current_size)
        
        # Simulated Annealing
        # Start with high temperature, cool down over time
        initial_temp = 2.0
        cooling_rate = 0.9995
        
        temperature = initial_temp
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
            
            temperature *= cooling_rate
            
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            delta = new_size - current_size
            
            if delta >= 0:
                # Accept better or equal solutions
                current_size = new_size
                update_best(order, current_size)
            else:
                # Accept worse solutions with probability exp(delta / temperature)
                if temperature > 1e-6:
                    prob = np.exp(delta / temperature)
                    if rng.random() < prob:
                        current_size = new_size
                    else:
                        # Revert
                        order[i], order[j] = order[j], order[i]
                else:
                    # Revert
                    order[i], order[j] = order[j], order[i]
    
    # Convert best_order to scores
    # The decoder sorts by descending score.
    # So the first element in best_order should have the highest score.
    if best_order is None:
        # Fallback: min degree
        scores = np.zeros(n)
        order = np.argsort(degrees)
        for rank, v in enumerate(order):
            scores[v] = n - rank
        return scores
    
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
