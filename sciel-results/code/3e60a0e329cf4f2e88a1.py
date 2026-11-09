import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # Convert adjacency matrix to list of neighbor lists for fast access
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = np.array([len(nei) for nei in neighbors])
    
    # Compute weighted degree: sum of neighbor degrees (captures 2-hop structure)
    weighted_deg = np.zeros(n, dtype=float)
    for i in range(n):
        if len(neighbors[i]) > 0:
            weighted_deg[i] = degrees[neighbors[i]].sum()
    
    # Fast k-core decomposition using a bucket-based approach
    core_numbers = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    deg = degrees.copy()
    
    max_deg = int(degrees.max()) if n > 0 else 0
    
    if max_deg > 0:
        # Bucket sort by degree for efficient peeling
        from collections import defaultdict
        buckets = defaultdict(list)
        for i in range(n):
            if deg[i] <= max_deg:
                buckets[deg[i]].append(i)
        
        # Process vertices from lowest degree upward
        for k in range(0, max_deg + 1):
            while k in buckets and len(buckets[k]) > 0:
                v = buckets[k].pop()
                if not remaining[v]:
                    continue
                remaining[v] = False
                core_numbers[v] = k
                for u in neighbors[v]:
                    if remaining[u]:
                        old_d = deg[u]
                        deg[u] -= 1
                        new_d = deg[u]
                        if new_d < old_d:
                            # Move to new bucket
                            if new_d not in buckets:
                                buckets[new_d] = []
                            buckets[new_d].append(u)
        
        # Any remaining vertices get the max core number
        for i in range(n):
            if remaining[i]:
                core_numbers[i] = max_deg
    
    # Assign scores based on core number (lower core = higher priority)
    # Use weighted degree as secondary signal (lower = more peripheral)
    # Higher score = picked first
    scores = -core_numbers.astype(float) * 10000.0 - degrees.astype(float) * 100.0 - weighted_deg.astype(float)
    
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
    
    # Local search with adaptive parameters
    num_restarts = 12
    base_steps = 40000
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        if r == 0:
            order = order[:]
        else:
            order = best_order[:]
            # Perturbations scaled by restart number
            num_swaps = min(30, n // 4)
            for _ in range(num_swaps):
                i, j = rng.integers(0, n, 2)
                order[i], order[j] = order[j], order[i]
            
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order[:]
            
        # Simulated annealing parameters
        temp = 4.0
        cooling = 0.9997
        min_temp = 0.05
        
        steps_this_restart = base_steps
        # Use remaining time to estimate how many steps we can afford
        elapsed = time.time() - start_time
        remaining_time = deadline - 15 - elapsed
        if remaining_time > 0:
            # Estimate ~0.0005s per step, be conservative
            affordable_steps = int(remaining_time * 1500)
            steps_this_restart = min(base_steps, max(1000, affordable_steps))
        
        for step in range(steps_this_restart):
            if time.time() > deadline - 10:
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
            if temp < min_temp:
                break
                
    # Final polish: try a few more targeted improvements
    if time.time() < deadline - 5:
        order = best_order[:]
        current_size = best_size
        temp = 1.0
        for step in range(20000):
            if time.time() > deadline - 3:
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
            temp *= 0.9999
            if temp < 0.01:
                break
    
    # Convert best_order to scores
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
