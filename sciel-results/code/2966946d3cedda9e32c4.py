import numpy as np
import time
from collections import deque

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # Precompute neighbor lists as integer arrays for fast access
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    def greedy_mis_size(order):
        """
        order: list of vertices in the order they should be considered (first is highest priority)
        Returns the size of the independent set.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as selected (blocked)
                if len(neighbors[v]) > 0:
                    selected[neighbors[v]] = True
        return count
    
    def greedy_mis_size_fast(order):
        """
        Optimized greedy MIS size using bitmask-like approach with numpy.
        order: list of vertices in the order they should be considered
        Returns the size of the independent set.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                if len(neighbors[v]) > 0:
                    selected[neighbors[v]] = True
        return count

    # Compute k-core decomposition to get removal order
    # Vertices removed later are in higher cores (more central)
    remaining = np.ones(n, dtype=bool)
    current_degrees = adjacency.sum(axis=1).copy()
    removal_order = np.zeros(n, dtype=int)
    removal_time = np.zeros(n, dtype=int)
    t = 0
    
    # Use BFS-like peeling with a priority queue (min-heap by degree)
    import heapq
    
    # Initialize heap with all vertices
    heap = [(int(current_degrees[i]), i) for i in range(n)]
    heapq.heapify(heap)
    
    in_heap = np.ones(n, dtype=bool)
    
    while heap:
        deg, v = heapq.heappop(heap)
        
        if not remaining[v]:
            continue
        
        if deg != current_degrees[v]:
            continue
        
        removal_order[v] = t
        removal_time[v] = t
        t += 1
        
        # Update degrees of neighbors
        for u in neighbors[v]:
            if remaining[u]:
                current_degrees[u] -= 1
                heapq.heappush(heap, (int(current_degrees[u]), u))
        
        remaining[v] = False
    
    # Now we have removal_time[v]: lower means removed earlier (less central)
    # Vertices with higher removal_time are in the core (more central)
    
    # For MIS, we want to pick low-degree, peripheral vertices first
    # So we want vertices with LOW removal_time to have HIGH priority (be picked first)
    
    # Combine degree and core information
    degrees = adjacency.sum(axis=1)
    
    # Score: low degree and low removal time (peripheral) should be high
    # Let's create a composite score
    # Normalize degrees and removal times
    deg_norm = degrees / (degrees.max() + 1e-9)
    time_norm = removal_time / (removal_time.max() + 1e-9)
    
    # Initial order: sort by combined score (ascending, so lowest score = highest priority)
    # Strategy: sort by (removal_time, degree) ascending
    # Vertices removed first (low core) with low degree should go first
    
    initial_order = list(np.lexsort((degrees, removal_time)))
    
    current_order = initial_order
    current_size = greedy_mis_size(current_order)
    
    best_order = current_order[:]
    best_size = current_size
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different strategies
    num_restarts = 12
    steps_per_restart = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        if r == 0:
            # Core-based order
            order = list(np.lexsort((degrees, removal_time)))
        elif r == 1:
            # Pure min degree
            order = list(np.argsort(degrees))
        elif r == 2:
            # Random
            order = list(rng.permutation(n))
        elif r == 3:
            # Max degree (control)
            order = list(np.argsort(-degrees))
        elif r == 4:
            # Noisy degree
            noisy_deg = degrees + rng.normal(0, 1, n)
            order = list(np.argsort(noisy_deg))
        elif r == 5:
            # Noisy core time
            noisy_time = removal_time + rng.normal(0, 1, n)
            order = list(np.argsort(noisy_time))
        elif r == 6:
            # Combined noisy
            combined = 0.5 * (degrees / (degrees.max() + 1e-9)) + 0.5 * (removal_time / (removal_time.max() + 1e-9))
            noisy_combined = combined + rng.normal(0, 0.1, n)
            order = list(np.argsort(noisy_combined))
        elif r == 7:
            # Weighted: emphasize low degree more
            weighted = 0.7 * deg_norm + 0.3 * time_norm
            noisy_weighted = weighted + rng.normal(0, 0.05, n)
            order = list(np.argsort(noisy_weighted))
        elif r == 8:
            # Weighted: emphasize core time more
            weighted = 0.3 * deg_norm + 0.7 * time_norm
            noisy_weighted = weighted + rng.normal(0, 0.05, n)
            order = list(np.argsort(noisy_weighted))
        elif r == 9:
            # Inverse: try higher degree first (sometimes helps)
            order = list(np.lexsort((-degrees, -removal_time)))
        elif r == 10:
            # Shuffled best order
            order = best_order[:]
            rng.shuffle(order)
        else:
            # More aggressive noise
            combined = 0.5 * deg_norm + 0.5 * time_norm
            noisy_combined = combined + rng.normal(0, 0.3, n)
            order = list(np.argsort(noisy_combined))
            
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order[:]
            
        # Local Search Phase: Hill Climbing with random swaps
        for step in range(steps_per_restart):
            if time.time() > deadline - 10:
                break
                
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            if new_size >= current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order[:]
            else:
                # Revert
                order[i], order[j] = order[j], order[i]
                
    # Final polish: try to improve best_order with targeted swaps
    if time.time() < deadline - 5:
        # Try swapping pairs in best_order to see if we can improve
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    best_order[i], best_order[j] = best_order[j], best_order[i]
                    new_size = greedy_mis_size(best_order)
                    if new_size > best_size:
                        best_size = new_size
                        improved = True
                    else:
                        best_order[i], best_order[j] = best_order[j], best_order[i]
                
    # Convert best_order to scores
    # The decoder sorts by descending score.
    # So the first element in best_order should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
