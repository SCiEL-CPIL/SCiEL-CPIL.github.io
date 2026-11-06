import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0
    n = adjacency.shape[0]
    
    # Precompute adjacency bitsets
    adj_bits = [0] * n
    for u in range(n):
        for v in range(n):
            if adjacency[u, v]:
                adj_bits[u] |= (1 << v)
    
    # Helper to evaluate greedy size
    def greedy_size(order):
        selected_bits = 0
        size = 0
        for v in order:
            if (adj_bits[v] & selected_bits) == 0:
                selected_bits |= (1 << v)
                size += 1
        return size

    # Initial Heuristic: BP
    A = adjacency.astype(float)
    x = np.ones(n)
    for _ in range(20):
        x = 1.0 / (1.0 + A @ x)
    
    # Initial order
    order = list(np.argsort(x)[::-1])
    current_size = greedy_size(order)
    
    # Local Search
    # We will try to improve the order by swapping elements.
    # To avoid O(n^2) swaps being too slow if we do them all, we pick random pairs.
    # Or we can do a pass of adjacent swaps (bubble sort style) but random is better for escaping local minima.
    
    no_improve_count = 0
    max_no_improve = 5000 # Restart or perturb if stuck
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Perturb: randomize a few positions
            for _ in range(5):
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                order[i], order[j] = order[j], order[i]
            current_size = greedy_size(order)
            no_improve_count = 0
            continue
            
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
            
        # Swap
        order[i], order[j] = order[j], order[i]
        
        new_size = greedy_size(order)
        
        if new_size > current_size:
            current_size = new_size
            no_improve_count = 0
        else:
            # Revert
            order[i], order[j] = order[j], order[i]
            no_improve_count += 1
            
    # Convert order back to scores
    # Higher score = earlier in order
    scores = np.zeros(n)
    for idx, v in enumerate(order):
        scores[v] = n - idx
        
    return scores
