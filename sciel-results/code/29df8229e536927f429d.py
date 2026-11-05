import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # 1. Initial Order: Start with a good heuristic (e.g., degree ascending, or random)
    # Low degree vertices are often better to pick early in MIS heuristics? 
    # Actually, greedy MIS usually picks high degree first to block many? 
    # Wait, the decoder picks descending score. 
    # If we assign score = position in order (higher score = picked earlier), 
    # then we want the order to be the picking order.
    # Let's define 'order' as the list of vertices in the order they are considered.
    # score[v] = n - index_in_order[v]
    
    degrees = adj.sum(axis=1)
    
    # Initial order: Sort by degree (descending? ascending?)
    # Standard greedy for MIS: pick vertex, remove neighbors.
    # Heuristic: Pick vertex with min degree? Or max?
    # Let's start with random to let SA find the structure, or degree-based.
    # Let's try degree ascending (min degree first) as a common heuristic for independent sets.
    # But we are optimizing the *order*.
    
    order = np.random.permutation(n)
    
    # Precompute neighbors as lists for fast access
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    
    def evaluate_order(ord_arr):
        """
        Evaluate the size of the independent set produced by the greedy decoder
        given the order 'ord_arr'.
        ord_arr[i] is the vertex considered at step i.
        """
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in ord_arr:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                # Block neighbors
                # Using vectorized update might be faster but list is fine for n=200
                blocked[neighbors[v]] = True
        return count
    
    current_score = evaluate_order(order)
    best_order = order.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    cooling_rate = 0.9995
    min_T = 0.001
    
    # We will perform swaps.
    # To speed up, we can use a delta evaluation for swaps, but for n=200,
    # full evaluation is O(N^2) ~ 40k ops. 
    # 115s / 40k ops is plenty for millions of evaluations if optimized.
    # However, Python loops are slow.
    # Let's try to optimize the evaluation or use a simpler move.
    
    # Actually, let's use a faster evaluation method using bitsets if possible, 
    # or just stick to numpy bool arrays which are fast enough for n=200.
    
    # Let's try a simple hill-climbing with random restarts first, 
    # or SA with full re-evaluation.
    
    # To make it faster, we can precompute the "impact" of a swap? 
    # That's complex. Let's rely on the speed of numpy bool ops.
    
    # Optimization: Use a list of vertices for the order.
    # Swapping two elements in a list is O(1).
    
    # Let's run SA
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate a random swap
        i, j = random.sample(range(n), 2)
        
        # Swap in order
        order[i], order[j] = order[j], order[i]
        
        new_score = evaluate_order(order)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_order = order.copy()
        else:
            # Revert
            order[i], order[j] = order[j], order[i]
            
        T *= cooling_rate
        
        # If we are stuck for a long time, we might want to restart or cool faster
        # But for now, simple SA.
        
    # Convert best_order to scores
    # The decoder picks in descending score order.
    # So if best_order[0] is the first picked, it should have the highest score.
    # score[best_order[0]] = n
    # score[best_order[1]] = n-1
    # ...
    # score[best_order[n-1]] = 1
    
    scores = np.zeros(n, dtype=float)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores
