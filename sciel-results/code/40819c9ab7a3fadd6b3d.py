import numpy as np
import time
import random

def priority(adjacency):
    """
    Computes a priority score for each vertex to maximize the size of the
    independent set found by a greedy decoder.
    
    Strategy:
    1. Initialize an ordering based on degree (low degree first).
    2. Use Simulated Annealing to optimize the vertex ordering by swapping pairs.
    3. The objective function is the size of the independent set produced by 
       applying the greedy algorithm to that specific ordering.
    4. Return scores such that sorting by score descending yields the optimized order.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s buffer for safety
    
    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]
    
    # Precompute neighbor lists for faster evaluation
    # neighbors[i] is a list of indices of neighbors of i
    neighbors = [np.where(A[i])[0] for i in range(n)]
    
    def evaluate_order(order):
        """
        Simulates the greedy MIS selection given a specific order of vertices.
        Returns the size of the independent set.
        """
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v]:
                count += 1
                # Mark all neighbors as blocked
                # Using a loop is fine for n=200, but we can vectorize if needed.
                # For small n, direct indexing is fast.
                for u in neighbors[v]:
                    blocked[u] = True
        return count

    # 1. Initialization: Low degree first is a strong heuristic for MIS
    degrees = A.sum(axis=1)
    # Sort by degree ascending. If degrees are equal, randomize to break ties
    init_order = np.argsort(degrees, kind='mergesort')
    # To ensure determinism in ties for the initial state, we can just use the index
    # but randomizing ties in the initial state might help exploration.
    # Let's stick to a deterministic start for reproducibility, but the search will randomize.
    
    current_order = init_order.copy()
    current_score = evaluate_order(current_order)
    
    best_order = current_order.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    # We don't know the exact number of steps, so we use time-based stopping.
    # Estimate steps per second to set an alpha? 
    # Simpler: Just run as many as possible until time runs out.
    
    # To make the search effective, we need a good cooling schedule.
    # Let's estimate how many iterations we can do.
    # A single evaluation is O(n^2) roughly (looping n times, inner loop avg degree).
    # For n=200, this is very fast. Maybe 10,000+ iters/sec.
    
    # Let's try a fixed number of iterations if time allows, or dynamic.
    # Dynamic is safer.
    
    # We will perform random swaps.
    # To escape local minima, we use SA.
    
    # Initial temperature
    T = T_start
    
    # We need a cooling factor. If we do ~100,000 iterations, 
    # alpha = (T_end / T_start) ** (1/100000)
    # Let's just use a simple linear decay or exponential with a guess.
    # Or, better: run in batches.
    
    # Let's use a simple hill climbing with random restarts if we get stuck, 
    # or SA. SA is generally better.
    
    # Let's estimate max iterations. 
    # If we assume 1000 iters/sec, 145s -> 145,000 iters.
    # Let's try to run as many as we can.
    
    # To avoid getting stuck in a very deep local minimum, we can also 
    # occasionally do a larger perturbation (e.g., reverse a segment).
    
    # For simplicity and robustness, let's do standard SA with random swaps.
    
    # We need to track time carefully.
    # We'll use a loop that checks time every N iterations.
    
    check_interval = 100
    iter_count = 0
    
    # Random seed for reproducibility in the search process itself?
    # The task doesn't require reproducible *scores* across runs, just a good score.
    # But using a fixed seed inside the function might be good for debugging, 
    # though usually random is fine. Let's use random module.
    
    while time.time() < deadline:
        # Pick two random indices to swap
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        if i == j:
            continue
            
        # Perform swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Evaluate
        new_score = evaluate_order(current_order)
        
        # Acceptance Criteria
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_order = current_order.copy()
        else:
            # Simulated Annealing: accept with probability exp((new - current) / T)
            # Since scores are integers, new - current <= 0.
            # If new_score == current_score, accept to explore.
            if new_score == current_score:
                # Accept neutral moves to wander
                pass 
            else:
                # Calculate probability
                # delta = new_score - current_score (negative)
                prob = np.exp((new_score - current_score) / T)
                if random.random() > prob:
                    # Reject: swap back
                    current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Cooling
        # We need a cooling schedule. 
        # Let's assume we want to cool down over the expected number of iterations.
        # If we can do ~100,000 iterations, alpha = (0.01/1.0)**(1/100000) = 0.99995
        # This is a guess. A safer approach for time-limited SA is to 
        # just decrease T slowly.
        
        # Let's use a simple linear decrease if we can estimate total time.
        # Or just a very small multiplicative factor.
        T *= 0.99995
        
        iter_count += 1
        if iter_count % check_interval == 0:
            if time.time() > deadline - 5.0:
                break

    # Convert the best order to scores.
    # The greedy decoder sorts by score descending.
    # So the vertex at index 0 in best_order should have the highest score.
    # The vertex at index 1 should have the next highest, etc.
    
    scores = np.zeros(n, dtype=float)
    for rank, vertex in enumerate(best_order):
        # Higher score for earlier rank (lower rank index)
        # So score = n - rank
        scores[vertex] = n - rank
        
    return scores
