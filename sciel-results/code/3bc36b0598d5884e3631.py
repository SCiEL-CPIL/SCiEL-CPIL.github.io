import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute neighbor indices for each vertex to speed up evaluation
    # Using lists of integers is often faster for iteration in Python than numpy arrays for small n
    neighbor_lists = [np.where(adjacency[i])[0] for i in range(n)]
    
    # Convert adjacency to a list of lists of neighbors for even faster access in the eval loop
    # Actually, numpy boolean indexing `selected[adjacency[v]] = True` is very fast.
    # Let's stick to numpy for the evaluation.
    
    def evaluate(order):
        """
        Evaluate the size of the independent set produced by the greedy decoder
        given the vertex order.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Block all neighbors
                selected[adjacency[v]] = True
        return count

    # Initial heuristic: Sort by degree (ascending). 
    # Low degree vertices are "cheaper" to pick and block fewer others.
    degrees = adjacency.sum(axis=1)
    
    # Create initial order
    # We want low degree first.
    initial_scores = -degrees
    current_order = np.argsort(-initial_scores) # Descending score -> Ascending degree
    
    # Ensure unique scores for deterministic tie-breaking in argsort? 
    # argsort is stable, but we want a specific order.
    # Actually, we are manipulating the order directly, so ties don't matter for the evaluation.
    # But for the final output, we need to map this order to scores.
    
    current_size = evaluate(current_order)
    best_size = current_size
    best_order = current_order.copy()
    
    # Local Search
    rng = np.random.RandomState(42)
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2.0:
        # Generate a random move
        # Strategy: Swap two random vertices in the order
        i = rng.randint(n)
        j = rng.randint(n)
        if i == j:
            continue
            
        new_order = current_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_size = evaluate(new_order)
        
        if new_size > current_size:
            current_order = new_order
            current_size = new_size
            no_improve_count = 0
            
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with a slightly perturbed version of the best order found so far
                # Or just randomize a portion of the current order
                current_order = best_order.copy()
                # Randomly swap 10% of the vertices
                num_swaps = max(1, n // 10)
                for _ in range(num_swaps):
                    i = rng.randint(n)
                    j = rng.randint(n)
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                current_size = evaluate(current_order)
                no_improve_count = 0

    # Convert best_order to scores
    # The greedy decoder picks in descending score order.
    # So the first vertex in best_order should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
