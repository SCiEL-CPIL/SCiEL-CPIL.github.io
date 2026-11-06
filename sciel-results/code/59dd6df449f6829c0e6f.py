"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: List-based Local Search on Vertex Permutation.
    1. Initialize with a heuristic ordering (e.g., degree-based).
    2. Use a fast vectorized simulator to evaluate the independent set size of a given order.
    3. Perform local search via insertion moves and swaps to optimize the ordering.
    4. Return scores derived from the optimized ordering (rank-based scores).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1)
    
    # Fast simulator: Given an order of vertices (indices), returns the size of the 
    # independent set produced by the greedy decoder.
    def simulate_order(order):
        # order is a list or np.array of vertex indices
        selected = 0
        # We need to check for each vertex in order if it is still available.
        # A vertex is available if it hasn't been selected and none of its neighbors 
        # have been selected previously in the order.
        
        # To make this fast, we can precompute neighbor masks? No, that's memory heavy for 200 nodes? 
        # 200x200 is small. Let's use bitsets or simple boolean arrays.
        
        # Efficient approach:
        # Maintain a 'blocked' mask. 
        # Iterate through order. If v is not blocked, select it, mark it and its neighbors as blocked.
        
        blocked = np.zeros(n, dtype=bool)
        for v in order:
            if not blocked[v]:
                selected += 1
                blocked[v] = True
                # Mark neighbors as blocked
                # adj_bool[v] is a boolean vector of neighbors
                blocked[adj_bool[v]] = True
                # Early exit if all blocked
                if blocked.all():
                    break
        return selected

    # Initialize ordering
    # Heuristic: Sort by -degree, then random tie-breaking
    # Or use a composite score to generate initial order
    init_scores = -degree + 0.1 * (adj_bool @ degree) + np.random.normal(0, 1, n)
    current_order = np.argsort(-init_scores, kind='stable').tolist()
    
    current_size = simulate_order(current_order)
    best_size = current_size
    best_order = current_order.copy()
    
    # Local Search
    # Moves: 
    # 1. Swap two random positions
    # 2. Insert a random vertex at a random position (shifting others)
    
    no_improve_count = 0
    max_no_improve = 100 # Allow more stagnation due to fast eval
    
    while time.time() < deadline:
        improved = False
        
        # Choose move type
        if np.random.random() < 0.5:
            # Swap move
            i, j = np.random.choice(n, 2, replace=False)
            new_order = current_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Insertion move (remove from i, insert at j)
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            new_order = current_order.copy()
            v = new_order.pop(i)
            # Adjust j if i < j
            if i < j:
                j -= 1
            new_order.insert(j, v)
        
        new_size = simulate_order(new_order)
        
        if new_size > current_size:
            current_order = new_order
            current_size = new_size
            improved = True
            
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
            
            # Accept worse moves with some probability (Simulated Annealing style)
            # But since eval is fast, we can be more aggressive or just greedy.
            # Let's stick to greedy for stability, maybe add slight SA later if time permits.
            # For now, greedy hill climbing on the permutation.
        else:
            # Try to accept worse moves to escape local minima?
            # Given the fast eval, we can afford a simple SA.
            # T decreases over time.
            # Let's keep it simple: Greedy first. If stuck, restart or random jump.
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Random restart or large perturbation
                # Large perturbation: Shuffle a chunk
                chunk_size = max(5, n // 10)
                start_idx = np.random.randint(n - chunk_size)
                chunk = current_order[start_idx:start_idx+chunk_size]
                np.random.shuffle(chunk)
                current_order[start_idx:start_idx+chunk_size] = chunk
                current_size = simulate_order(current_order)
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()

    # Convert best_order back to scores
    # The greedy decoder picks vertices in descending score order.
    # So, if vertex v is at position i in best_order (0 is first), it should have a high score.
    # Score = n - position
    scores = np.zeros(n, dtype=float)
    for i, v in enumerate(best_order):
        scores[v] = n - i
    
    return scores
# EVOLVE-BLOCK-END
