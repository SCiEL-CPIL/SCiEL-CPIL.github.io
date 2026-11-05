import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy: 
    1. Initialize order using a static heuristic: Low Degree.
    2. Perform Simulated Annealing on the permutation (order) directly.
       - Move: Pick a random vertex and move it to a random position.
       - Evaluation: Simulate the greedy decoder on the current order.
    3. Return scores that induce the best order found.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Precompute neighbor lists as numpy arrays for fast vectorized deactivation
    neighbors_list = []
    for i in range(n):
        neighbors_list.append(np.where(adj[i])[0])
    
    # 1. Initial Heuristic
    # Low degree is generally good for MIS
    deg = adj.sum(axis=1).astype(float)
    # Sort by degree ascending (low degree first)
    initial_order = np.argsort(deg)
    
    # 2. Fast Simulation Function
    def simulate(order):
        """
        Simulates the greedy decoder on a given order and returns the size.
        """
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                if len(neighbors_list[v]) > 0:
                    active[neighbors_list[v]] = False
        return count

    # 3. Simulated Annealing on Permutation
    current_order = initial_order.copy()
    current_size = simulate(current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    # Time budget: Use most of the available time, but leave a safety margin.
    # Evaluator kills at 150s. We aim for 140s.
    max_time = 140.0
    rng = np.random.default_rng(42)
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    
    # Main loop
    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / max_time)
        T = T_start * (T_end / T_start) ** progress
        
        # Move: Pick a random vertex and move it to a random position
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        
        if i == j:
            continue
            
        # Perform the move on a copy to evaluate
        new_order = current_order.copy()
        
        # Remove element at i
        v = new_order[i]
        new_order = np.delete(new_order, i)
        
        # Insert at j
        new_order = np.insert(new_order, j, v)
        
        # Re-evaluate
        new_size = simulate(new_order)
        
        delta = new_size - current_size
        
        if delta > 0:
            current_order = new_order
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = new_order.copy()
        else:
            if T > 0:
                prob = np.exp(delta / T)
            else:
                prob = 0.0
                
            if rng.random() < prob:
                current_order = new_order
                current_size = new_size
            # else: revert (do nothing, current_order remains unchanged)
            
    # 4. Construct scores from best_order
    # The decoder sorts by descending score.
    # We want the decoder to pick vertices in the order of best_order.
    scores = np.zeros(n)
    scores[best_order] = np.arange(n - 1, -1, -1)
    
    return scores
