import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Ensure adjacency is a numpy array for efficient operations
    adj = np.asarray(adjacency)
    
    def get_mis_size(order):
        """
        Calculate the size of the independent set obtained by 
        greedy decoding with the given vertex order.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as blocked
                # adj[v] is a boolean vector
                selected[adj[v]] = True
        return count

    # Initial heuristic: Lowest Degree First
    # Degrees are sums of rows
    degrees = adj.sum(axis=1)
    # We want to pick low degree first.
    # So the order should be vertices sorted by degree ascending.
    initial_order = np.argsort(degrees)
    
    best_order = initial_order.copy()
    best_size = get_mis_size(best_order)
    current_order = initial_order.copy()
    current_size = best_size
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-4
    cooling_rate = 0.9995
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin for safety
    
    # We run until time limit
    # To avoid infinite loop if time_limit is not reached (should not happen),
    # we can also cap iterations, but time is the primary constraint.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Propose a move: swap two random positions in the order
        # np.random.choice is fast enough for N=200
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Evaluate
        new_size = get_mis_size(current_order)
        
        # Calculate delta
        delta = new_size - current_size
        
        # Acceptance probability
        # If delta > 0, always accept.
        # If delta < 0, accept with prob exp(delta / T)
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = current_order.copy()
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Convert best_order to scores
    # Vertex at index k in best_order gets score n - k
    # This ensures descending score order matches best_order
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores
