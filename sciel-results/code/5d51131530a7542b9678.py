"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # Convert to float matrix for efficient matrix multiplication
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = adj > 0.5
    degrees = adj.sum(axis=1)
    
    # Precompute neighbor indices for each vertex for fast access
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    def greedy_size_from_order(order):
        """Compute greedy MIS size given a priority order (array of vertex indices)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count

    def order_from_scores(s):
        """Get vertex order by descending score."""
        return np.argsort(-s)

    # Initial order: low degree first (greedy MIS heuristic)
    # We assign initial scores so that low-degree vertices come first
    # Use negative degree as initial score
    scores = -degrees.copy()
    
    current_order = order_from_scores(scores)
    current_size = greedy_size_from_order(current_order)
    best_size = current_size
    best_scores = scores.copy()
    
    # Simulated Annealing
    # We will perturb the scores and accept based on the change in greedy size
    # Since the objective is discrete (integer count), we need a careful acceptance criterion
    # or we can just do a random hill-climbing with restarts if SA is too slow to tune.
    # However, the directive asks for a structural change. SA on score vectors is a standard
    # approach for combinatorial optimization where the objective is not smooth.
    
    # To make it efficient, we will perturb the scores slightly and check the new order.
    # But recalculating greedy size is O(n^2) in worst case if we do it naively.
    # With n=200, n^2 = 40,000 operations. This is fast in Python if optimized or using numpy.
    # Let's use a vectorized greedy check if possible, or just rely on the fact that n is small.
    
    # Actually, let's stick to the order-based perturbation which is equivalent to score perturbation
    # but allows us to directly manipulate the sequence.
    
    # Strategy:
    # 1. Start with low-degree order.
    # 2. Randomly swap two elements in the order.
    # 3. Calculate new size.
    # 4. If better, accept.
    # 5. If worse, accept with probability exp(-delta/T).
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.99
    
    # We need a fast way to evaluate the size.
    # The current greedy_size_from_order is O(n * avg_degree).
    # For n=200, this is very fast.
    
    max_iterations = 10000
    iter_count = 0
    
    while iter_count < max_iterations:
        if time.time() > deadline:
            break
        
        # Random swap
        i, j = random.sample(range(n), 2)
        
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_size = greedy_size_from_order(current_order)
        delta = new_size - current_size
        
        if delta > 0 or random.random() < np.exp(-delta / T):
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                # Update best scores to reflect this order
                # Use rank-based scores: higher rank = higher score
                best_scores = np.zeros(n)
                for rank, v in enumerate(current_order):
                    best_scores[v] = n - rank
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
        
        T *= cooling_rate
        iter_count += 1
        
        if T < T_min:
            # Restart with higher T? Or just stop?
            # Given time constraints, let's just stop if T is too low to avoid getting stuck.
            # But we have 140s, so we can afford multiple restarts.
            # Let's do a simple restart: go back to best known state and reset T.
            current_order = order_from_scores(best_scores)
            current_size = greedy_size_from_order(current_order)
            T = 10.0
            # We don't need to increment iter_count for the restart logic itself, 
            # but we should be careful not to loop forever.
            # Let's just let the main loop continue.
    
    # Final check: ensure we return the scores corresponding to the best order found
    # If best_size is the same as initial, return initial scores (which are consistent)
    # Otherwise, return the rank-based scores of the best order
    
    # It's safer to just return the scores that produce the best order we found.
    # If we updated best_scores, they are rank-based.
    # If we didn't, best_scores are -degrees.
    
    return best_scores.astype(float)

# EVOLVE-BLOCK-END
