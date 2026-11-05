import numpy as np
import time

def _greedy_decode(adj, scores):
    """
    Simulate the greedy decoder.
    Returns the size of the independent set.
    """
    n = adj.shape[0]
    selected = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    
    # Get indices sorted by score descending
    # Use stable sort for determinism if scores are equal
    order = np.argsort(-scores, kind='stable')
    
    count = 0
    for v in order:
        if not selected[v] and not blocked[v]:
            selected[v] = True
            count += 1
            # Block neighbors
            blocked[adj[v]] = True
            
    return count

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # 1. Compute Degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # 2. Compute Average Neighbor Degree
    nbr_deg_sums = adj @ degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)
        
    # 3. Compute Principal Eigenvector
    v1 = np.zeros(n)
    try:
        if time.time() < deadline - 10:
            eigvals, eigvecs = np.linalg.eigh(adj.astype(float))
            v1 = eigvecs[:, -1]
            max_val = np.max(np.abs(v1))
            if max_val > 0:
                v1 = v1 / max_val
    except Exception:
        v1 = np.zeros(n)

    # Initial Weights
    w_deg = 1.0
    w_nbr = 1.0
    w_spec = 0.5
    
    scores = -w_deg * degrees + w_nbr * avg_nbr_deg + w_spec * v1
    
    # Local Search to optimize weights
    # We treat the weights (w_deg, w_nbr, w_spec) as parameters to optimize
    # Objective: Maximize greedy_decode(adj, scores)
    
    best_scores = scores.copy()
    best_size = _greedy_decode(adj, best_scores)
    
    # Simple coordinate ascent / random restarts on weights
    # Since the function is non-smooth (greedy choice), we use discrete/random perturbations
    # or a simple grid search around the initial point.
    
    # Given the time budget, we can do a few hundred iterations of random perturbation
    # or a simple hill climbing.
    
    # Let's try a simple local search on the 3 weights.
    # We will try small perturbations in random directions.
    
    current_w = np.array([w_deg, w_nbr, w_spec])
    current_size = best_size
    
    # Time check: how much time do we have?
    # eigh took maybe 0.1s. We have ~140s left.
    # Each eval is O(N log N) for sort + O(N^2) for blocking. N=200 -> 40k ops. Very fast.
    # We can do many iterations.
    
    max_iters = 5000
    step_size = 0.5
    
    for i in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Random direction in weight space
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        new_w = current_w + step_size * direction
        
        # Ensure weights are not too negative or positive? 
        # Actually, negative weights are fine, they just invert preference.
        
        new_scores = -new_w[0] * degrees + new_w[1] * avg_nbr_deg + new_w[2] * v1
        new_size = _greedy_decode(adj, new_scores)
        
        if new_size > current_size:
            current_size = new_size
            current_w = new_w
            best_scores = new_scores
            best_size = current_size
            # Optional: adapt step size? Keep simple for now.
        # If we want to be more aggressive, we could also accept worse moves (simulated annealing)
        # but simple hill climbing on a 3D space should find a good local max quickly.
        
        # Also try swapping the sign of the spectral component?
        # The sign of v1 is arbitrary. 
        # If we flip v1, we might find a better independent set on the other side.
        # Let's add a toggle for the sign of v1 in the search space?
        # Or just try both signs in the initial setup?
        # Actually, let's just let the optimizer find the sign of w_spec.
        # If w_spec becomes negative, it effectively flips the sign.
        
    return best_scores
