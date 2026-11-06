"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Return a score per vertex for the greedy independent set decoder.
    
    Strategy: Model the greedy selection as a dynamic process where vertex priority
    is determined by marginal contribution. We simulate the greedy process with
    state-dependent scoring based on the ratio of current degree to remaining
    available vertices, capturing how each vertex's value changes as the set grows.
    
    Key structural change: Instead of static features, we use a forward simulation
    that tracks how the graph evolves as vertices are removed, and scores each
    vertex by its "efficiency" - how many vertices it blocks relative to the
    shrinking pool of available vertices.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Initial degrees
    degrees = adjacency.sum(axis=1).astype(float)
    
    # We'll simulate the greedy process and accumulate "efficiency scores"
    # For each vertex, track when it would be selected and its marginal value
    
    # Multiple simulations with different tie-breaking to get robust scores
    num_sims = 5
    accumulated_score = np.zeros(n, dtype=float)
    
    for sim in range(num_sims):
        if time.time() > deadline - 10:
            break
        
        # Create a working copy of adjacency
        adj = adjacency.copy()
        available = np.ones(n, dtype=bool)
        
        # Base priority: negative degree (lower degree preferred)
        # Add small perturbation for diversity across simulations
        rng = np.random.default_rng(42 + sim)
        perturbation = rng.normal(0, 0.5, n)
        
        # Dynamic scoring: we'll track the "efficiency" of each potential selection
        # At each step, the value of picking vertex v is:
        #   -1 (it gets added to IS)
        #   - (number of available neighbors it blocks) / (remaining available vertices)
        # We want vertices that block FEWER available vertices relative to pool size
        # So we want to MINIMIZE the blocking ratio, or equivalently MAXIMIZE (1 - ratio)
        
        sim_scores = np.zeros(n, dtype=float)
        step = 0
        
        while True:
            if time.time() > deadline - 5:
                break
            
            # Find available vertices
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Current degrees among available vertices
            current_deg = np.zeros(n, dtype=float)
            for i in avail_indices:
                current_deg[i] = adj[i, available].sum()
            
            # Compute dynamic score for each available vertex:
            # efficiency = 1 - (blocked_available / total_available)
            # Where blocked_available = number of available neighbors
            # We want high efficiency (low blocking ratio)
            
            total_avail = len(avail_indices)
            
            # For each available vertex, compute its blocking impact
            # blocked_v = number of available neighbors of v
            blocked = np.zeros(n, dtype=float)
            for i in avail_indices:
                blocked[i] = current_deg[i]
            
            # Efficiency: 1 - blocked/total_avail (higher is better)
            # Add base preference for low degree
            dynamic_score = (1.0 - blocked / max(total_avail, 1)) - 0.1 * current_deg
            
            # Select the vertex with highest dynamic score
            # Among available vertices
            scores_for_avail = dynamic_score[avail_indices]
            best_local_idx = np.argmax(scores_for_avail)
            v = avail_indices[best_local_idx]
            
            # Record this vertex's efficiency score
            sim_scores[v] = dynamic_score[v]
            
            # Remove v and its neighbors from available
            available[v] = False
            neighbors = np.where(adj[v])[0]
            available[neighbors] = False
            adj[v, :] = False
            adj[:, v] = False
            
            step += 1
            
            # Early termination if we've selected enough
            if step >= n // 2:
                break
        
        accumulated_score += sim_scores
    
    # Average across simulations
    num_completed = min(num_sims, max(1, int((deadline - 10 - start_time) / (time.time() - start_time + 1e-9)) if time.time() > start_time else num_sims))
    if num_completed > 0:
        final_scores = accumulated_score / num_completed
    else:
        # Fallback to static score
        final_scores = -degrees
    
    # Add a small static component to break remaining ties
    mean_nbr_deg = np.zeros(n, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_nbr_deg = np.where(degrees > 0, (adjacency @ degrees) / degrees, 0.0)
    
    final_scores += 0.1 * mean_nbr_deg
    
    return final_scores
# EVOLVE-BLOCK-END
