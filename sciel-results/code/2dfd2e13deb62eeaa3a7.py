# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Computes a priority score for each vertex using an ensemble of
    randomized greedy simulations.
    
    Approach:
    1. Run the greedy MIS selection K times.
    2. In each run, use the current degree in the remaining graph.
    3. Break ties with random noise to explore different selection orders.
    4. Score each vertex based on the average step at which it was selected.
       Earlier selection implies higher importance/robustness.
    5. Return the average score.
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
        
    N = adjacency.shape[0]
    deg = adjacency.sum(axis=1)
    
    # Pre-compute neighbor lists for efficiency
    neighbors = [np.where(adjacency[i] == 1)[0] for i in range(N)]
    
    K = 100  # Number of simulations
    total_scores = np.zeros(N)
    
    for _ in range(K):
        total_removed = np.zeros(N, dtype=int)
        active = np.ones(N, dtype=bool)
        steps = np.full(N, np.inf)
        
        for t in range(N):
            if not np.any(active):
                break
            
            # Calculate current degrees: initial degree minus removed neighbors
            cur_deg = deg - (adjacency @ total_removed)
            
            # Mask inactive vertices
            masked_deg = np.where(active, cur_deg, np.inf)
            
            # Add small random noise to break ties and diversify runs
            masked_deg += np.random.rand(N) * 1e-6
            
            v = np.argmin(masked_deg)
            
            if masked_deg[v] == np.inf:
                break
                
            steps[v] = t
            
            # Remove v and its neighbors
            active[v] = False
            total_removed[v] = 1
            
            nbs = neighbors[v]
            if len(nbs) > 0:
                active[nbs] = False
                total_removed[nbs] = 1
            
        # Accumulate scores
        # Score = 1 / (step + 1) for selected vertices
        selected_mask = steps < np.inf
        if np.any(selected_mask):
            run_scores = np.zeros(N)
            run_scores[selected_mask] = 1.0 / (steps[selected_mask] + 1.0)
            total_scores += run_scores
            
    final_scores = total_scores / K
    
    return final_scores
# EVOLVE-BLOCK-END
