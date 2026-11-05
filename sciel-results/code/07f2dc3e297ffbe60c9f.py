import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Local Search Optimization with Fast Simulation
    1. Initialize scores based on a strong static heuristic (Low Degree + Hub Penalty).
    2. Simulate the greedy decoding to get a baseline Independent Set size.
    3. Iteratively attempt to improve the set size by swapping the priority scores of pairs of vertices.
    4. Return the final scores.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic: Low Degree + Hub Penalty
    deg = adj.sum(axis=1)
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    hub_penalty = neighbor_deg_sum / safe_deg
    scores = -deg.astype(float) - 0.1 * hub_penalty
    
    def simulate_fast(s):
        """Fast greedy simulation using numpy boolean array."""
        order = np.argsort(-s)
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count
    
    # 2. Baseline
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # 3. Local Search with time guard
    rng = np.random.default_rng(42)
    max_time = 115.0  # Safety margin before 120s limit
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < max_time and no_improve < max_no_improve:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        # Swap
        best_scores[i], best_scores[j] = best_scores[j], best_scores[i]
        new_val = simulate_fast(best_scores)
        
        if new_val > best_val:
            best_val = new_val
            no_improve = 0
        else:
            best_scores[i], best_scores[j] = best_scores[j], best_scores[i]
            no_improve += 1
    
    return best_scores
