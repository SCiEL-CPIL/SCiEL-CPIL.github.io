import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s limit
    
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1)
    
    # Mean neighbor degree
    # For each vertex v, mean degree of its neighbors
    # adjacency @ deg gives sum of neighbor degrees
    neighbor_deg_sum = adjacency @ deg
    mean_neighbor_deg = np.zeros(n)
    mask = deg > 0
    mean_neighbor_deg[mask] = neighbor_deg_sum[mask] / deg[mask]
    
    # Base heuristic
    alpha = 0.1
    base_score = -deg.astype(float) + alpha * mean_neighbor_deg
    
    # Sort vertices by base score descending
    order = np.argsort(-base_score)
    
    # Simulate greedy MIS for each vertex as first pick
    sim_score = np.zeros(n)
    
    for i, v in enumerate(order):
        if time.time() > deadline - 30.0:
            # Not enough time; use base score for remaining
            break
        
        # Simulate: pick v first, then greedily pick rest by base_score
        picked = np.zeros(n, dtype=bool)
        picked[v] = True
        # Mark neighbors of v as unavailable (they won't be picked)
        neighbors = np.where(adjacency[v])[0]
        unavailable = np.zeros(n, dtype=bool)
        unavailable[neighbors] = True
        unavailable[v] = True  # v is picked, not available
        
        count = 1  # v is picked
        
        # Greedily pick remaining vertices
        available = ~unavailable
        while available.any():
            if time.time() > deadline - 30.0:
                break
            # Find available vertex with highest base_score
            scores = np.where(available, base_score, -np.inf)
            u = np.argmax(scores)
            if not available[u]:
                break
            picked[u] = True
            count += 1
            # Remove u and its neighbors from available
            available[u] = False
            nbrs = np.where(adjacency[u])[0]
            available[nbrs] = False
        
        sim_score[v] = count
    
    # For vertices not simulated, use base_score as a proxy
    # Convert base_score to a comparable scale
    # base_score is negative (since -deg dominates), sim_score is positive (count)
    # We need to combine them. Let's use sim_score where available, and a scaled base_score otherwise.
    
    # Actually, let's just use sim_score for simulated vertices and for others,
    # estimate using base_score normalized.
    
    # Simpler: use sim_score for all, but for un-simulated, use a fallback.
    # Let's check how many were simulated.
    simulated = np.zeros(n, dtype=bool)
    simulated[order[:i+1]] = True  # approximately
    
    # For un-simulated, use base_score scaled to be comparable
    # base_score range: roughly -max_deg to 0
    # sim_score range: roughly n/2 to n
    # Let's normalize base_score to [0, 1] and multiply by n
    if base_score.max() > base_score.min():
        base_norm = (base_score - base_score.min()) / (base_score.max() - base_score.min())
    else:
        base_norm = np.zeros(n)
    fallback = base_norm * n  # scale to comparable range
    
    final_score = np.where(simulated, sim_score, fallback)
    
    return final_score
