"""Program for Maximum Independent Set via parallel discrete-time exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel discrete-time exclusion process.
    
    In each time step, we identify vertices with the fewest active neighbors.
    Among these, we greedily select a maximal independent set.
    Vertices selected at earlier time steps get higher priority scores.
    This captures dynamic interference patterns as the graph evolves.
    
    Key insights:
    1. Vertices with fewer active neighbors are less constrained and can be selected earlier.
    2. Earlier selection means higher priority, as these vertices have less interference.
    3. The process naturally handles local density variations.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    scores = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    
    # We'll assign scores based on time step (earlier = higher score)
    # Use a large base so that earlier steps get higher scores
    base_score = float(n) + 10.0
    time_step = 0
    max_steps = n  # Safety bound
    
    while active.any() and time_step < max_steps:
        # Compute degrees within the active subgraph
        active_float = active.astype(np.float64)
        deg = (adj * active_float).sum(axis=1)
        
        # Get active vertices and their degrees
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        active_degrees = deg[active_indices]
        
        # Find the minimum degree among active vertices
        min_deg = active_degrees.min()
        
        # Candidates: active vertices with minimum degree
        candidate_mask = active_degrees == min_deg
        candidates = active_indices[candidate_mask]
        
        if len(candidates) == 0:
            break
        
        # Among candidates, sort by their original degree (ascending) to prefer less connected vertices
        # This helps break ties in a way that favors vertices with fewer total connections
        original_degrees = deg[candidates]
        sorted_candidates = candidates[np.argsort(original_degrees)]
        
        # Greedy selection: pick a maximal independent set from sorted_candidates
        selected = []
        for v in sorted_candidates:
            # Check if v is adjacent to any already selected vertex in this round
            if selected:
                if adj[v, selected].any():
                    continue
            selected.append(v)
        
        if not selected:
            break
        
        selected_arr = np.array(selected)
        
        # Assign scores: earlier time steps get higher scores
        # Use a smooth decay to distinguish time steps
        score_value = base_score - time_step * 1.0
        
        for v in selected_arr:
            scores[v] = score_value
        
        # Remove selected vertices and their neighbors from active set
        # First, mark selected as inactive
        active[selected_arr] = False
        
        # Find all neighbors of selected vertices
        neighbor_mask = adj[selected_arr, :].any(axis=0)
        active[neighbor_mask] = False
        
        # Assign lower scores to neighbors (they were blocked by this round's selections)
        # These get scores slightly below the current time step
        blocked = np.where(neighbor_mask)[0]
        for u in blocked:
            if scores[u] == 0.0:  # Only assign if not already assigned
                scores[u] = score_value - 0.5
        
        time_step += 1
    
    # Any remaining vertices (shouldn't happen, but safety) get low scores
    remaining = np.where(active)[0]
    if len(remaining) > 0:
        for i, v in enumerate(remaining):
            scores[v] = base_score - time_step * 1.0 - i * 0.1
    
    return scores
# EVOLVE-BLOCK-END
