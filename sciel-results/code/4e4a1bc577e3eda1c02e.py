"""Program for Maximum Independent Set via parallel min-degree exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel min-degree exclusion process.
    Improved tie-breaking: prefer vertices with lower global degree among candidates.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    global_deg = adj.sum(axis=1)
    
    scores = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    
    round_id = 0
    base_score = n + 1.0
    
    while active.any():
        # Calculate degrees of active vertices considering only active neighbors
        active_mask = active.astype(np.int32)
        deg = (adj * active_mask).sum(axis=1)
        
        # Only consider active vertices
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        active_deg = deg[active_indices]
        min_deg = active_deg.min()
        
        # Find candidates: active vertices with min degree
        candidate_indices = active_indices[active_deg == min_deg]
        
        if len(candidate_indices) == 0:
            break
            
        # Sort candidates: Primary key: Global Degree (ascending), Secondary key: Index (for stability)
        # Lower global degree is often "safer" as it constrains fewer other vertices in the original graph
        global_deg_candidates = global_deg[candidate_indices]
        sorted_order = np.argsort(global_deg_candidates)
        sorted_candidates = candidate_indices[sorted_order]
        
        # Greedy Maximal Independent Set on sorted_candidates
        selected_globals = []
        for v in sorted_candidates:
            # Check if v is adjacent to any already selected vertex in this round
            if selected_globals:
                if adj[v, selected_globals].any():
                    continue
            selected_globals.append(v)
            
        if not selected_globals:
            break
            
        # Assign scores to selected vertices
        score_val = base_score - round_id
        for v in selected_globals:
            scores[v] = score_val
            
        # Identify neighbors of selected vertices to remove and assign lower scores
        neighbor_mask = adj[selected_globals, :].any(axis=0)
        
        # Update active: remove selected and their neighbors
        active[selected_globals] = False
        active[neighbor_mask] = False
        
        # Assign scores to neighbors (lower than selected, higher than next round)
        # We only assign to neighbors that are still active (or were active) and haven't been scored
        # Note: neighbor_mask includes selected vertices too, but they are already scored.
        # We should only assign to neighbors that are NOT selected.
        neighbor_indices = np.where(neighbor_mask)[0]
        # Exclude selected vertices from neighbor scoring
        selected_set = set(selected_globals)
        for u in neighbor_indices:
            if u not in selected_set and scores[u] == 0.0: 
                scores[u] = score_val - 0.1
        
        round_id += 1
        
    return scores
# EVOLVE-BLOCK-END
