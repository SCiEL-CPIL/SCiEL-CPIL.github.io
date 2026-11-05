"""Program for Maximum Independent Set via parallel min-degree exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel min-degree exclusion process.
    In each round, vertices with the minimum degree are identified.
    A maximal independent set is greedily selected from these candidates.
    Selected vertices and their neighbors are removed from the active set.
    Scores are assigned such that vertices selected in earlier rounds have higher priority,
    and within a round, selected vertices have higher priority than their neighbors.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    
    # Precompute original degrees for tie-breaking
    orig_deg = adj.sum(axis=1)
    
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
            
        # Sort candidates by original degree (ascending) to prefer less connected vertices
        # This helps in "sacrificing" fewer neighbors in terms of global impact
        candidate_orig_degrees = orig_deg[candidate_indices]
        sorted_order = np.argsort(candidate_orig_degrees)
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
        # Add a tiny penalty based on original degree to ensure consistent ordering in decoder
        # if there are any ties or dependencies (though they are independent here)
        score_val = base_score - round_id
        for i, v in enumerate(selected_globals):
            # Smaller original degree -> higher score (less penalty)
            # We use a small factor so it doesn't cross round boundaries
            penalty = orig_deg[v] * 0.001
            scores[v] = score_val - penalty
            
        # Identify neighbors of selected vertices to remove and assign lower scores
        neighbor_mask = adj[selected_globals, :].any(axis=0)
        
        # Update active: remove selected and their neighbors
        active[selected_globals] = False
        active[neighbor_mask] = False
        
        # Assign scores to neighbors (lower than selected, higher than next round)
        # We assign a score that is clearly lower than the minimum score in this round
        # and higher than the maximum score in the next round (base_score - (round_id+1))
        # Current round min score is approx base_score - round_id - max_orig_deg*0.001
        # Next round max score is approx base_score - (round_id + 1)
        # So score_val - 0.5 is a safe gap
        neighbors_to_score = np.where(neighbor_mask)[0]
        for u in neighbors_to_score:
            if scores[u] == 0.0: 
                scores[u] = score_val - 0.5
        
        round_id += 1
        
    return scores
# EVOLVE-BLOCK-END
