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
            
        # Calculate degrees within the candidate set to prefer vertices with fewer conflicts
        # This helps in selecting a larger independent set in the current round
        internal_degrees = np.zeros(len(candidate_indices), dtype=int)
        for i, v in enumerate(candidate_indices):
            # Count neighbors of v that are also in candidate_indices
            # adj[v, candidate_indices] gives boolean mask of neighbors in candidates
            internal_degrees[i] = np.sum(adj[v, candidate_indices])
            
        # Sort by internal degree ascending (prefer isolated/low-conflict candidates)
        # If internal degrees are equal, fallback to global degree (though they are all min_deg)
        # or just stable sort
        sorted_order = np.argsort(internal_degrees)
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
        neighbors_to_score = np.where(neighbor_mask)[0]
        for u in neighbors_to_score:
            # Ensure we don't overwrite a score if it was somehow set (shouldn't happen)
            if scores[u] == 0.0: 
                scores[u] = score_val - 0.1
        
        round_id += 1
        
    return scores
# EVOLVE-BLOCK-END
