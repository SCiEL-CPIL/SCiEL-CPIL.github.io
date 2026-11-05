"""Program for Maximum Independent Set via dynamic resource competition simulation."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Simulates a parallel discrete-time exclusion process.
    Vertices compete for selection based on their current active degree.
    In each time step, the set of vertices with the minimum active degree forms
    the 'competing resource pool'. Within this pool, vertices are ordered by their
    'interference potential' (degree within the pool) to maximize the number of
    non-conflicting selections in the current step.
    Scores reflect the earliest time step at which a vertex is successfully selected.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    scores = np.zeros(n, dtype=float)
    active_mask = np.ones(n, dtype=bool)
    
    # Base score high enough to ensure earlier rounds have higher priority
    # We use a large number minus the round index
    max_score = n * 2.0 + 100.0
    current_round = 0
    
    while active_mask.any():
        # 1. Calculate active degrees
        # deg[i] is number of active neighbors of i
        active_counts = active_mask.astype(np.int32)
        degrees = (adj * active_counts).sum(axis=1)
        
        # 2. Identify competing resources (min degree active vertices)
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            break
            
        active_degrees = degrees[active_indices]
        min_deg = active_degrees.min()
        
        # Candidates are active vertices with the minimum degree
        candidate_indices = active_indices[active_degrees == min_deg]
        
        if len(candidate_indices) == 0:
            break
            
        # 3. Analyze dynamic interference within the candidate pool
        # Create a mask for candidates to compute induced degrees
        candidate_mask = np.zeros(n, dtype=bool)
        candidate_mask[candidate_indices] = True
        
        # Induced degree: how many neighbors does each candidate have 
        # that are also in the candidate pool?
        # This measures the "local conflict" intensity.
        induced_degrees = (adj[candidate_indices] * candidate_mask).sum(axis=1)
        
        # 4. Order candidates to maximize selection count
        # Prefer candidates with lower induced degree (less conflict with other candidates)
        # This allows more vertices to be selected in this round before conflicts arise.
        sort_indices = np.argsort(induced_degrees)
        ordered_candidates = candidate_indices[sort_indices]
        
        # 5. Greedy selection from the ordered pool
        selected_in_round = []
        for v in ordered_candidates:
            # Check if v conflicts with any already selected vertex in this round
            if selected_in_round:
                # Vectorized check: is v adjacent to any in selected_in_round?
                if adj[v, selected_in_round].any():
                    continue
            selected_in_round.append(v)
            
        if not selected_in_round:
            # Safety break, though min degree 0 should always allow selection
            break
            
        # 6. Assign scores based on time step (round)
        # Earlier rounds get higher scores.
        score_val = max_score - current_round
        for v in selected_in_round:
            scores[v] = score_val
            
        # 7. Update active set (Exclusion Process)
        # Remove selected vertices and their neighbors from the active pool
        selected_array = np.array(selected_in_round)
        neighbor_mask = adj[selected_array, :].any(axis=0)
        
        # Mark selected and neighbors as inactive
        active_mask[selected_array] = False
        active_mask[neighbor_mask] = False
        
        current_round += 1
        
    return scores
# EVOLVE-BLOCK-END
