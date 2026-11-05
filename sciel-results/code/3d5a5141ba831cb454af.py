"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy: Dynamic exclusion simulation.
    Simulate multiple rounds of vertex selection where in each round,
    we tentatively select vertices with the fewest remaining neighbors
    (i.e., those most "available"). Vertices that remain unselected for
    more rounds are in low-density regions and get higher priority.
    This captures interference patterns that static degree misses.
    """
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=bool)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    deg = A.sum(axis=1).astype(float)
    
    # Dynamic exclusion simulation
    # We simulate R rounds of "tentative selection"
    # In each round, vertices with the fewest currently-active neighbors
    # are considered "most available" and get selected first.
    # We track how many rounds each vertex survives before being excluded.
    # Vertices that survive longer are in less contested regions.
    
    R = 50  # Number of simulation rounds
    
    # survival[v] = number of rounds v remains "available" (not excluded)
    survival = np.zeros(N, dtype=float)
    
    # active[v] = whether v is still in the candidate pool
    active = np.ones(N, dtype=bool)
    # cur_deg[v] = current degree in the active subgraph
    cur_deg = deg.copy()
    
    for round_idx in range(R):
        if time.time() - start_time > DEADLINE:
            break
        
        if not active.any():
            break
        
        # In this round, find vertices with minimum cur_deg among active
        # These are the "most available" - they have fewest conflicts
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
        
        active_deg = cur_deg[active_indices]
        min_deg = active_deg.min()
        
        # All active vertices with minimum degree are "selected" in this round
        # They get excluded along with their neighbors
        to_exclude = active_indices[active_deg == min_deg]
        
        # Mark survival: vertices that are still active get +1
        # Actually, let's think differently:
        # We want vertices that are HARD to exclude to have high priority.
        # A vertex is hard to exclude if it has few neighbors that get selected early.
        
        # Alternative approach: track when each vertex gets excluded
        # Vertices excluded later (by neighbors selected in later rounds) 
        # are in less dense regions.
        
        # Let's use: exclusion_round[v] = the round in which v's neighbors
        # force v to be excluded (i.e., the round when v loses its last 
        # "shield" or when a neighbor is selected that covers v)
        
        # Simpler: just count how many rounds v stays active.
        # Vertices that stay active longer are in sparser regions.
        
        # But we need to actually simulate exclusion.
        # In each round, select min-degree vertices, then exclude them
        # and their neighbors.
        
        # Exclude the selected vertices and their neighbors
        for v in to_exclude:
            if not active[v]:
                continue
            # v is selected - it will be in the independent set
            # Its neighbors are excluded
            neighbors = np.where(A[v, :])[0]
            for u in neighbors:
                if active[u]:
                    active[u] = False
                    cur_deg[u] -= 1  # v is removed
            # v itself is also removed from active
            active[v] = False
            # Update cur_deg for remaining active neighbors
            for u in neighbors:
                if active[u]:
                    cur_deg[u] -= 1
        
        # All still-active vertices get +1 survival
        survival[active] += 1.0
    
    # Now combine survival with degree
    # High survival = low density region = good for independent set
    # Low degree = good for independent set
    
    # Z-score normalize
    def zscore(x):
        m = x.mean()
        s = x.std()
        if s < 1e-9:
            return np.zeros_like(x)
        return (x - m) / s
    
    z_survival = zscore(survival)
    z_deg = zscore(deg)
    
    # Score: high survival and low degree are good
    # We want higher score = picked first
    score = 1.0 * z_survival - 0.8 * z_deg
    
    # Safety check
    if time.time() - start_time > DEADLINE:
        return -deg.astype(float)
    
    return score
# EVOLVE-BLOCK-END
