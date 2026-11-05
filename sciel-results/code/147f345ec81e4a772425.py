import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    adj = adjacency.astype(bool)
    
    # Precompute neighbor sets as boolean arrays for fast intersection
    # neighbors[i] is a boolean array of length n
    neighbors = [adj[i] for i in range(n)]
    
    # 1. Greedy Initialization: Lowest Degree First
    degrees = adj.sum(axis=1)
    init_scores = -degrees
    order = np.argsort(init_scores)[::-1]
    
    selected = []
    removed = np.zeros(n, dtype=bool)
    
    for v in order:
        if not removed[v]:
            selected.append(v)
            removed[v] = True
            removed |= neighbors[v]
            
    S_mask = np.zeros(n, dtype=bool)
    S_mask[selected] = True
    S_size = len(selected)
    
    # 2. Local Search Improvement
    # We will maintain S as a boolean mask
    # And a list of vertices in S for iteration
    
    def get_maximal_is_from_candidates(cand_mask):
        # cand_mask: boolean array, True if vertex is a candidate
        # Returns a list of vertices forming a maximal IS from candidates
        # using greedy lowest-degree
        
        # Get indices of candidates
        cands = np.where(cand_mask)[0]
        if len(cands) == 0:
            return []
        
        # Sort candidates by degree (lowest first)
        # Degree is global degree, which is a good proxy
        cands_sorted = cands[np.argsort(degrees[cands])]
        
        is_set = []
        local_removed = np.zeros(n, dtype=bool)
        
        for v in cands_sorted:
            if not local_removed[v]:
                is_set.append(v)
                local_removed[v] = True
                # Remove neighbors that are also candidates
                # We only care about removing neighbors within the candidate set
                # because non-candidates are already "blocked" by S_temp
                local_removed |= neighbors[v] & cand_mask
                
        return is_set

    improved = True
    # Limit iterations to avoid infinite loops or timeouts
    max_iters = 50
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Get list of vertices in S
        s_vertices = np.where(S_mask)[0]
        
        for u in s_vertices:
            # Remove u from S
            S_mask[u] = False
            
            # Find candidates:
            # v is candidate if:
            # 1. v is not in S_mask
            # 2. v is not adjacent to any vertex in S_mask
            # 
            # Adjacency to S_mask: 
            # We can compute this by ORing neighbors of all v in S_mask?
            # Or better: A vertex v is blocked if any neighbor is in S_mask.
            # blocked_by_S = np.zeros(n, dtype=bool)
            # for w in s_vertices (excluding u):
            #    blocked_by_S |= neighbors[w]
            
            # Efficient way:
            # We can precompute "covered" by S?
            # Let's just compute it.
            blocked = np.zeros(n, dtype=bool)
            for w in s_vertices:
                if w == u:
                    continue
                blocked |= neighbors[w]
            
            # Candidates are those not in S_mask and not blocked
            cand_mask = ~S_mask & ~blocked
            
            # Find maximal IS from candidates
            new_adds = get_maximal_is_from_candidates(cand_mask)
            
            # Check if improvement
            # Old size was S_size (before removing u)
            # New size would be (S_size - 1) + len(new_adds)
            # But S_mask currently has S_size - 1 vertices.
            # So if len(new_adds) > 1, we improve.
            # Actually, if len(new_adds) > 1, the new set size is S_size - 1 + len(new_adds) > S_size.
            
            if len(new_adds) > 1:
                # Update S
                for v in new_adds:
                    S_mask[v] = True
                improved = True
                break # Restart
            
            # If no improvement, put u back
            S_mask[u] = True
            
    # 3. Assign Scores
    scores = np.zeros(n)
    
    # Vertices in S get high scores
    s_list = np.where(S_mask)[0]
    # Order them by degree (lowest first) for consistency
    s_list = s_list[np.argsort(degrees[s_list])]
    
    for i, v in enumerate(s_list):
        scores[v] = 10000 - i
        
    # Others get low scores based on degree
    other_mask = ~S_mask
    other_indices = np.where(other_mask)[0]
    # We want the greedy decoder to pick good vertices from the rest.
    # So assign scores such that lower degree = higher score.
    # Score = -degree - offset
    for v in other_indices:
        scores[v] = -degrees[v] - 10000
        
    return scores.astype(float)
