<<<<<<< SEARCH
def _greedy_grow(u, deadline):
    """Greedy growth of the set."""
    u = sorted(set(u))
    if 0 not in u:
        u = [0] + u
        u = sorted(set(u))
    
    current_score = _compute_score(u)
    best_score = current_score
    best_u = list(u)
    
    # Limit the number of growth steps to avoid infinite loops or excessive time
    max_steps = 100
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        max_u = max(u)
        # Candidate range: from 1 to 2 * max_u + 10
        # To keep it fast, we might limit this if max_u is very large
        upper_bound = 2 * max_u + 10
        if upper_bound > 1000:
            upper_bound = 1000 # Cap for safety
            
        candidates = [x for x in range(1, upper_bound + 1) if x not in u]
        
        if not candidates:
            break
            
        best_candidate = None
        best_candidate_score = current_score
        
        # Evaluate each candidate
        for c in candidates:
            new_u = sorted(u + [c])
            s = _compute_score(new_u)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = new_u
                
        if best_candidate is not None:
            u = best_candidate
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best_u = list(u)
        else:
            break
            
    return best_u, best_score
=======
def _greedy_grow(u, deadline):
    """Greedy growth of the set with smarter candidate selection."""
    u = sorted(set(u))
    if 0 not in u:
        u = [0] + u
        u = sorted(set(u))
    
    current_score = _compute_score(u)
    best_score = current_score
    best_u = list(u)
    
    # Limit the number of growth steps to avoid infinite loops or excessive time
    max_steps = 150
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        max_u = max(u)
        # Candidate range: from 1 to 3 * max_u + 20 for wider exploration
        upper_bound = 3 * max_u + 20
        if upper_bound > 2000:
            upper_bound = 2000 # Cap for safety
            
        # Prioritize candidates: near existing elements, in gaps, and new extremes
        candidates = set()
        
        # Near existing elements (within ±5)
        for x in u:
            for d in range(-5, 6):
                if d != 0:
                    cand = x + d
                    if 1 <= cand <= upper_bound and cand not in u:
                        candidates.add(cand)
        
        # In gaps between consecutive elements
        for i in range(len(u) - 1):
            gap_start = u[i] + 1
            gap_end = u[i + 1] - 1
            if gap_end >= gap_start:
                # Add midpoint and quartiles
                candidates.add((gap_start + gap_end) // 2)
                candidates.add(gap_start + (gap_end - gap_start) // 4)
                candidates.add(gap_start + 3 * (gap_end - gap_start) // 4)
                candidates.add(gap_start)
                candidates.add(gap_end)
        
        # New extremes
        candidates.add(max_u + 1)
        candidates.add(max_u + 2)
        candidates.add(max_u + 5)
        candidates.add(max_u + 10)
        
        candidates = sorted(candidates)
        
        if not candidates:
            break
            
        best_candidate = None
        best_candidate_score = current_score
        
        # Evaluate each candidate
        for c in candidates:
            if c in u or c <= 0:
                continue
            new_u = sorted(u + [c])
            s = _compute_score(new_u)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = new_u
                
        if best_candidate is not None:
            u = best_candidate
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best_u = list(u)
        else:
            break
            
    return best_u, best_score
>>>>>>> REPLACE

