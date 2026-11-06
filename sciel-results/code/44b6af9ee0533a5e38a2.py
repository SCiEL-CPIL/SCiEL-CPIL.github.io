<<<<<<< SEARCH
def _local_search(S, pair_conflicts, rng, deadline):
    """
    Local search with removal and swap moves to minimize |S| while maintaining
    validity and maximality.
    """
    S = list(S)
    improved = True
    iters = 0
    max_iters = 200
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > deadline:
            break
        
        # Try removals first
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            trial_S = S[:idx] + S[idx+1:]
            if not trial_S:
                continue
            # Check validity: no isosceles triangle in trial_S
            valid = True
            for i in range(len(trial_S)):
                for j in range(i + 1, len(trial_S)):
                    a, b = trial_S[i], trial_S[j]
                    for k in trial_S:
                        if k != a and k != b:
                            if k in pair_conflicts[a, b]:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            # Check maximality
            if _is_maximal(trial_S, pair_conflicts):
                S = trial_S
                improved = True
                break
        
        if improved or time.time() > deadline:
            continue
        
        # Try swaps: remove one point, add another
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            S_without = S[:idx] + S[idx+1:]
            
            # Find points that can be added to S_without
            S_set = set(S_without)
            addable = []
            for p in range(V):
                if p in S_set:
                    continue
                if _can_add(p, S_without, pair_conflicts):
                    addable.append(p)
            
            for p in addable:
                if time.time() > deadline:
                    break
                trial_S = S_without + [p]
                # Check validity
                valid = True
                for i in range(len(trial_S)):
                    for j in range(i + 1, len(trial_S)):
                        a, b = trial_S[i], trial_S[j]
                        for k in trial_S:
                            if k != a and k != b:
                                if k in pair_conflicts[a, b]:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Check maximality
                if _is_maximal(trial_S, pair_conflicts):
                    S = trial_S
                    improved = True
                    break
            
            if improved:
                break
    
    return S
=======
def _is_valid(S, pair_conflicts):
    """Check if S is a valid no-isosceles set."""
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            # If any other point in S is in conflict[a,b], it's invalid
            # conflict[a,b] contains c such that {a,b,c} is isosceles.
            # So we check if (S_mask & conflict[a,b]) has any bits set other than a and b.
            # But a and b are not in conflict[a,b] by construction.
            # So we just check if S_mask & conflict[a,b] is non-zero.
            # To do this efficiently, we need S_mask.
            pass # This logic is slightly inefficient if we don't have S_mask.
    
    # Better: Precompute S_mask
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
        
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            # Check if any point in S is in conflict[a,b]
            # Since a,b are not in conflict[a,b], we just check intersection
            if S_mask & pair_conflicts[a][b]:
                return False
    return True

def _local_search(S, pair_conflicts, rng, deadline):
    """
    Local search with removal and swap moves to minimize |S| while maintaining
    validity and maximality.
    """
    S = list(S)
    improved = True
    iters = 0
    max_iters = 100 # Reduced iterations due to faster checks
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > deadline:
            break
        
        # Try removals first
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            trial_S = S[:idx] + S[idx+1:]
            if not trial_S:
                continue
            
            # Check validity
            if not _is_valid(trial_S, pair_conflicts):
                continue
            
            # Check maximality
            if _is_maximal(trial_S, pair_conflicts):
                S = trial_S
                improved = True
                break
        
        if improved or time.time() > deadline:
            continue
        
        # Try swaps: remove one point, add another
        # To speed up, we can precompute the "cannot_add_mask" for S_without
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            S_without = S[:idx] + S[idx+1:]
            
            # Calculate cannot_add_mask for S_without
            if not S_without:
                cannot_add_mask = 0
            else:
                cannot_add_mask = 0
                n_wo = len(S_without)
                for i in range(n_wo):
                    a = S_without[i]
                    for j in range(i + 1, n_wo):
                        b = S_without[j]
                        cannot_add_mask |= pair_conflicts[a][b]
            
            # Candidates for addition: points not in S_without and not in cannot_add_mask
            S_wo_mask = 0
            for p in S_without:
                S_wo_mask |= (1 << p)
            
            candidates_mask = ((1 << V) - 1) & ~S_wo_mask & ~cannot_add_mask
            
            if not candidates_mask:
                continue
                
            # Extract candidates
            candidates = []
            temp_mask = candidates_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx_c = lsb.bit_length() - 1
                candidates.append(idx_c)
                temp_mask ^= lsb
                
            # Try a few random candidates
            if len(candidates) > 10:
                sampled = rng.sample(candidates, 10)
            else:
                sampled = candidates
                
            for p in sampled:
                if time.time() > deadline:
                    break
                trial_S = S_without + [p]
                
                # Check validity
                if not _is_valid(trial_S, pair_conflicts):
                    continue
                
                # Check maximality
                if _is_maximal(trial_S, pair_conflicts):
                    S = trial_S
                    improved = True
                    break
            
            if improved:
                break
    
    return S
>>>>>>> REPLACE

