<<<<<<< SEARCH
def _local_search(code, deadline):
    """Try 1-swap moves: remove one codeword, add another if coverage maintained and size decreases."""
    if not code:
        return code
    
    code_set = set(code)
    improved = True
    while improved:
        improved = False
        # Compute coverage
        covered = np.zeros(NUM_WORDS, dtype=bool)
        for c in code:
            covered[ball_indices[c]] = True
        
        # For each codeword in code, try removing it
        for i in range(len(code)):
            if time.time() > deadline:
                return code
            c = code[i]
            # Check what would become uncovered if we remove c
            # Words only covered by c: those in ball[c] with coverage count == 1
            cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
            for cc in code:
                cov_counts[ball_indices[cc]] += 1
            
            only_covered = ball_indices[c][cov_counts[ball_indices[c]] == 1]
            if len(only_covered) == 0:
                # Can remove c entirely
                code.pop(i)
                improved = True
                break
            else:
                # Try to find a codeword that covers all of only_covered
                # and is not already in code
                # Score for each candidate: number of only_covered words it covers
                # We want a candidate that covers ALL of only_covered
                # and we'd gain by removing c and adding candidate if net size doesn't increase
                # Actually we want to replace c with a better codeword
                # For now, just try removal if possible
                
        if not improved:
            # Try 1-exchange: remove one, add one different
            # This is more complex; skip for now to save time
            break
    
    return code
=======
def _local_search(code, deadline):
    """Try removals and 1-exchanges to reduce code size while maintaining coverage."""
    if not code:
        return code
    
    code_set = set(code)
    
    # Compute coverage counts once
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        cov_counts[ball_indices[c]] += 1
    
    improved = True
    while improved:
        improved = False
        
        # Phase A: Remove any redundant codeword
        for i in range(len(code)):
            if time.time() > deadline:
                return code
            c = code[i]
            if np.all(cov_counts[ball_indices[c]] > 1):
                cov_counts[ball_indices[c]] -= 1
                code.pop(i)
                code_set.discard(c)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase B: Try 1-exchange: remove c, add d, if coverage maintained and size same or smaller
        # Only useful if we can remove c and the words it uniquely covers are all covered by some d
        # Try random pairs
        if len(code) > 10:  # Only do exchanges if code is large enough
            for _ in range(100):
                if time.time() > deadline:
                    return code
                i = random.randint(0, len(code) - 1)
                c = code[i]
                
                # Words uniquely covered by c
                bi = ball_indices[c]
                only_c = bi[cov_counts[bi] == 1]
                if len(only_c) == 0:
                    continue  # Would have been caught in Phase A
                
                # Find candidates d not in code that cover all of only_c
                # For each d not in code, check if ball_indices[d] contains all of only_c
                # This is expensive; limit to a random subset
                d_candidates = random.sample(range(NUM_WORDS), min(50, NUM_WORDS))
                d_candidates = [d for d in d_candidates if d not in code_set]
                
                for d in d_candidates:
                    if time.time() > deadline:
                        return code
                    # Check if d covers all of only_c
                    if np.all(np.isin(only_c, ball_indices[d])):
                        # Check coverage after swap
                        # Remove c: cov_counts[ball_indices[c]] -= 1
                        # Add d: cov_counts[ball_indices[d]] += 1
                        # Need all words still covered
                        new_cov = cov_counts.copy()
                        new_cov[ball_indices[c]] -= 1
                        new_cov[ball_indices[d]] += 1
                        if np.all(new_cov > 0):
                            # Valid swap
                            cov_counts = new_cov
                            code.pop(i)
                            code_set.discard(c)
                            code.append(d)
                            code_set.add(d)
                            improved = True
                            break
                if improved:
                    break
    
    return code
>>>>>>> REPLACE
