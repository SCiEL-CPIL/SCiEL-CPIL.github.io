def construct():
    import random
    import math
    
    # 1. Enumerate forbidden triples
    triples = []
    for b in range(1, 181):
        for c in range(b + 1, 181):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a <= 180 and a != b and a != c:
                    triples.append(tuple(sorted((a, b, c))))
    
    triples = list(set(triples))
    num_triples = len(triples)
    
    # Map element to list of triple indices containing it
    adj = [[] for _ in range(181)]
    for i, t in enumerate(triples):
        for e in t:
            adj[e].append(i)
    
    # 2. Simulated Annealing
    # State
    in_set = [True] * 181  # 1-indexed, index 0 unused
    # triple_counts[i] = number of elements of triples[i] currently in_set
    triple_counts = [0] * num_triples
    violations = 0
    size = 0
    
    # Initialize
    for i, t in enumerate(triples):
        cnt = 0
        for e in t:
            if in_set[e]:
                cnt += 1
        triple_counts[i] = cnt
        if cnt == 3:
            violations += 1
    size = sum(in_set[1:])
    
    def get_score(s, v):
        # We want max s, min v.
        # If v > 0, heavily penalize.
        return s - 1000 * v
        
    current_score = get_score(size, violations)
    
    # Annealing parameters
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    iterations = 50000 # Needs to be fast enough
    
    best_size = size
    best_set = in_set[:]
    
    # We need to be careful with time. 50k iterations * small constant should be < 1s.
    # But we might need multiple restarts or longer runs.
    # Let's run one long SA run.
    
    for _ in range(iterations):
        if T < min_T:
            break
            
        # Pick random element 1..180
        x = random.randint(1, 180)
        
        # Calculate delta if we flip x
        # We will simulate the flip, check score, and revert if needed.
        
        old_size = size
        old_violations = violations
        
        if in_set[x]:
            # Remove x
            in_set[x] = False
            size -= 1
            for t_idx in adj[x]:
                triple_counts[t_idx] -= 1
                if triple_counts[t_idx] == 2: # Was 3, now 2 -> violation resolved
                    violations -= 1
        else:
            # Add x
            in_set[x] = True
            size += 1
            for t_idx in adj[x]:
                triple_counts[t_idx] += 1
                if triple_counts[t_idx] == 3: # Was 2, now 3 -> violation created
                    violations += 1
        
        new_score = get_score(size, violations)
        delta = new_score - current_score
        
        if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
            # Accept
            current_score = new_score
            if violations == 0 and size > best_size:
                best_size = size
                best_set = in_set[:]
        else:
            # Reject: Revert
            if in_set[x]:
                # We added x, so revert to remove
                in_set[x] = False
                size -= 1
                for t_idx in adj[x]:
                    triple_counts[t_idx] -= 1
                    if triple_counts[t_idx] == 2:
                        violations -= 1
            else:
                # We removed x, so revert to add
                in_set[x] = True
                size += 1
                for t_idx in adj[x]:
                    triple_counts[t_idx] += 1
                    if triple_counts[t_idx] == 3:
                        violations += 1
        
        T *= cooling
        
    # Ensure we return a valid set. 
    # If best_set has violations, we need to clean it up.
    # But SA with high penalty should end in valid states.
    # Let's verify best_set.
    # If best_set is invalid, fall back to greedy repair on best_set.
    
    final_set = best_set[:]
    
    # Check if final_set is valid
    # Reconstruct triple counts for final_set to be safe? 
    # Or just trust the tracking if we didn't revert incorrectly.
    # The tracking is local to the loop. best_set is a snapshot.
    # Let's do a quick validation and repair if needed.
    
    # Quick repair: while there are violations in final_set, remove one element from a violating triple.
    # This is cheap and ensures validity.
    
    # Build a set for fast lookup
    final_set_bool = [False] * 181
    for i in range(1, 181):
        if final_set[i]:
            final_set_bool[i] = True
            
    # Check violations
    # We can just iterate through triples.
    # If any triple is fully in final_set_bool, remove the smallest element (or random) to break it.
    changed = True
    while changed:
        changed = False
        for t in triples:
            if all(final_set_bool[e] for e in t):
                # Remove the element that appears in the most other active triples? 
                # Or just remove the smallest to keep large numbers?
                # Let's remove the smallest to match previous heuristic preference, or random.
                # Removing smallest is safe.
                remove_elem = min(t)
                final_set_bool[remove_elem] = False
                changed = True
                break # Restart loop as other triples might be affected? 
                # Actually, removing one element only affects triples containing it.
                # But `break` restarts the outer `while`, which re-checks all triples. 
                # This is O(N * T) in worst case, but N=180, T~1000. 180k ops. Fine.
                
    # Convert back to list
    result = [i for i in range(1, 181) if final_set_bool[i]]
    
    if not result:
        return [180]
        
    return result
