import itertools

P = {'n': 12, 'm': 40}

def construct():
    # Use a greedy approach on B_6 to maximize comparable pairs.
    # We start with the empty set and singletons, then greedily add sets
    # that create the most new comparable pairs with the existing family.
    
    vertices = list(range(6))
    
    # Generate all subsets of {0..5}
    all_sets = []
    for r in range(7):
        for combo in itertools.combinations(vertices, r):
            all_sets.append(frozenset(combo))
            
    # Precompute subset relations
    # is_subset[a][b] is True if a is a subset of b
    # To save memory/time, we can just check on the fly or precompute a matrix for 64 sets.
    n_sets = len(all_sets)
    index_map = {s: i for i, s in enumerate(all_sets)}
    
    # Precompute count of subsets/supersets?
    # Greedy:
    family = set()
    family.add(frozenset()) # Empty set
    
    # Add all singletons (they are always good)
    for v in vertices:
        family.add(frozenset([v]))
        
    # Current count of pairs
    # We will add sets one by one.
    # Candidate sets are the remaining ones.
    candidates = [s for s in all_sets if s not in family]
    
    for _ in range(P['m'] - len(family)):
        best_candidate = None
        best_gain = -1
        
        for c in candidates:
            # Calculate gain: number of sets in family that are subsets of c OR supersets of c
            # Since we start small and add larger, mostly we care about c being a superset of family members.
            # But c could be a subset of a family member?
            # We are adding from small to large generally.
            # Gain = |{f in family : f < c} | + |{f in family : f > c} |
            
            gain = 0
            for f in family:
                if f < c:
                    gain += 1
                elif c < f:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
        
        if best_candidate is not None:
            family.add(best_candidate)
            candidates.remove(best_candidate)
        else:
            break
            
    # Convert back to lists of ints
    result = []
    for s in family:
        result.append(sorted(list(s)))
        
    # Ensure we have exactly 40
    if len(result) < P['m']:
        # Fill with remaining sets if greedy failed (unlikely)
        for s in all_sets:
            if s not in family:
                family.add(s)
                result.append(sorted(list(s)))
                if len(result) == P['m']:
                    break
                    
    return result[:P['m']]
