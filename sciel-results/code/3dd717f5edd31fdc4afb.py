def construct():
    import itertools
    import random
    import time

    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute pair sets for each triple for quick linearity check
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = frozenset({(a,b), (a,c), (b,c)})

    def try_cover(k, num_attempts=20):
        """
        Try to find a linear hypergraph covering all k-subsets.
        If successful, independence number <= k-1.
        """
        subsets = list(itertools.combinations(range(n), k))
        subset_to_idx = {s: i for i, s in enumerate(subsets)}
        num_subsets = len(subsets)
        
        # Precompute coverage: for each triple, which subset indices it covers
        triple_coverage = {}
        for t in all_triples:
            t_set = set(t)
            covered = []
            for i, s in enumerate(subsets):
                if t_set.issubset(set(s)):
                    covered.append(i)
            triple_coverage[t] = covered

        best_solution = None
        best_uncovered_count = num_subsets + 1
        
        for attempt in range(num_attempts):
            random.seed(attempt)
            
            selected = []
            used_pairs = set()
            uncovered = set(range(num_subsets))
            
            # To speed up, we can shuffle the order in which we consider candidates
            # but the greedy step picks the best one anyway.
            # We can add random tie-breaking.
            
            available_triples = all_triples[:]
            
            while uncovered:
                best_t = None
                best_gain = 0
                best_cov = []
                
                # Find the compatible triple with max gain
                # Optimization: Only check triples that are compatible
                for t in available_triples:
                    # Check linearity
                    pairs = triple_pairs[t]
                    if any(p in used_pairs for p in pairs):
                        continue
                    
                    # Calculate gain
                    cov = triple_coverage[t]
                    # Count how many are in uncovered
                    # Using set intersection is fast
                    gain = len(uncovered.intersection(cov))
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_t = t
                        best_cov = cov
                        if best_gain == len(uncovered): # Perfect cover, can break early? No, need to finish
                            pass

                if best_gain == 0:
                    break
                
                # Add the best triple
                selected.append(best_t)
                used_pairs.update(triple_pairs[best_t])
                
                # Remove covered subsets
                uncovered.difference_update(best_cov)
                
                # Remove best_t from available
                available_triples.remove(best_t)
                
            if not uncovered:
                return [list(t) for t in selected]
            
            if len(uncovered) < best_uncovered_count:
                best_uncovered_count = len(uncovered)
                best_solution = [list(t) for t in selected]

        return best_solution

    # Try to cover 6-subsets (alpha <= 5)
    res = try_cover(6, num_attempts=10)
    if res:
        return res
        
    # Try to cover 7-subsets (alpha <= 6)
    res = try_cover(7, num_attempts=10)
    if res:
        return res
        
    # Fallback: Return a greedy max-edge construction
    # Just to ensure we return *something* valid
    triples = []
    used_pairs = set()
    random.seed(42)
    shuffled = all_triples[:]
    random.shuffle(shuffled)
    for t in shuffled:
        pairs = triple_pairs[t]
        if not any(p in used_pairs for p in pairs):
            triples.append(list(t))
            used_pairs.update(pairs)
            
    return triples
