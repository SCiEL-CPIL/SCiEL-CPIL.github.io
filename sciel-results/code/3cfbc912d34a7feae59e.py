import itertools

def construct():
    p = 101
    denominators = list(range(1, 11))
    
    # Precompute all non-empty subsets, their sums modulo p, and their lengths
    # We organize them to facilitate a greedy selection: prioritize shorter subsets
    candidates = []
    for k in range(1, 11):
        for comb in itertools.combinations(denominators, k):
            s = sum(pow(x, -1, p) for x in comb) % p
            candidates.append((k, s, list(comb)))
            
    # Sort candidates by length (k) ascending. 
    # This ensures we prioritize shorter subsets, which is the tie-breaker 
    # for the greedy set cover heuristic where "coverage count" is 1 for all valid picks.
    candidates.sort(key=lambda x: x[0])
    
    assigned = {}
    uncovered_count = 101
    
    # Greedy selection:
    # Iterate through candidates in order of preference (shorter first).
    # If a candidate covers an uncovered residue, assign it.
    for k, s, comb in candidates:
        if s not in assigned:
            assigned[s] = comb
            uncovered_count -= 1
            if uncovered_count == 0:
                break
                
    # Construct result
    result = [None] * 101
    for r in range(101):
        result[r] = assigned[r]
        
    return result
