import time

def construct():
    start_time = time.time()
    deadline = start_time + 50

    # Baseline: {63, ..., 128} has size 66 and is valid.
    # Sums of two distinct elements >= 63+64=127.
    # Possible sums in set: {127, 128}.
    # Need 3 distinct sums for a violation. Impossible.
    baseline = list(range(63, 129))
    
    # Try to find a set of size 67 or more.
    # Candidate: {61, ..., 128} has size 68.
    # We need to remove elements to break all violating triples.
    
    cand = list(range(61, 129))
    cand_set = set(cand)
    
    # Find all violating triples in cand
    # a < b < c
    # a+b, a+c, b+c in cand_set
    # Since cand is an interval [61, 128], sums are in cand iff sum <= 128.
    # So we need b+c <= 128.
    
    bad_triples = []
    # Optimization: iterate a, b, c such that b+c <= 128
    # c <= 128 - b
    # b < c => b < 128 - b => 2b < 128 => b < 64
    # So b can be at most 63.
    # a < b => a <= 62.
    # a >= 61.
    
    for b in range(62, 64): # b must be < 64
        # a in [61, b-1]
        for a in range(61, b):
            # c in [b+1, 128-b]
            max_c = 128 - b
            if max_c <= b:
                continue
            for c in range(b + 1, max_c + 1):
                # Check sums
                if (a + b in cand_set) and (a + c in cand_set) and (b + c in cand_set):
                    bad_triples.append((a, b, c))
    
    # If no bad triples, cand is valid
    if not bad_triples:
        return cand
        
    # Try to remove 1 element to break all bad triples
    elements_in_bad = set()
    for t in bad_triples:
        elements_in_bad.update(t)
        
    for e in elements_in_bad:
        # Check if removing e breaks all bad triples
        is_broken = True
        for t in bad_triples:
            if e not in t:
                is_broken = False
                break
        if is_broken:
            new_set = cand_set - {e}
            return list(new_set)
            
    # Try to remove 2 elements
    # Only do this if the list of bad triples is small
    if len(bad_triples) < 200:
        pairs = [(e1, e2) for e1 in elements_in_bad for e2 in elements_in_bad if e1 < e2]
        for e1, e2 in pairs:
            is_broken = True
            for t in bad_triples:
                if e1 not in t and e2 not in t:
                    is_broken = False
                    break
            if is_broken:
                new_set = cand_set - {e1, e2}
                return list(new_set)
                
    # Fallback to baseline
    return baseline

def is_valid(S):
    # This function is kept for compatibility but not used in the optimized path
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i+1, n):
            for l in range(j+1, n):
                a, b, c = S_list[i], S_list[j], S_list[l]
                if a+b in S_set and a+c in S_set and b+c in S_set:
                    return False
    return True
