"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [1,22] such that ALL reciprocal subset sums
    (including empty sum = 0) are distinct. Maximize cardinality.
    
    Approach: Greedy construction with multiple orderings, then local search.
    """
    from fractions import Fraction
    import time
    
    start_time = time.time()
    TIMEOUT = 55  # Leave margin for 60s limit
    
    def check_time():
        return time.time() - start_time < TIMEOUT
    
    def is_dissociated_fast(subset):
        """Check if all subset sums of reciprocals are distinct, with early exit."""
        if not subset:
            return True
        recips = [Fraction(1, x) for x in subset]
        n = len(recips)
        sums = set()
        sums.add(Fraction(0, 1))
        for i in range(n):
            r = recips[i]
            new_sums = set()
            for s in sums:
                ns = s + r
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True
    
    all_elements = list(range(1, 23))
    best_set = []
    
    # Try multiple heuristics for ordering
    orderings = [
        sorted(all_elements),
        sorted(all_elements, reverse=True),
        [1, 2, 4, 8, 16, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 6, 10, 12, 14, 18, 20, 22],
        [1, 2, 4, 8, 16, 5, 3, 7, 9, 11, 13, 15, 17, 19, 21, 6, 10, 12, 14, 18, 20, 22],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
    ]
    
    for order in orderings:
        if not check_time():
            break
        current = []
        current_sums = {Fraction(0, 1)}
        for elem in order:
            r = Fraction(1, elem)
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + r
                if ns in current_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                current.append(elem)
                current_sums = current_sums | new_sums
        
        if len(current) > len(best_set):
            best_set = current
    
    # Local search: try to swap elements to improve
    improved = True
    while improved and check_time():
        improved = False
        for i in range(len(best_set)):
            if not check_time():
                break
            for j in all_elements:
                if j in best_set:
                    continue
                candidate = list(best_set)
                candidate[i] = j
                candidate.sort()
                if is_dissociated_fast(candidate):
                    best_set = candidate
                    improved = True
                    break
            if improved:
                break
    
    # Try adding elements
    for elem in all_elements:
        if not check_time():
            break
        if elem in best_set:
            continue
        candidate = best_set + [elem]
        candidate.sort()
        if is_dissociated_fast(candidate):
            best_set = candidate
    
    if not best_set:
        return [1, 2, 4, 8, 16]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END

