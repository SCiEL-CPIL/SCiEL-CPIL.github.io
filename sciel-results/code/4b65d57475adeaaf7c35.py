def construct():
    # Start with the known valid set [66, 100]
    base = list(range(66, 101))
    
    # Verify no subset sums to 200
    # Since all elements >= 66, subsets of size >= 4 sum to >= 264 > 200
    # So only check subsets of size 1, 2, 3
    def is_valid(s):
        # Check size 1
        if 200 in s:
            return False
        # Check size 2
        from itertools import combinations
        for a, b in combinations(s, 2):
            if a + b == 200:
                return False
        # Check size 3
        for a, b, c in combinations(s, 3):
            if a + b + c == 200:
                return False
        # Subsets of size >= 4 all sum to > 200 since min element >= 66
        # But we should verify this assumption
        if min(s) >= 66:
            return True
        return False
    
    # Try to improve by adding small elements
    # For each candidate small element, check if we can add it
    # while maintaining validity (may need to remove some large elements)
    
    best = base[:]
    
    # Try adding elements from 1 to 65, one at a time, greedily
    # For each, find minimal removals needed
    for s in range(1, 66):
        candidate = best + [s]
        if not is_valid(candidate):
            # Find which elements conflict
            # We need to remove elements so that no subset sums to 200
            # This is complex; skip for now
            pass
        else:
            best = candidate
    
    return best
