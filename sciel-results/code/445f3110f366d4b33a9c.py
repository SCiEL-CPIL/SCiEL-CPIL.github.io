import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    def check_divisibility(sums_set):
        """Check no two distinct sums divide each other."""
        sorted_sums = sorted(sums_set)
        n = len(sorted_sums)
        for i in range(n):
            si = sorted_sums[i]
            for j in range(i + 1, n):
                sj = sorted_sums[j]
                if sj % si == 0:
                    return False
        return True
    
    def try_add_incremental(current_sums, element):
        """Try adding element to current set. Returns new sums or None if invalid."""
        new_sums = set(current_sums)
        for s in current_sums:
            new_s = s + element
            if new_s in new_sums:
                return None
            new_sums.add(new_s)
        if element in new_sums:
            return None
        new_sums.add(element)
        if not check_divisibility(new_sums):
            return None
        return new_sums
    
    # Use binary-signature-inspired candidate ordering:
    # Prefer numbers with few set bits (close to powers of 2) and large values
    # This ensures subset sums are "spread out" and less likely to divide each other
    
    # Build candidate list sorted by priority
    candidates = list(range(256, 0, -1))
    candidates.sort(key=lambda x: (-x, bin(x).count('1')))
    
    best = [31, 32, 34]
    best_sums = set([31, 32, 34, 63, 65, 67, 98, 130, 162, 194])
    
    def backtrack(current, current_sums, candidates_list, depth):
        nonlocal best, best_sums
        if time.time() > deadline:
            return
        if len(current) > len(best):
            best = list(current)
            best_sums = set(current_sums)
        
        if not candidates_list:
            return
        
        for i, cand in enumerate(candidates_list):
            if time.time() > deadline:
                return
            new_sums = try_add_incremental(current_sums, cand)
            if new_sums is not None:
                new_current = current + [cand]
                remaining = candidates_list[i+1:]
                backtrack(new_current, new_sums, remaining, depth + 1)
    
    # Limit candidate pool to top ~100 to keep search tractable
    limited_candidates = candidates[:100]
    backtrack([], set(), limited_candidates, 0)
    
    # Ensure we return a valid set
    if len(best) < 3:
        best = [31, 32, 34]
    
    return [int(x) for x in best]
