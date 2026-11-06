"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def _check_sidon_and_cover(a, target_diffs):
    """Check if a is a Sidon set covering target_diffs."""
    n = len(a)
    if n > 32:
        return False
    if max(a) > 10**6:
        return False
    if 0 not in a:
        return False
    
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d <= 0:
                return False
            if d in diffs:
                return False
            diffs.add(d)
    return all(d in diffs for d in target_diffs)

def _search_fast(M, target_diffs, deadline):
    """
    Fast DFS for Sidon set in [0, M] covering target_diffs.
    """
    initial_set = [0]
    initial_diffs = set()
    remaining = set(target_diffs)
    
    def dfs(current_set, used_diffs, remaining):
        if not remaining:
            return current_set[:]
        
        if len(current_set) >= 32:
            return None
            
        if time.time() > deadline:
            return None
        
        last = current_set[-1]
        
        # Generate candidates: x > last, x <= M
        # We want to cover remaining diffs.
        # Candidate x creates diffs x - a for a in current_set.
        # We need these diffs to be new (not in used_diffs) and ideally in remaining.
        
        candidates = []
        
        # Optimization: Only consider x that can cover at least one remaining diff.
        # x - a = d => x = a + d.
        # So x must be of the form a + d where a in current_set, d in remaining.
        # Also x > last.
        
        potential_xs = set()
        for a in current_set:
            for d in remaining:
                x = a + d
                if x > last and x <= M:
                    potential_xs.add(x)
        
        if not potential_xs:
            return None

        for x in potential_xs:
            # Check if x creates any conflict with used_diffs
            conflict = False
            new_covered = 0
            for a in current_set:
                d = x - a
                if d in used_diffs:
                    conflict = True
                    break
                if d in remaining:
                    new_covered += 1
            
            if conflict:
                continue
            
            # Heuristic: Prioritize x that covers more remaining diffs.
            # Tie-breaker: Smaller x (to keep diameter small)
            candidates.append((x, -new_covered))
        
        if not candidates:
            return None
            
        # Sort by coverage (desc), then value (asc)
        candidates.sort()
        
        for x, _ in candidates:
            new_diffs = set(x - a for a in current_set)
            new_used = used_diffs | new_diffs
            new_remaining = remaining - new_used
            
            res = dfs(current_set + [x], new_used, new_remaining)
            if res is not None:
                return res
                
        return None

    return dfs(initial_set, initial_diffs, remaining)

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop searching at 110s.
    deadline = start_time + 110.0
    
    target_diffs = set(range(1, 31))
    
    # Minimum size is 9 (since 8 elements have 28 diffs < 30).
    # Let's try to find a solution with small M.
    # Lower bound for M:
    # For k=9, min diameter is approx k^2 - 2k + 1? No, that's for perfect.
    # Let's just iterate M from a reasonable lower bound.
    # 9 elements, 36 diffs. Smallest 30 diffs are 1..30.
    # If we use 0, 1, ..., 8, diffs are 1..8. Not enough.
    # A known small Sidon set is {0, 1, 4, 10, 12, 17, 21, 23, 28}?
    # Let's just search.
    
    best_result = None
    best_max = 10**6
    
    # Start M from 30 (impossible for 9 elements to have max 30 and cover 1..30? 
    # If max is 30, elements are in 0..30. 9 elements.
    # Max diff is 30. We need 1..30. So 30 must be a diff.
    # If 0 is in set, then 30 must be in set.
    # Then we need 1..29 from the other 7 elements.
    # 7 elements have 21 diffs. We need 29 diffs. Impossible.
    # So M must be larger.
    
    # Let's estimate lower bound more carefully.
    # k=9. Diffs are distinct.
    # Sum of smallest 30 diffs is 1+..+30 = 465.
    # Sum of all diffs in a set {a_0=0 < a_1 < ... < a_8=M}
    # is sum_{i<j} (a_j - a_i).
    # This sum must be >= 465.
    # Also, the largest diff is M.
    # The next largest is M - a_1, etc.
    
    # Let's just try M from 40 upwards.
    # To avoid timeout, we limit the search time per M or globally.
    
    for M in range(40, 150):
        if time.time() > deadline:
            break
            
        # If we have a better result, we can skip M if M >= best_max
        if best_result and M >= best_max:
            break
            
        result = _search_fast(M, target_diffs, deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                best_result = result
                best_max = max(result)
                # Since we are iterating M upwards, the first valid M is the minimal M?
                # Not necessarily, because for a fixed M, there might be multiple solutions,
                # but the max element is bounded by M.
                # If we find a solution with max <= M, and we are increasing M,
                # the first M that yields a solution gives the minimal possible max?
                # Yes, because if a solution exists with max = K, it will be found when M >= K.
                # So the first M that works is the minimal K.
                return result
                
    # If no solution found within deadline, return a known valid set (greedy)
    # This ensures we don't fail with an exception or None.
    # Greedy construction for fallback
    a = [0]
    used = set()
    required = set(range(1, 31))
    # Greedy: pick smallest x that doesn't create conflict and covers a required diff
    # This might result in a larger max, but it's valid.
    for x in range(1, 1000):
        if not required:
            break
        new_diffs = set()
        ok = True
        for v in a:
            d = abs(x - v)
            if d in used:
                ok = False
                break
            new_diffs.add(d)
        if ok and (new_diffs & required):
            a.append(x)
            used |= new_diffs
            required -= new_diffs
            
    if required:
        # Fallback: add large numbers to cover remaining
        # This is a last resort
        last = max(a)
        while required:
            last += 100
            a.append(last)
            for v in a[:-1]:
                d = abs(last - v)
                used.add(d)
                if d in required:
                    required.remove(d)
                    
    return a
# EVOLVE-BLOCK-END
