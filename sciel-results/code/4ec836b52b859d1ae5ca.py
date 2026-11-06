import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Constraint satisfaction approach:
    # We build a Sidon set S containing 0, with |S| <= 32, all differences 1..30 present.
    # Minimize max(S).
    #
    # Key insight: For a Sidon set, all pairwise differences are distinct.
    # If we have n elements, we have C(n,2) differences. We need at least 30, so n >= 9.
    # 
    # Strategy: Branch-and-bound DFS where we assign elements in increasing order.
    # At each step, we try candidate values for the next element, checking:
    # 1. Sidon constraint (no duplicate differences)
    # 2. Progress toward covering all required differences
    # 3. Pruning based on remaining capacity
    
    target_diffs = frozenset(range(1, 31))
    max_size = 32
    min_size = 9  # C(9,2) = 36 >= 30
    
    best_result = None
    best_max_elem = float('inf')
    
    def _is_valid_sidon(elements, seen_diffs, x):
        """Check if adding x to elements maintains Sidon property."""
        for e in elements:
            d = x - e
            if d > 0 and d in seen_diffs:
                return False, None
        # All differences x - e for e in elements are positive and distinct by construction
        new_diffs = frozenset(x - e for e in elements if x - e > 0)
        return True, new_diffs
    
    def _can_cover_remaining(remaining, num_elements, max_size, current_max):
        """Check if remaining differences can be covered with up to (max_size - num_elements) more elements."""
        if not remaining:
            return True
        slots_left = max_size - num_elements
        if slots_left <= 0:
            return False
        # Each new element can cover at most (num_elements) new differences (one with each existing element)
        # But some might be > 30 or already seen, so this is an upper bound
        max_possible_new = 0
        for _ in range(slots_left):
            max_possible_new += num_elements
            num_elements += 1
        if len(remaining) > max_possible_new:
            return False
        return True
    
    def dfs(elements, seen_diffs, covered_diffs):
        nonlocal best_result, best_max_elem, deadline
        
        if time.time() > deadline:
            return
        
        # Check if we've covered all required differences
        if covered_diffs == target_diffs:
            max_elem = elements[-1]
            if max_elem < best_max_elem:
                best_max_elem = max_elem
                best_result = list(elements)
            return
        
        # If we can't add more elements
        if len(elements) >= max_size:
            return
        
        remaining = target_diffs - covered_diffs
        if not remaining:
            return
        
        # Pruning: can we still cover remaining differences?
        if not _can_cover_remaining(remaining, len(elements), max_size, elements[-1]):
            return
        
        current_max = elements[-1]
        upper_bound = best_max_elem - 1 if best_result else 10**6
        
        # Generate candidate next elements
        # A candidate x > current_max must:
        # 1. Not violate Sidon: x - e not in seen_diffs for all e in elements
        # 2. Ideally cover at least one remaining difference
        # 
        # Candidates that cover a remaining difference d with some existing element e:
        # x = e + d for e in elements, d in remaining
        # Also consider x that might enable future coverage
        
        candidates = set()
        for e in elements:
            for d in remaining:
                x = e + d
                if current_max < x <= upper_bound:
                    candidates.add(x)
        
        # Also add some "bridge" candidates that are just above current_max
        # to allow flexibility in the search
        for delta in range(1, min(6, upper_bound - current_max + 1)):
            candidates.add(current_max + delta)
        
        if not candidates:
            return
        
        # Sort candidates: prefer smaller values (tighter packing) and those covering more diffs
        def score_candidate(x):
            # Count how many remaining diffs this x would cover
            count = 0
            for e in elements:
                d = x - e
                if 0 < d <= 30 and d in remaining:
                    count += 1
            # Prefer smaller x, then more coverage
            return (x, -count)
        
        sorted_candidates = sorted(candidates, key=score_candidate)
        
        for x in sorted_candidates:
            if time.time() > deadline:
                return
            
            # Check Sidon constraint
            valid, new_diffs = _is_valid_sidon(elements, seen_diffs, x)
            if not valid:
                continue
            
            new_covered = covered_diffs | (new_diffs & target_diffs)
            
            # Additional pruning: if adding x doesn't cover any new required diff,
            # it's only useful if it enables future coverage. Be more selective.
            if new_covered == covered_diffs and len(elements) >= 5:
                # Only allow "waste" elements early in the search
                continue
            
            dfs(elements + [x], seen_diffs | new_diffs, new_covered)
    
    # Start with [0]
    dfs([0], frozenset(), frozenset())
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: greedy construction
        a = [0]
        seen = set()
        for d in range(1, 31):
            if d in seen:
                continue
            found = False
            for e in a:
                x = e + d
                if x not in a:
                    # Check Sidon
                    valid = True
                    for e2 in a:
                        diff = x - e2
                        if diff > 0 and diff in seen:
                            valid = False
                            break
                    if valid:
                        new_diffs = set(x - e2 for e2 in a if x - e2 > 0)
                        if not (new_diffs & seen):
                            a.append(x)
                            a.sort()
                            seen.update(new_diffs)
                            found = True
                            break
            if not found:
                # Try larger x
                x = max(a) + d
                while True:
                    valid = True
                    for e in a:
                        diff = x - e
                        if diff > 0 and diff in seen:
                            valid = False
                            break
                    if valid:
                        new_diffs = set(x - e for e in a if x - e > 0)
                        if not (new_diffs & seen):
                            a.append(x)
                            a.sort()
                            seen.update(new_diffs)
                            break
                    x += 1
        return a
