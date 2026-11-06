import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    # We know no two distinct elements in [1,100] sum to 200
    # So we only need to worry about subsets of size >= 3
    
    # Baseline: [66, 100] gives 35 elements, provably valid
    baseline = list(range(66, 101))
    best = baseline
    
    # Try to add elements below 66
    # For a candidate set S, check if any subset sums to 200
    # Use DP to check subset sum feasibility
    
    def has_subset_sum_200(elements):
        """Check if any subset of elements sums to exactly 200."""
        reachable = {0}
        for x in elements:
            new = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new.add(ns)
            reachable |= new
            if 200 in reachable:
                return True
        return False
    
    # Verify baseline
    assert not has_subset_sum_200(baseline)
    
    # Try greedy: add elements from 65 down to 1, one at a time
    # if adding it doesn't create a subset sum of 200
    current = list(baseline)
    for x in range(65, 0, -1):
        if time.time() > deadline:
            break
        candidate = current + [x]
        if not has_subset_sum_200(candidate):
            current.append(x)
    
    if len(current) > len(best):
        best = current
    
    # Try different greedy orders
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Start with [66,100] and try adding in random order
        current = list(baseline)
        candidates = list(range(1, 66))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            candidate = current + [x]
            if not has_subset_sum_200(candidate):
                current.append(x)
        
        if len(current) > len(best):
            best = current
    
    # Local search: try to swap elements
    if time.time() < deadline:
        current = list(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in current[:]:
                for y in range(1, 101):
                    if y in current or y >= 66:
                        continue
                    candidate = [e for e in current if e != x] + [y]
                    if not has_subset_sum_200(candidate) and len(candidate) > len(current):
                        current = candidate
                        improved = True
                        break
                if improved:
                    break
    
    if len(current) > len(best):
        best = current
    
    return best
