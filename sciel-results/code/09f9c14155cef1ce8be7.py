import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Fallback known good solution
    best_set = set(range(66, 101))
    best_size = 35
    
    def is_valid(s):
        sums = {0}
        for x in s:
            new = {y + x for y in sums if y + x <= 200}
            if 200 in new:
                return False
            sums |= new
        return True
        
    # New Strategy: Start with {49..100} and remove elements to break sums of 200
    # Subsets of size >= 4 from {49..100} sum to > 200 (min sum 49+50+51+52 = 202)
    # So we only need to avoid pairs and triplets summing to 200.
    
    initial_set = list(range(49, 101))
    current_set = set(initial_set)
    
    # Find all bad pairs and triplets
    def get_bad_subsets(s):
        bad = []
        lst = sorted(s)
        n = len(lst)
        # Pairs
        for i in range(n):
            for j in range(i+1, n):
                if lst[i] + lst[j] == 200:
                    bad.append((lst[i], lst[j]))
        # Triplets
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if lst[i] + lst[j] + lst[k] == 200:
                        bad.append((lst[i], lst[j], lst[k]))
        return bad
        
    bad_subsets = get_bad_subsets(current_set)
    
    # Greedy removal
    while bad_subsets:
        if time.time() > deadline:
            break
            
        # Count occurrences
        counts = {}
        for sub in bad_subsets:
            for x in sub:
                counts[x] = counts.get(x, 0) + 1
                
        # Pick element with max count
        # Tie-breaker: random or smallest? Let's say smallest to keep larger numbers?
        # Or just max count.
        max_count = max(counts.values())
        candidates = [x for x, c in counts.items() if c == max_count]
        # Pick one
        x_to_remove = min(candidates) # Arbitrary choice
        
        current_set.remove(x_to_remove)
        bad_subsets = get_bad_subsets(current_set)
        
    # Now try to add elements from 1..48
    for x in range(1, 49):
        if time.time() > deadline:
            break
        if x not in current_set:
            # Check if adding x creates a bad subset
            # We need to check if there exists y, z in current_set such that:
            # 1. x + y = 200
            # 2. x + y + z = 200
            # Note: x is small (1..48), so y must be large.
            # If x + y = 200, y = 200 - x.
            # If x + y + z = 200, y+z = 200 - x.
            
            conflict = False
            y_needed = 200 - x
            if y_needed in current_set:
                conflict = True
            else:
                # Check triplets
                # We need y, z in current_set such that y + z = 200 - x
                # and y != z, y != x, z != x (x is small, y,z are large, so y!=x, z!=x automatically)
                target_sum = 200 - x
                # Iterate through current_set to find pair summing to target_sum
                # Since current_set is large, we can use a set for O(1) lookups
                # But we need distinct y, z.
                for y in current_set:
                    z = target_sum - y
                    if z in current_set and z != y:
                        conflict = True
                        break
                        
            if not conflict:
                current_set.add(x)
                
    # Verify and update best
    if is_valid(current_set):
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
    else:
        # If invalid, fallback to previous best
        pass
        
    # Try to improve further with local search?
    # The current set is likely quite good.
    # Let's return the best found.
    
    if len(best_set) > best_size: # Just in case
         best_size = len(best_set)
         
    return list(best_set)
