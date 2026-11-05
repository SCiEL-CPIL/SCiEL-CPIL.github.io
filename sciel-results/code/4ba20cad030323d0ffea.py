import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # We want a subset S of {1,...,22} such that all 2^|S| subset sums of {1/s : s in S} are distinct.
    # Use integer arithmetic: represent each 1/s as LCM/s where LCM = lcm(1..22)
    # Then subset sums are integers; check distinctness.
    
    # Precompute LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    LCM = 1
    for i in range(1, 23):
        LCM = lcm(LCM, i)
    
    # For each element s in 1..22, its weight is LCM // s
    weights = [LCM // s for s in range(1, 23)]
    
    # Backtracking: build a set, maintain all subset sums as a set of integers.
    # When adding element w to a set with existing subset sums S (set of ints),
    # new subset sums = S union {x + w for x in S}.
    # Check that no collision occurs (i.e., S and {x+w for x in S} are disjoint,
    # and {x+w for x in S} has no internal collisions — but since S has distinct elements,
    # x+w are distinct if x are distinct. So we only need to check S ∩ (S+w) = ∅).
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # fallback from seed
    
    # Try to find larger sets via backtracking
    # Order: try adding elements in various orders
    candidates = list(range(1, 23))
    
    # Use recursive backtracking with pruning
    # State: current set of elements (indices into candidates), current subset sums (frozenset of ints)
    
    def backtrack(start, current_elements, subset_sums):
        nonlocal best
        if time.time() > deadline:
            return
        
        if len(current_elements) > len(best):
            best = list(current_elements)
        
        # Try adding more elements
        for i in range(start, len(candidates)):
            if time.time() > deadline:
                return
            
            elem = candidates[i]
            w = weights[elem - 1]
            
            # Check: can we add w?
            # New sums = subset_sums ∪ {x + w for x in subset_sums}
            # Need subset_sums ∩ {x + w for x in subset_sums} = ∅
            # i.e., no x in subset_sums such that x + w in subset_sums
            # Equivalently, no x in subset_sums such that x - w in subset_sums (same thing)
            
            # Quick check: for each x in subset_sums, is x + w in subset_sums?
            valid = True
            for x in subset_sums:
                if (x + w) in subset_sums:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add it
            new_sums = set(subset_sums)
            for x in subset_sums:
                new_sums.add(x + w)
            
            current_elements.append(elem)
            backtrack(i + 1, current_elements, frozenset(new_sums))
            current_elements.pop()
    
    # Initial state: empty set, subset_sums = {0}
    # But we want nonempty, and we want to maximize size.
    # Start with empty and grow.
    
    # However, the search space is large. Let's be smarter:
    # Try different orderings and use the deadline.
    
    # Strategy: greedy + local search, or randomized restarts
    # Let's do multiple randomized attempts
    
    import random
    
    best_size = len(best)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        
        # Greedy: add elements if possible
        current_elements = []
        subset_sums = frozenset([0])
        
        for elem in candidates:
            if time.time() > deadline:
                break
            w = weights[elem - 1]
            valid = True
            for x in subset_sums:
                if (x + w) in subset_sums:
                    valid = False
                    break
            if valid:
                new_sums = set(subset_sums)
                for x in subset_sums:
                    new_sums.add(x + w)
                current_elements.append(elem)
                subset_sums = frozenset(new_sums)
        
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best = list(current_elements)
    
    # Try to extend the best set found
    if best_size >= 9:
        if time.time() < deadline - 5:
            # Try to add more elements to best
            current_elements = list(best)
            subset_sums = set([0])
            for elem in current_elements:
                w = weights[elem - 1]
                for x in list(subset_sums):
                    subset_sums.add(x + w)
            subset_sums = frozenset(subset_sums)
            
            for elem in range(1, 23):
                if elem in current_elements:
                    continue
                w = weights[elem - 1]
                valid = True
                for x in subset_sums:
                    if (x + w) in subset_sums:
                        valid = False
                        break
                if valid:
                    new_sums = set(subset_sums)
                    for x in subset_sums:
                        new_sums.add(x + w)
                    current_elements.append(elem)
                    subset_sums = frozenset(new_sums)
                    best_size = len(current_elements)
                    best = list(current_elements)
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return sorted(best)
# EVOLVE-BLOCK-END
