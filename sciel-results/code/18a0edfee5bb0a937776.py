# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Constructs a set of 12 distinct positive integers such that all 4096 subset
    sums are distinct, and the largest integer is minimized.
    """
    n = 12
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    def has_distinct_subset_sums(a):
        sums = {0}
        for x in a:
            new_sums = {s + x for s in sums}
            if len(new_sums) != len(sums):
                return False
            sums = new_sums | sums
        return len(sums) == (1 << n)
    
    # Greedy construction: add the smallest possible element at each step
    # that maintains distinct subset sums
    current = [1]
    
    for i in range(1, n):
        # Try values starting from 2, then 3, etc.
        # But we need to ensure distinctness from existing elements
        existing = set(current)
        
        # The sum of all existing elements is S
        # New element x must be > 0 and not in existing
        # For distinct subset sums, x must not create collisions
        # A sufficient condition: x > sum(current), but we can do better
        
        # Try from 2 upward, skipping existing values
        S = sum(current)
        for x in range(2, S + 100):  # Upper bound heuristic
            if time.time() > deadline:
                break
            if x in existing:
                continue
            
            test = current + [x]
            if has_distinct_subset_sums(test):
                current.append(x)
                break
        else:
            # Fallback: use S + 1
            current.append(S + 1)
    
    # Now try to improve by local search
    best_max = max(current)
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Try to reduce the largest element
        max_idx = max(range(n), key=lambda i: current[i])
        old_val = current[max_idx]
        other_vals = set(current[i] for i in range(n) if i != max_idx)
        
        found = False
        for _ in range(20):
            if time.time() > deadline:
                break
            new_val = random.randint(1, old_val - 1)
            if new_val in other_vals:
                continue
            
            test = current[:]
            test[max_idx] = new_val
            test.sort()
            
            if has_distinct_subset_sums(test):
                current = test
                best_max = max(current)
                found = True
                break
        
        if not found and it % 100 == 0:
            # Try reducing other elements
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            other_vals = set(current[i] for i in range(n) if i != idx)
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_val = random.randint(1, old_val - 1)
                if new_val in other_vals:
                    continue
                
                test = current[:]
                test[idx] = new_val
                test.sort()
                
                if has_distinct_subset_sums(test):
                    current = test
                    best_max = max(current)
                    break
    
    return current
# EVOLVE-BLOCK-END
