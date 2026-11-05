import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit to be safe against 90s kill
    
    n = 12
    
    def is_valid(a):
        # a is a tuple of n distinct positive integers
        # Check if all subset sums are distinct
        # Using DP to generate sums
        sums = [0] * (1 << n)
        for i in range(n):
            ai = a[i]
            bit = 1 << i
            for mask in range(bit):
                sums[bit | mask] = sums[mask] + ai
        
        # Check uniqueness
        # For n=12, 4096 sums.
        # Using a set is O(2^n)
        seen = set(sums)
        return len(seen) == (1 << n)

    def greedy_construct():
        # Greedy: pick smallest possible next element
        a = []
        for i in range(n):
            start_val = 1
            if a:
                start_val = a[-1] + 1
            
            found = False
            # Upper bound for search in this step
            # Heuristic: 2^i * 2
            upper = 1 << (i + 2)
            for v in range(start_val, upper):
                new_a = a + [v]
                if is_valid(tuple(new_a)):
                    a = new_a
                    found = True
                    break
            if not found:
                # If not found in range, try wider range
                for v in range(start_val, 5000):
                    new_a = a + [v]
                    if is_valid(tuple(new_a)):
                        a = new_a
                        found = True
                        break
            if not found:
                return None
        return a

    # Initial greedy
    best_set = greedy_construct()
    if best_set is None:
        # Fallback to powers of 2
        best_set = [1 << i for i in range(n)]
    
    current_max = max(best_set)
    best_score = current_max
    a = best_set[:]
    
    # Local Search
    # We want to reduce the max element.
    iterations = 0
    no_improve_count = 0
    last_improve_time = start_time
    
    while time.time() < deadline:
        iterations += 1
        # Pick a random index
        i = random.randint(0, n - 1)
        old_val = a[i]
        
        # Strategy:
        # 1. If a[i] is the current max, try to reduce it.
        # 2. Otherwise, random perturbation.
        
        if a[i] == max(a):
            if a[i] > 1:
                # Try to reduce
                # Sample from 1 to a[i]-1
                # Bias towards lower values?
                new_val = random.randint(1, a[i] - 1)
            else:
                continue
        else:
            # Random perturbation
            # Range: 1 to max(a) + 10
            # Avoid values already in a
            # Simple rejection sampling
            max_val = max(a)
            new_val = random.randint(1, max_val + 10)
        
        if new_val == old_val:
            continue
            
        # Check distinctness
        # If new_val is in a (excluding i), it's invalid
        # We can check this quickly
        if new_val in a:
            # If new_val is at index i, it's the same value, handled above
            # If new_val is at another index, invalid
            # Note: 'new_val in a' checks the whole list
            # If new_val == a[j] for j != i, it's a duplicate
            # We need to be careful if new_val == old_val (handled)
            # If new_val is in a, and a[i] != new_val, then there is a duplicate
            # Unless a[i] == new_val (which means we are keeping it, but we changed it)
            # So if new_val in a, it means either:
            # 1. new_val == a[i] (old_val) -> handled
            # 2. new_val == a[j] for j != i -> duplicate, invalid
            # So we can just check if new_val in a and new_val != old_val
            # But we already checked new_val != old_val
            # So if new_val in a, it must be a duplicate with another element.
            continue
            
        a[i] = new_val
        
        if is_valid(tuple(a)):
            new_max = max(a)
            if new_max < best_score:
                best_score = new_max
                best_set = a[:]
                no_improve_count = 0
                last_improve_time = time.time()
            else:
                no_improve_count += 1
            
            # Acceptance criteria
            # Accept if it doesn't make things too much worse
            # Or use simulated annealing?
            # For now, accept if new_max <= best_score + 50
            if new_max > best_score + 50:
                a[i] = old_val # Revert
            else:
                a[i] = new_val # Keep
        else:
            a[i] = old_val # Revert
            
        # Restart logic: if stuck for too long, reset to best_set and randomize
        if no_improve_count > 5000:
            a = best_set[:]
            # Randomly perturb a few elements
            for _ in range(3):
                idx = random.randint(0, n-1)
                a[idx] = random.randint(1, max(a) + 20)
            no_improve_count = 0

    return best_set
