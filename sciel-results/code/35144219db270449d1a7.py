import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Candidate sets
    # 1. Consecutive integers
    # 2. Even numbers
    # 3. Other structured sets
    
    candidates = [
        list(range(12)),
        list(range(0, 24, 2)),
        [0, 1, 2, 4, 8, 16, 32, 64, 2, 3, 5, 9], # Mixed
        [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 120], # Fibonacci
        [0, 1, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75], # Random-ish from seed
    ]
    
    best_set = None
    best_size = 13 # Higher than possible max 12
    
    def check_sidon_size(s, k):
        """Check if s contains a Sidon subset of size k."""
        n = len(s)
        # Sort to help with pruning? Not strictly necessary for small k.
        s = sorted(s)
        
        # If k > n, return False
        if k > n:
            return False
            
        # Use backtracking to find if ANY subset of size k is Sidon
        # This is faster than checking all if we prune early.
        
        def backtrack(start, current_subset, used_sums):
            if len(current_subset) == k:
                return True
            # Pruning: if remaining elements not enough
            if n - start < k - len(current_subset):
                return False
                
            for i in range(start, n):
                x = s[i]
                # Check collisions with x + x and x + y for y in current_subset
                conflict = False
                new_sums = set()
                
                # Diagonal
                if x + x in used_sums:
                    continue
                new_sums.add(x + x)
                
                # Off-diagonal with existing
                for y in current_subset:
                    s_val = x + y
                    if s_val in used_sums or s_val in new_sums:
                        conflict = True
                        break
                    new_sums.add(s_val)
                
                if conflict:
                    continue
                
                # Add x and recurse
                if backtrack(i + 1, current_subset + [x], used_sums | new_sums):
                    return True
                    
            return False
        
        return backtrack(0, [], set())

    for c in candidates:
        if time.time() > deadline:
            break
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
            
        # Check if it has a Sidon subset of size 5
        if check_sidon_size(c, 5):
            sz = 5 # At least 5, likely higher, but we just need to know it's not 4 or 3
            # We can refine if needed, but 5 is "bad"
        else:
            # Check if it has a Sidon subset of size 4
            if check_sidon_size(c, 4):
                sz = 4
            else:
                sz = 3
                # If we find 3, we can stop immediately as it's the minimum possible
                # (Assuming 2 is impossible for size 12? 
                # A set of 12 must have a Sidon subset of size at least 3?
                # Yes, by Erdos-Turan, but practically, 3 is very likely the floor.)
        
        if sz < best_size:
            best_size = sz
            best_set = c
            if best_size == 3:
                break

    # If no good candidate found, fall back to random search with fast check
    if best_size > 4: # If we didn't find a 4 or 3
        import random
        random.seed(42)
        for _ in range(1000):
            if time.time() > deadline:
                break
            s = random.sample(range(129), 12)
            if check_sidon_size(s, 5):
                continue
            if check_sidon_size(s, 4):
                sz = 4
            else:
                sz = 3
            
            if sz < best_size:
                best_size = sz
                best_set = s
                if best_size == 3:
                    break

    return best_set
