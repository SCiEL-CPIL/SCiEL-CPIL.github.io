import random

def construct():
    n = 512
    # Efficient check if a set is 4-AP free
    # We will maintain the set as a Python set for O(1) lookups.
    
    def is_valid_4ap_free(S):
        # This is for initial check or debugging, but we will do incremental checks.
        pass

    def creates_4ap_with_x(S, x):
        # Check if adding x to S creates a 4-AP involving x.
        # x is in [1, n]
        # Case 1: x is the first term (x, x+d, x+2d, x+3d)
        # d can be such that x+3d <= n
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return True
        # Case 2: x is the second term (x-d, x, x+d, x+2d)
        # x-d >= 1 => d <= x-1
        # x+2d <= n => d <= (n-x)//2
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return True
        # Case 3: x is the third term (x-2d, x-d, x, x+d)
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= n => d <= n-x
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return True
        # Case 4: x is the fourth term (x-3d, x-2d, x-d, x)
        # x-3d >= 1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return True
        return False

    # Greedy initialization
    current_set = set()
    for i in range(1, n + 1):
        if not creates_4ap_with_x(current_set, i):
            current_set.add(i)
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Local Search
    # Time limit 60s. We can run many iterations.
    # Let's estimate speed. n=512. Check cost is ~ O(n^2) in worst case but practically much less.
    # Let's try 100,000 iterations or until time limit.
    
    import time
    start_time = time.time()
    time_limit = 55.0 # Leave buffer for overhead
    
    random.seed(42) # Deterministic for reproducibility in eval if needed, but randomness helps escape local minima
    
    # We will use a "kick" strategy:
    # 1. Try to add a random element.
    # 2. If no element can be added (stuck), remove k random elements (k=1,2,3) and then greedily fill back.
    
    current_set = set(best_set)
    
    for _ in range(500000): # Max iterations
        if time.time() - start_time > time_limit:
            break
            
        # Try to add a random element
        added = False
        for _ in range(10): # Try a few random adds
            if len(current_set) == n:
                break
            candidates = [i for i in range(1, n + 1) if i not in current_set]
            if not candidates:
                break
            x = random.choice(candidates)
            if not creates_4ap_with_x(current_set, x):
                current_set.add(x)
                added = True
                break
        
        if added:
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
        else:
            # Stuck. Perform a kick.
            # Remove 1-3 random elements
            k = random.randint(1, 3)
            if len(current_set) > k:
                to_remove = random.sample(list(current_set), k)
                for x in to_remove:
                    current_set.remove(x)
                
                # Now greedily try to fill back with any available element
                # We can just iterate through all elements and add if possible
                # Or randomize order
                all_elements = list(range(1, n + 1))
                random.shuffle(all_elements)
                for x in all_elements:
                    if x not in current_set:
                        if not creates_4ap_with_x(current_set, x):
                            current_set.add(x)
                            
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                    
    return sorted(best_set)
