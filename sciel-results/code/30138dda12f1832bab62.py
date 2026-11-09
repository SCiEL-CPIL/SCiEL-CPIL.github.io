def construct():
    """Construct a Sidon set (distinct pairwise sums) in [1, 284] containing {1, 2, 5, 11}.
    
    Uses Iterated Local Search with time management to maximize the set size.
    """
    import random
    import time
    
    # Required elements
    required = [1, 2, 5, 11]
    N = 284
    req_set = set(required)
    
    # Precompute all possible pairs sums? No, dynamic is better.
    
    def compute_sums(arr):
        """Compute set of all pairwise sums a+b with a<=b."""
        sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                sums.add(arr[i] + arr[j])
        return sums

    def can_add(arr, sums, x):
        """Check if x can be added to arr without conflict."""
        # x is not in arr
        # New sums: x+y for y in arr, and 2x
        # Must not intersect with existing sums
        # Also x+y must be distinct for different y (guaranteed if y distinct)
        # 2x must be distinct from x+y (guaranteed if x not in arr)
        
        for y in arr:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def is_valid(arr):
        """Check if the entire set is a valid Sidon set."""
        sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                s = arr[i] + arr[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_construct(order='asc'):
        """Construct a greedy Sidon set."""
        a = required[:]
        sums = compute_sums(a)
        
        # Candidates 12 to 284
        candidates = list(range(12, N + 1))
        
        if order == 'desc':
            candidates.reverse()
        elif order == 'rand':
            random.shuffle(candidates)
            
        for x in candidates:
            if can_add(a, sums, x):
                a.append(x)
                sums = compute_sums(a)
                a.sort()
                
        return a

    def local_search(arr, max_iters=5000):
        """
        Local search: try to add elements, or swap elements to allow additions.
        Returns the best set found.
        """
        current = sorted(arr)
        best = current[:]
        best_size = len(current)
        sums = compute_sums(current)
        
        for _ in range(max_iters):
            improved = False
            
            # 1. Try to add any valid element
            # Optimize: Check candidates not in current
            # We can maintain a list of "blocked" candidates, but recomputing is safer
            # For speed, we can just iterate through 1..284
            
            added = False
            for x in range(1, N + 1):
                if x in current:
                    continue
                if can_add(current, sums, x):
                    current.append(x)
                    current.sort()
                    sums = compute_sums(current)
                    improved = True
                    added = True
                    if len(current) > best_size:
                        best = current[:]
                        best_size = len(current)
                    break # Restart loop with new state
            
            if added:
                continue
                
            # 2. If no addition, try swaps
            # Pick a random non-required element to remove
            non_required = [x for x in current if x not in req_set]
            
            if non_required:
                # Try to kick one element and replace with another
                # To be efficient, we can try a few random swaps
                kick_idx = random.randint(0, len(non_required) - 1)
                old_val = non_required[kick_idx]
                
                # Remove it temporarily
                current.remove(old_val)
                temp_sums = compute_sums(current)
                
                # Try to add a new value
                # Heuristic: Try values that were previously blocked by old_val?
                # Or just random search for a valid addition
                found_new = False
                # Try a subset of candidates to save time
                # Candidates are 1..284 excluding current
                # Shuffle candidates for randomness
                candidates = [x for x in range(1, N + 1) if x not in current and x not in req_set]
                random.shuffle(candidates)
                
                # Limit candidates to try to keep iteration fast
                for x in candidates[:200]: 
                    if can_add(current, temp_sums, x):
                        current.append(x)
                        current.sort()
                        sums = compute_sums(current)
                        found_new = True
                        if len(current) > best_size:
                            best = current[:]
                            best_size = len(current)
                        break
                        
                if not found_new:
                    # Restore old_val
                    current.append(old_val)
                    current.sort()
                    sums = compute_sums(current)
            else:
                # Only required elements left? Unlikely, but break if so
                break

        return best

    # Main Loop: Iterated Local Search
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    best_result = []
    
    # Initial runs with different strategies
    strategies = ['asc', 'desc', 'rand', 'rand', 'rand']
    
    for i, strat in enumerate(strategies):
        if time.time() - start_time > time_limit:
            break
            
        # Seed random for reproducibility
        random.seed(100 + i)
        
        init = greedy_construct(strat)
        
        # Run local search
        # Allocate time based on remaining time
        # Simple: run for a fixed number of iterations, or until time slice
        # Let's do time-sliced local search
        slice_time = 20.0
        local_start = time.time()
        
        # We need a time-aware local search
        # Modify local_search to accept a deadline
        # Or just run standard local_search and check time after
        
        # To be safe, let's run local_search with a large iteration count
        # but break early if time is up.
        # Since local_search doesn't check time, we'll wrap it.
        
        # Actually, let's just run a few rounds of local search
        current_best = init
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            improved = local_search(current_best, max_iters=2000)
            if len(improved) > len(current_best):
                current_best = improved
            else:
                # Perturb
                # Remove a random non-required element
                non_req = [x for x in current_best if x not in req_set]
                if non_req:
                    to_remove = random.choice(non_req)
                    current_best.remove(to_remove)
                    current_best.sort()
                # Add a random valid element if possible
                sums = compute_sums(current_best)
                for x in range(1, N + 1):
                    if x not in current_best and can_add(current_best, sums, x):
                        current_best.append(x)
                        current_best.sort()
                        break

        if len(current_best) > len(best_result):
            best_result = current_best[:]

    # Final verification
    if not all(x in best_result for x in req_set):
        return required[:]
        
    if not is_valid(best_result):
        # Fallback to a safe small set
        return required[:]

    return sorted(best_result)
