<<<<<<< SEARCH
def construct():
    """
    Find 12 distinct positive integers with all subset sums distinct, minimizing the largest element.
    Uses A* best-first search over sorted tuples.
    """
    start_time = time.time()
    deadline = 85  # seconds, with safety margin
    
    n = 12
    
    def subset_sums(a):
        """Compute all subset sums of list a."""
        sums = {0}
        for x in a:
            new_sums = {s + x for s in sums}
            sums = sums | new_sums
        return sums
    
    def is_valid(a):
        """Check if all subset sums are distinct."""
        return len(subset_sums(a)) == (1 << len(a))
    
    def greedy_construct():
        """Greedy: add smallest possible element at each step."""
        result = []
        current_sums = {0}
        
        for i in range(n):
            # Try candidates starting from 1
            candidate = 1
            while True:
                # Check if adding candidate keeps all subset sums distinct
                new_sums = current_sums | {s + candidate for s in current_sums}
                if len(new_sums) == (1 << (i + 1)):
                    result.append(candidate)
                    current_sums = new_sums
                    break
                candidate += 1
                # Safety: if we've gone too high, break
                if candidate > 1048576:
                    return None
                if time.time() - start_time > deadline:
                    return None
        return result
    
    # First try greedy to get an upper bound
    greedy_result = greedy_construct()
    if greedy_result is not None:
        best_max = max(greedy_result)
    else:
        # Fallback: powers of 2
        return [1 << i for i in range(n)]
    
    # Now use A* to try to find a better solution
    # State: (heuristic, max_element, tuple_of_elements)
    # Heuristic: lower bound on final max element
    
    def heuristic(current_tuple):
        """
        Lower bound on the maximum element given current partial solution.
        The current elements have sum S. The remaining (n - len) elements must be distinct,
        positive, and > all current elements. The minimal possible max for the remaining
        elements, given they must keep subset sums distinct, is at least:
        - The next element must be > max(current) and such that no collision occurs.
        - A simple lower bound: if we have k elements with max M, the next element must be
          at least M+1, and to keep things minimal, we can use a superincreasing-like bound.
        """
        if not current_tuple:
            return 1
        k = len(current_tuple)
        current_max = current_tuple[-1]
        current_sum = sum(current_tuple)
        
        # The next element must be > current_max and must not create subset sum collisions.
        # A lower bound: the next element must be at least current_max + 1.
        # But more tightly, for the remaining (n-k) elements, the minimum possible max
        # is bounded by the fact that we need 2^(n-k) distinct subset sums from the remaining
        # elements, all shifted by sums of current elements.
        
        # Simple admissible heuristic: the final max must be at least current_max + (n - k)
        # because we need (n-k) more distinct positive integers all > current_max.
        # But this is weak. A better bound: if the current elements have sum S,
        # the remaining elements must generate 2^(n-k) distinct sums, and the largest
        # of those sums (when added to current sums) must be distinct from existing ones.
        
        # For now, use: final_max >= current_max + (n - k)
        # This is admissible (never overestimates) and guides the search.
        return current_max + (n - k)
    
    # A* search
    # Priority queue: (heuristic_value, max_element, step, tuple_of_elements)
    # We want to minimize the final max element.
    
    initial_state = ()
    initial_heuristic = heuristic(initial_state)
    
    # Use a priority queue. We prioritize by the heuristic (lower bound on final max).
    # To break ties, we prefer states with smaller current max.
    
    open_set = []
    heapq.heappush(open_set, (initial_heuristic, 0, 0, initial_state))
    
    best_solution = greedy_result
    best_max_value = best_max
    
    visited = set()
    
    while open_set:
        if time.time() - start_time > deadline:
            break
        
        h, current_max, step, state = heapq.heappop(open_set)
        
        # If we've already found a solution with max <= h, we can prune
        if h >= best_max_value:
            continue
        
        # Check if this state has been visited
        state_key = state
        if state_key in visited:
            continue
        visited.add(state_key)
        
        if len(state) == n:
            # Complete solution
            if is_valid(state):
                final_max = max(state)
                if final_max < best_max_value:
                    best_max_value = final_max
                    best_solution = list(state)
            continue
        
        # Generate next states: try adding the next element
        # The next element must be > current_max (to keep them distinct and sorted)
        # and must keep all subset sums distinct.
        
        # Compute current subset sums
        if step == 0:
            current_sums = {0}
        else:
            current_sums = subset_sums(state)
        
        # Try candidates for the next element
        # Start from current_max + 1
        start_candidate = current_max + 1 if current_max > 0 else 1
        
        # Limit the number of candidates we try per state to avoid explosion
        max_candidates = 50
        candidate = start_candidate
        candidates_tried = 0
        
        while candidate <= best_max_value and candidates_tried < max_candidates:
            # Check if adding candidate keeps all subset sums distinct
            new_sums = current_sums | {s + candidate for s in current_sums}
            if len(new_sums) == (1 << (step + 1)):
                new_state = state + (candidate,)
                new_max = candidate
                new_h = heuristic(new_state)
                
                # Only push if the heuristic is better than current best
                if new_h < best_max_value:
                    heapq.heappush(open_set, (new_h, new_max, step + 1, new_state))
            
            candidate += 1
            candidates_tried += 1
        
        # Also try a few candidates above best_max_value in case we can still improve
        # Actually, if candidate > best_max_value, the final max will be >= candidate > best_max_value,
        # so we can't improve. So we stop here.
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
=======
def construct():
    """
    Find 12 distinct positive integers with all subset sums distinct, minimizing the largest element.
    Uses A* best-first search with efficient subset sum tracking.
    """
    start_time = time.time()
    deadline = 85  # seconds, with safety margin
    
    n = 12
    
    def greedy_construct():
        """Greedy: add smallest possible element at each step."""
        result = []
        current_sums = {0}
        
        for i in range(n):
            candidate = 1
            while True:
                new_sums = current_sums | {s + candidate for s in current_sums}
                if len(new_sums) == (1 << (i + 1)):
                    result.append(candidate)
                    current_sums = new_sums
                    break
                candidate += 1
                if candidate > 1048576:
                    return None
                if time.time() - start_time > deadline:
                    return None
        return result
    
    # First try greedy to get an upper bound
    greedy_result = greedy_construct()
    if greedy_result is not None:
        best_max = max(greedy_result)
    else:
        return [1 << i for i in range(n)]
    
    # A* search with better heuristic
    def heuristic(current_tuple):
        """
        Lower bound on final max element.
        For k elements with max M, the remaining (n-k) elements must be > M.
        The minimal possible max for the remaining elements, assuming they can be chosen
        to minimize the max while keeping subset sums distinct, is bounded below by:
        M + (n - k)  [weak]
        
        Better: if the current elements have sum S, the total sum of all n elements
        must be at least S + (n-k)*(M+1) [if remaining are M+1, M+2, ...].
        But we need 2^n distinct subset sums in [0, total_sum], so total_sum >= 2^n - 1.
        Thus S + (n-k)*(M+1) + ... >= 2^n - 1.
        
        Actually, a simpler admissible heuristic: the final max must be at least
        max(current_max, ceil((2^n - 1) / n) - something).
        
        For now, use: final_max >= current_max + (n - len(current_tuple))
        """
        if not current_tuple:
            return 1
        k = len(current_tuple)
        current_max = current_tuple[-1]
        return current_max + (n - k)
    
    # A* search
    open_set = []
    initial_state = ()
    initial_h = heuristic(initial_state)
    heapq.heappush(open_set, (initial_h, 0, 0, initial_state))
    
    best_solution = greedy_result
    best_max_value = best_max
    visited = set()
    
    while open_set:
        if time.time() - start_time > deadline:
            break
        
        h, current_max, step, state = heapq.heappop(open_set)
        
        if h >= best_max_value:
            continue
        
        if state in visited:
            continue
        visited.add(state)
        
        if len(state) == n:
            # Verify
            sums = {0}
            valid = True
            for x in state:
                new_sums = {s + x for s in sums}
                if len(sums | new_sums) != (1 << (len(state))):
                    valid = False
                    break
                sums = sums | new_sums
            if valid and len(state) == n:
                final_max = max(state)
                if final_max < best_max_value:
                    best_max_value = final_max
                    best_solution = list(state)
            continue
        
        # Compute current subset sums
        current_sums = {0}
        for x in state:
            current_sums = current_sums | {s + x for s in current_sums}
        
        start_candidate = current_max + 1 if current_max > 0 else 1
        
        # Try a limited number of candidates
        max_candidates = 30
        candidate = start_candidate
        for _ in range(max_candidates):
            if candidate > best_max_value:
                break
            
            new_sums = current_sums | {s + candidate for s in current_sums}
            if len(new_sums) == (1 << (step + 1)):
                new_state = state + (candidate,)
                new_h = heuristic(new_state)
                
                if new_h < best_max_value:
                    heapq.heappush(open_set, (new_h, candidate, step + 1, new_state))
            
            candidate += 1
    
    if best_solution is not None:
        return best_solution
    
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
