import time
import random

def construct():
    N = 256
    # Precompute conflicts
    # adj[i] is a set of j such that i and j conflict
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
                
    best_set = set()
    best_size = 0
    
    start_time = time.time()
    max_time = 85.0  # Safety margin
    
    # Strategy: Randomized Greedy with Local Search
    # Initial set: All odds
    odds = set(i for i in range(1, N + 1) if i % 2 == 1)
    
    # We will run multiple iterations
    # Each iteration starts with a random perturbation of the odds set
    # Or just starts with odds and randomizes the order of even additions
    
    # To maximize diversity, we can also start with random subsets of odds + some evens
    
    for restart in range(1000):
        if time.time() - start_time > max_time:
            break
            
        # Initialize current set
        # Start with all odds
        current_set = set(odds)
        
        # Randomly remove some odds to allow different even structures?
        # Let's try: Randomly remove 0-10 odds
        if restart > 0:
            num_removals = random.randint(0, 10)
            if num_removals > 0:
                odds_list = list(odds)
                random.shuffle(odds_list)
                for _ in range(num_removals):
                    current_set.remove(odds_list.pop())
                    
        # Greedy Addition Phase
        # We want to add evens (and potentially removed odds)
        # Candidates are all vertices not in current_set
        # We prioritize vertices with fewer conflicts
        
        improved = True
        while improved:
            if time.time() - start_time > max_time:
                break
                
            improved = False
            candidates = [v for v in range(1, N + 1) if v not in current_set]
            
            # Sort candidates by number of conflicts with current_set
            # To make it fast, we just calculate conflicts on the fly or sort by degree?
            # Sorting by current conflicts is better.
            
            # To avoid O(N^2) sort every time, we can just iterate and pick the best one?
            # Or sort once per iteration. N=256, so sorting 256 items is fast.
            
            def conflict_count(v):
                return len(adj[v] & current_set)
            
            candidates.sort(key=conflict_count)
            
            # Try to add the first candidate with 0 conflicts
            for v in candidates:
                if conflict_count(v) == 0:
                    current_set.add(v)
                    improved = True
                    break
            # If no 0-conflict vertex found, try 1-conflict swaps?
            # Swapping doesn't increase size, so we only do it if we are stuck and want to explore.
            # But if we swap, we might get stuck in a loop.
            # Let's try a limited number of swaps.
            if not improved:
                # Try to find a swap that might help
                # Find v with min conflicts > 0
                # If min conflicts is 1, swap it
                min_conf = float('inf')
                best_v = -1
                for v in candidates:
                    c = conflict_count(v)
                    if c < min_conf:
                        min_conf = c
                        best_v = v
                        if c == 1:
                            break
                            
                if best_v != -1 and min_conf == 1:
                    # Find the neighbor
                    nbs = adj[best_v] & current_set
                    # There is exactly 1 neighbor
                    u = next(iter(nbs))
                    # Swap: remove u, add best_v
                    current_set.remove(u)
                    current_set.add(best_v)
                    # Mark as improved to continue search?
                    # This is a neutral move. We should limit the number of such moves.
                    # For now, let's just do it and see if it helps.
                    # To prevent infinite loops, we can randomize the choice of u if there are multiple?
                    # Here there is only 1.
                    # We need a mechanism to stop cycling.
                    # Simple heuristic: don't swap back immediately?
                    # Or just accept that we might cycle and rely on time limit.
                    # Better: Only swap if it leads to a state where a 0-conflict vertex appears?
                    # That requires lookahead.
                    
                    # Let's stick to just adding 0-conflict vertices for now.
                    # The odds set is a strong start.
                    # Maybe we can just restart with different random removals.
                    pass
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    return list(best_set)
