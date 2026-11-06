import time
import sys

N = 256
START_TIME = time.time()
TIME_LIMIT = 25.0 # Leave 5s buffer

def construct():
    best_size = N + 1
    best_set = None
    
    def check_time():
        return time.time() - START_TIME > TIME_LIMIT

    def is_valid(x, sums, selected):
        if 2 * x in sums:
            return False
        for s in selected:
            if x + s in sums:
                return False
        return True

    def get_valid_candidates(candidates, sums, selected):
        valid = []
        for x in candidates:
            if is_valid(x, sums, selected):
                valid.append(x)
        return valid

    def compute_destruction(x, valid, sums, selected):
        # Returns number of y in valid \ {x} that become invalid if x is added
        new_sums = set(sums)
        for s in selected:
            new_sums.add(x + s)
        new_sums.add(2 * x)
        
        destroyed = 0
        for y in valid:
            if y == x:
                continue
            # Check if y is blocked by new_sums
            blocked = False
            if 2 * y in new_sums:
                blocked = True
            else:
                for s in selected:
                    if y + s in new_sums:
                        blocked = True
                        break
                if not blocked:
                    # Also check against x itself? 
                    # The condition is: y is invalid if there exists s' in selected U {x} 
                    # such that y + s' is in sums U {new sums from x}.
                    # We already checked s in selected against new_sums.
                    # We need to check s=x against new_sums?
                    # y + x in new_sums?
                    # new_sums contains x+s for s in selected, and 2x.
                    # Does new_sums contain y+x?
                    # If y+x is in new_sums, then y is blocked by s=x.
                    if y + x in new_sums:
                        blocked = True
            
            if blocked:
                destroyed += 1
        return destroyed

    def backtrack(selected, sums, candidates):
        nonlocal best_size, best_set
        
        if check_time():
            return
            
        current_size = len(selected)
        if current_size >= best_size:
            return
            
        # Check if we can form a maximal set
        # If no valid candidates, we have a maximal set
        valid = get_valid_candidates(candidates, sums, selected)
        
        if not valid:
            # We have a maximal Sidon set
            if current_size < best_size:
                best_size = current_size
                best_set = selected[:]
            return
            
        # Heuristic: Order valid candidates by destruction (highest first)
        # To save time, if valid is large, we might approximate or just pick a few top ones?
        # Given N=256, let's try full sorting.
        
        # Optimization: If current_size + 0 >= best_size, we already pruned.
        # Simple bound: We must pick at least 1 more element if valid is not empty.
        # So if current_size + 1 >= best_size, prune.
        if current_size + 1 >= best_size:
            return

        # Calculate destruction for each valid candidate
        # This is the expensive part.
        # To speed up, we can precompute something? 
        # For now, compute on the fly.
        
        scored_valid = []
        for x in valid:
            # Quick check: if x is large, it might destroy more?
            # Let's just compute.
            if check_time():
                return
            d = compute_destruction(x, valid, sums, selected)
            scored_valid.append((d, x))
            
        # Sort by destruction descending
        scored_valid.sort(reverse=True)
        
        # To avoid exploring too many branches, we can limit the branching factor
        # or just let it run until time limit.
        # Let's iterate through all, but break if time is up.
        
        for d, x in scored_valid:
            if check_time():
                return
            
            new_selected = selected + [x]
            new_sums = set(sums)
            for s in selected:
                new_sums.add(x + s)
            new_sums.add(2 * x)
            
            # New candidates: remove x and any invalidated
            # We can just pass the same candidates list and let get_valid_candidates filter,
            # but that's O(N*S). 
            # Better to filter here? 
            # Actually, passing the full list and filtering in get_valid_candidates is simpler and 
            # the cost is amortized over the recursion depth? No, it's O(N*S) per node.
            # Let's optimize get_valid_candidates to be faster?
            # N=256, S~15. 256*15 = 3840 ops. Very fast.
            
            backtrack(new_selected, new_sums, candidates)
            
            # If we found a solution of size 1, we can't do better (min size is 1? 
            # Can a set of size 1 be maximal? 
            # If S={1}, sums={2}. 
            # Can we add 2? 2+2=4, 2+1=3. 3,4 not in {2}. So 2 is valid. 
            # So {1} is not maximal.
            # Min size is likely > 1.
            if best_size == 1:
                return

    # Initial candidates
    initial_candidates = list(range(1, N + 1))
    
    # To help the search, we can start with a few random restarts or 
    # just one deep search. Backtracking is systematic.
    # However, the order of candidates in the initial list matters for the first few steps.
    # We can shuffle the initial candidates to break symmetry?
    # But `valid` is derived from `candidates`. 
    # If we shuffle `initial_candidates`, the `valid` list will be in that order.
    # The sorting by destruction will re-order them anyway.
    # So the initial order doesn't matter much after the first sort.
    
    # One issue: `compute_destruction` is called for every x in valid.
    # This is O(V^2 * S).
    # If V is 256, S is 1, it's 256^2 * 1 = 65k.
    # If S is 10, it's 650k.
    # This might be too slow for deep nodes.
    # We can limit the number of candidates we consider for branching?
    # E.g., only consider the top 10 candidates by destruction?
    # This makes it a beam search / limited branching backtracking.
    # Given the time limit, this is a good strategy.
    
    MAX_BRANCH = 10
    
    # Modified backtrack to limit branching
    def backtrack_limited(selected, sums, candidates):
        nonlocal best_size, best_set
        
        if check_time():
            return
            
        current_size = len(selected)
        if current_size >= best_size:
            return
            
        valid = get_valid_candidates(candidates, sums, selected)
        
        if not valid:
            if current_size < best_size:
                best_size = current_size
                best_set = selected[:]
            return
            
        if current_size + 1 >= best_size:
            return

        scored_valid = []
        for x in valid:
            if check_time():
                return
            d = compute_destruction(x, valid, sums, selected)
            scored_valid.append((d, x))
            
        scored_valid.sort(reverse=True)
        
        # Only explore top MAX_BRANCH candidates
        for d, x in scored_valid[:MAX_BRANCH]:
            if check_time():
                return
            
            new_selected = selected + [x]
            new_sums = set(sums)
            for s in selected:
                new_sums.add(x + s)
            new_sums.add(2 * x)
            
            backtrack_limited(new_selected, new_sums, candidates)
            
            if best_size == 1:
                return

    # Run the search
    backtrack_limited([], set(), initial_candidates)
    
    return best_set
