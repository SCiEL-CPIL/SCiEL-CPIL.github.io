import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 90s limit, leave 5s margin
    
    # 1. Two-Phase Greedy Initialization
    # Phase 1: Greedy on [51, 100]
    # Phase 2: Greedy on [1, 50]
    selected = []
    reachable = 1
    
    # Phase 1
    for x in range(100, 50, -1):
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
            
    # Phase 2
    for x in range(50, 0, -1):
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
            
    best_set = set(selected)
    best_size = len(best_set)
    
    # 2. Local Search with Directed Moves
    current_set = best_set.copy()
    current_reachable = reachable
    
    all_elements = list(range(1, 101))
    
    def can_add(x, R):
        return not ((R >> (200 - x)) & 1)
        
    def update_reachable(R, x):
        return R | (R << x)
        
    def is_valid_set(S):
        r = 1
        for x in S:
            r |= (r << x)
            if (r >> 200) & 1:
                return False
        return True

    def compute_blocking(S, R):
        """For each element in S, compute how many missing elements it blocks."""
        missing = [x for x in all_elements if x not in S]
        blocking = {x: 0 for x in S}
        for x in S:
            # If we remove x, what's the new reachable?
            # We can't easily compute this incrementally, so estimate:
            # x blocks any missing element m where (R_without_x >> (200-m)) & 1 would be 0
            # but (R >> (200-m)) & 1 is 1
            # Approximation: count missing elements m where 200-m is reachable and 
            # involves x in some subset
            # Simpler: just count how many missing elements m have 200-m reachable
            # (these are the ones currently blocked)
            pass
        # Better approach: for each missing element, check if it's blocked
        # and attribute the blocking to elements in S
        blocked_missing = []
        for m in missing:
            if not can_add(m, R):
                blocked_missing.append(m)
        
        # For each blocked missing element, try to identify which elements in S 
        # are "responsible" by checking subsets
        # This is expensive, so use a heuristic: 
        # An element x in S "contributes to blocking" m if there exists a subset 
        # of S containing x that sums to 200-m
        # Approximate by: x contributes if (200-m) >= x and (200-m-x) is reachable 
        # from S\{x}... but we don't have S\{x} reachable easily.
        
        # Simpler heuristic: just return the count of blocked missing elements
        # and let the search try random removals
        return len(blocked_missing)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any single missing element
        missing = [x for x in all_elements if x not in current_set]
        random.shuffle(missing)
        
        for x in missing:
            if can_add(x, current_reachable):
                current_set.add(x)
                current_reachable = update_reachable(current_reachable, x)
                improved = True
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)
        
        # Phase 2: Swaps with multi-removal
        if not improved:
            # Try swaps: remove k elements, add up to k+1 new elements
            for k in [1, 2, 3]:
                if time.time() > deadline:
                    break
                    
                attempts = 200 if k == 1 else (100 if k == 2 else 50)
                for _ in range(attempts):
                    if time.time() > deadline:
                        break
                    
                    if len(current_set) < k:
                        break
                    
                    # Select k elements to remove
                    removals = random.sample(list(current_set), k)
                    
                    missing_now = [x for x in all_elements if x not in current_set]
                    if len(missing_now) < 1:
                        break
                    
                    # Try adding 1 to k+1 new elements
                    max_add = min(k + 1, len(missing_now))
                    for n_add in range(1, max_add + 1):
                        if n_add <= k:
                            continue  # Only try if we can potentially gain size
                        # Actually, we want net gain, so n_add > k
                        if n_add <= k:
                            continue
                        
                        additions = random.sample(missing_now, n_add)
                        
                        new_set = current_set.copy()
                        for x in removals:
                            new_set.remove(x)
                        for x in additions:
                            new_set.add(x)
                        
                        if len(new_set) <= best_size:
                            continue
                        
                        if is_valid_set(new_set):
                            current_set = new_set
                            current_reachable = 1
                            for x in current_set:
                                current_reachable |= (current_reachable << x)
                            
                            improved = True
                            if len(current_set) > best_size:
                                best_set = current_set.copy()
                                best_size = len(current_set)
                            break  # Restart outer loop to try additions again
                    if improved:
                        break
                if improved:
                    break

    return list(best_set)
