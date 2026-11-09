import time

def construct():
    deadline = time.time() + 50  # safety margin
    
    def compute_reachable(S):
        """Compute all subset sums of S that are <= 200"""
        reachable = {0}
        for x in S:
            new_reachable = set()
            for s in reachable:
                ns = s + x
                if ns == 200:
                    return reachable, True
                if ns < 200:
                    new_reachable.add(ns)
            reachable |= new_reachable
        return reachable, False
    
    def can_add(reachable, x):
        """Check if adding x to a set with given reachable sums still avoids 200"""
        for s in reachable:
            ns = s + x
            if ns == 200:
                return False, None
        # Compute new reachable sums
        new_reachable = set(reachable)
        for s in reachable:
            ns = s + x
            if ns < 200:
                new_reachable.add(ns)
        return True, new_reachable
    
    # Try multiple strategies and keep the best
    best_size = 0
    best_set = set()
    
    strategies = [
        list(range(66, 0, -1)),  # descending
        list(range(1, 67)),      # ascending
        list(range(66, 0, -2)) + list(range(65, 0, -2)),  # evens desc, odds desc
    ]
    
    for order in strategies:
        if time.time() > deadline:
            break
        
        base = set(range(67, 101))
        reachable, has_200 = compute_reachable(base)
        if has_200:
            continue  # base itself has sum 200, skip this strategy
        
        current_set = set(base)
        current_reachable = reachable
        
        for x in order:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            ok, new_reachable = can_add(current_reachable, x)
            if ok:
                current_set.add(x)
                current_reachable = new_reachable
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Local improvement: try to swap elements to increase size
    if time.time() < deadline - 5:
        improved = True
        while improved:
            improved = False
            for x in range(1, 101):
                if time.time() > deadline:
                    break
                if x in best_set:
                    continue
                # Try adding x and removing one element to keep or increase size
                # First try just adding x
                reachable, _ = compute_reachable(best_set)
                ok, new_reachable = can_add(reachable, x)
                if ok:
                    best_set.add(x)
                    improved = True
                    break
                # Try swapping: remove y from best_set, add x
                for y in sorted(best_set):
                    if time.time() > deadline:
                        break
                    if y >= x:
                        continue
                    candidate = (best_set - {y}) | {x}
                    if len(candidate) < best_size:
                        continue
                    r, h = compute_reachable(candidate)
                    if not h:
                        best_set = candidate
                        improved = True
                        break
                if improved:
                    break
    
    if not best_set:
        return list(range(67, 101))
    
    return list(best_set)
