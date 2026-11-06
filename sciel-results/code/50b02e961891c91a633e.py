import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    all_vals = list(range(1, max_val + 1))
    
    def init_sums(S):
        Sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                Sums.add(S[i] + S[j])
        return Sums

    def can_add(x, S, Sums):
        if 2 * x in Sums:
            return False
        for y in S:
            if x + y in Sums:
                return False
        return True

    def add_element(x, S, Sums):
        Sums.add(2 * x)
        for y in S:
            Sums.add(x + y)
        S.append(x)
        S.sort()

    def remove_element(x, S, Sums):
        S.remove(x)
        Sums.discard(2 * x)
        for y in S:
            Sums.discard(x + y)

    def greedy_fill(S, Sums, order='random'):
        """Fill S greedily. order can be 'asc', 'desc', or 'random'."""
        S_set = set(S)
        if order == 'asc':
            candidates = [x for x in range(1, max_val + 1) if x not in S_set]
        elif order == 'desc':
            candidates = [x for x in range(max_val, 0, -1) if x not in S_set]
        else:
            candidates = [x for x in all_vals if x not in S_set]
            random.shuffle(candidates)
            
        for x in candidates:
            if can_add(x, S, Sums):
                add_element(x, S, Sums)
                S_set.add(x)
        return S, Sums

    def local_search(S, Sums, max_iters=100):
        """Local search with 1-swap and 2-swap."""
        S_set = set(S)
        improved = False
        iters = 0
        stuck = 0
        
        while iters < max_iters and time.time() < deadline:
            iters += 1
            
            # Try to add
            available = [x for x in all_vals if x not in S_set]
            if available:
                random.shuffle(available)
                added = False
                for x in available:
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        improved = True
                        added = True
                        stuck = 0
                        break
                if added:
                    continue
            
            # Try 1-swap
            removable = [x for x in S if x not in prescribed_set]
            if not removable:
                break
            
            random.shuffle(removable)
            swap_done = False
            for r in removable:
                remove_element(r, S, Sums)
                S_set.discard(r)
                
                available = [x for x in all_vals if x not in S_set]
                if available:
                    random.shuffle(available)
                    for x in available:
                        if can_add(x, S, Sums):
                            add_element(x, S, Sums)
                            S_set.add(x)
                            swap_done = True
                            improved = True
                            stuck = 0
                            break
                
                if swap_done:
                    break
                else:
                    add_element(r, S, Sums)
                    S_set.add(r)
            
            if swap_done:
                continue

            # Try 2-swap
            if len(removable) >= 2:
                random.shuffle(removable)
                r1, r2 = removable[0], removable[1]
                remove_element(r1, S, Sums)
                S_set.discard(r1)
                remove_element(r2, S, Sums)
                S_set.discard(r2)
                
                available = [x for x in all_vals if x not in S_set]
                if available:
                    random.shuffle(available)
                    added_count = 0
                    added_elems = []
                    for x in available:
                        if can_add(x, S, Sums):
                            add_element(x, S, Sums)
                            S_set.add(x)
                            added_elems.append(x)
                            added_count += 1
                            if added_count >= 2:
                                break
                    
                    if added_count >= 2:
                        improved = True
                        stuck = 0
                        swap_done = True
                    else:
                        for x in added_elems:
                            remove_element(x, S, Sums)
                            S_set.discard(x)
                        add_element(r1, S, Sums)
                        S_set.add(r1)
                        add_element(r2, S, Sums)
                        S_set.add(r2)
            
            if not swap_done:
                stuck += 1
                if stuck > 20:
                    break

        return S, Sums, improved

    def backtrack_search(S, Sums, target_size, depth=0):
        """Attempt to fill to target_size using backtracking."""
        if len(S) == target_size:
            return S, Sums, True
        
        if time.time() > deadline:
            return S, Sums, False
            
        # Heuristic: try candidates that are far apart
        S_set = set(S)
        candidates = [x for x in all_vals if x not in S_set]
        
        # Sort by "potential" - maybe distance to existing elements?
        # For now, random shuffle with a bias towards larger numbers if we are small
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S, Sums):
                add_element(x, S, Sums)
                S_set.add(x)
                
                new_S, new_Sums, success = backtrack_search(S, Sums, target_size, depth + 1)
                
                if success:
                    return new_S, new_Sums, True
                else:
                    remove_element(x, S, Sums)
                    S_set.discard(x)
                    
                # Pruning: if we've tried too many, stop
                if depth > 0 and depth % 50 == 0 and time.time() > deadline:
                    return S, Sums, False

        return S, Sums, False

    best_size = 0
    best_S = []
    
    # Strategies to try
    strategies = ['asc', 'desc', 'random', 'random', 'random']
    
    while time.time() < deadline:
        # Cycle through strategies
        strat = strategies[int(time.time()) % len(strategies)]
        
        S = list(prescribed)
        S.sort()
        Sums = init_sums(S)
        
        S, Sums = greedy_fill(S, Sums, order=strat)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        # Local search
        S, Sums, improved = local_search(S, Sums, max_iters=50)
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        # If we are close to 20, try backtracking to reach 20
        if best_size >= 18:
            target = 20
            # Only try backtracking if we have time
            if time.time() < deadline - 10:
                S_bt, Sums_bt, success = backtrack_search(S, Sums, target)
                if success:
                    best_size = target
                    best_S = S_bt.copy()
                    break # Found optimal
        
        if best_size >= 20:
            break

    return best_S
