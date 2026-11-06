import time
import random

def _can_add(S, x):
    """
    Check if adding x to set S maintains the validity condition.
    S is a set of integers. x is an integer not in S.
    Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements in S U {x}.
    """
    S_list = list(S)
    m = len(S_list)
    
    # 1) Check if x divides the sum of any 3 distinct elements in S.
    if m >= 3:
        # Count residues of elements in S modulo x
        res_counts = {}
        for v in S_list:
            r = v % x
            res_counts[r] = res_counts.get(r, 0) + 1
        
        # We need to find a, b, c in S such that (a+b+c) % x == 0
        # Iterate pairs (a, b)
        for i in range(m - 1):
            a = S_list[i]
            ra = a % x
            for j in range(i + 1, m):
                b = S_list[j]
                rb = b % x
                target_r = (-ra - rb) % x
                
                # We need a c in S such that c % x == target_r and c != a and c != b
                avail = res_counts.get(target_r, 0)
                if ra == target_r:
                    avail -= 1
                if rb == target_r:
                    avail -= 1
                    
                if avail > 0:
                    return False
                        
    # 2) Check for each y in S, if y divides (x + u + v) for any distinct u,v in S \ {y}.
    for y in S_list:
        # We need u, v in S \ {y} such that (u + v) % y == (-x) % y
        target = (-x) % y
        
        # Count residues of S \ {y} modulo y
        res_counts_y = {}
        for v in S_list:
            if v == y:
                continue
            r = v % y
            res_counts_y[r] = res_counts_y.get(r, 0) + 1
        
        # Iterate u in S \ {y}
        for u in S_list:
            if u == y:
                continue
            ru = u % y
            needed_r = (target - ru) % y
            
            # We need a v in S \ {y, u} such that v % y == needed_r
            avail = res_counts_y.get(needed_r, 0)
            if ru == needed_r:
                avail -= 1
                
            if avail > 0:
                return False
                
    return True

def _is_valid(S):
    """Full validation."""
    S_list = list(S)
    m = len(S_list)
    for idx_a, a in enumerate(S_list):
        others = [v for v in S_list if v != a]
        if len(others) < 3:
            continue
        if not _can_add(set(others), a):
            return False
    return True

def _greedy_with_order(order, deadline):
    """Greedy selection following a specific order."""
    current_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(current_set, x):
            current_set.add(x)
        if len(current_set) >= 48:
            break
    return current_set

def _local_search_improve(S, deadline):
    """Try to improve set S by adding elements or doing swaps."""
    improved = True
    max_iters = 200
    iter_count = 0
    
    while improved and time.time() < deadline and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # 1. Try adding any unselected element
        for x in range(1, 193):
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    improved = True
                    if len(S) >= 48:
                        return S
                    break
        
        # 2. If no direct add, try 1-out 1-in swap
        if not improved:
            # Try removing one element and adding one different element
            for a in list(S):
                if time.time() > deadline:
                    break
                trial_set = S - {a}
                for x in range(1, 193):
                    if x == a or x in trial_set:
                        continue
                    if _can_add(trial_set, x):
                        S.remove(a)
                        S.add(x)
                        improved = True
                        break
                if improved:
                    break
        
        # 3. If still no improvement, try 1-out 2-in swap
        if not improved and len(S) < 48:
            best_swap = None
            for a in list(S):
                if time.time() > deadline:
                    break
                trial_set = S - {a}
                
                addable = []
                for x in range(1, 193):
                    if x in trial_set:
                        continue
                    if _can_add(trial_set, x):
                        addable.append(x)
                
                if len(addable) >= 2:
                    for i in range(len(addable)):
                        for j in range(i+1, len(addable)):
                            x = addable[i]
                            y = addable[j]
                            if _can_add(trial_set | {x}, y):
                                best_swap = (a, x, y)
                                break
                        if best_swap:
                            break
                if best_swap:
                    break
            
            if best_swap:
                a, x, y = best_swap
                S.remove(a)
                S.add(x)
                S.add(y)
                improved = True

    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 192
    best_set = set()
    
    # Strategy 1: Descending Greedy
    order = list(range(n, 0, -1))
    current_set = _greedy_with_order(order, deadline)
    if len(current_set) > len(best_set):
        best_set = current_set.copy()
    
    if len(best_set) >= 48:
        return list(best_set)

    # Strategy 2: Randomized Greedy with local search
    rng = random.Random(42)
    
    num_restarts = 1000
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Generate a random order, biased towards larger numbers
        if attempt % 3 == 0:
            # Descending with slight perturbation
            order = list(range(n, 0, -1))
            for _ in range(10):
                i = rng.randint(0, n-2)
                order[i], order[i+1] = order[i+1], order[i]
        elif attempt % 3 == 1:
            # Random shuffle
            order = list(range(1, n + 1))
            rng.shuffle(order)
        else:
            # Weighted random: larger numbers more likely
            weights = [i for i in range(1, n+1)]
            order = rng.choices(weights, k=n)
            # Ensure unique
            order = list(dict.fromkeys(order))
            # Fill in missing if any (should not happen if k=n and choices from n items with replacement? 
            # rng.choices allows replacement. We need unique.
            # Let's just use shuffle for simplicity in this branch or fix it.
            # Actually rng.choices with k=n from n items might have duplicates.
            # Let's stick to shuffle or descending.
            order = list(range(1, n + 1))
            rng.shuffle(order)
        
        current_set = _greedy_with_order(order, deadline)
        
        # Local search
        current_set = _local_search_improve(current_set, deadline)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            if len(best_set) >= 48:
                break
    
    # Final local search on best_set
    if time.time() < deadline - 2:
        best_set = _local_search_improve(best_set, deadline)
    
    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    
    # Fallback
    return [192]
