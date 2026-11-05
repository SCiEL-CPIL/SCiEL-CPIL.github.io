import time
import random
import math

def _count_collisions(values):
    """Return number of product collisions."""
    n = len(values)
    if n == 0:
        return 0
    prods = []
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            prods.append(prod)
    
    seen = set()
    collisions = 0
    for p in prods:
        if p in seen:
            collisions += 1
        else:
            seen.add(p)
    return collisions

def _is_valid(values):
    return _count_collisions(values) == 0

def _greedy_build(candidates, order_func):
    """Build a sequence greedily using a specific order of candidates."""
    vals = []
    for x in order_func(candidates):
        test = vals + [x]
        if _is_valid(test):
            vals = test
    return vals

def construct():
    deadline = time.time() + 55.0
    candidates = list(range(2, 129))
    
    # 1. Multiple Greedy Initializations
    # Ascending
    init1 = _greedy_build(candidates, lambda c: c)
    # Descending
    init2 = _greedy_build(candidates, lambda c: sorted(c, reverse=True))
    # Random (few tries)
    init3 = []
    for _ in range(5):
        random.shuffle(candidates)
        temp = _greedy_build(candidates, lambda c: c)
        if len(temp) > len(init3):
            init3 = temp
            
    # Pick the best initial solution
    best_init = max([init1, init2, init3], key=len)
    
    current_list = list(best_init)
    current_set = set(current_list)
    current_collisions = _count_collisions(current_list)
    current_score = len(current_list) - 10 * current_collisions
    
    best_vals = list(current_list) if current_collisions == 0 else []
    best_score = len(best_vals)
    
    # 2. Simulated Annealing
    T = 5.0
    cooling_rate = 0.9998
    min_T = 0.1
    
    # Precompute available list for faster access? 
    # available = [v for v in candidates if v not in current_set]
    # This changes every step, so we have to recompute or maintain.
    # Recomputing is O(128) which is fine.
    
    while time.time() < deadline:
        # Generate neighbor
        r = random.random()
        if r < 0.5:
            # Swap
            if not current_list:
                continue
            u = random.choice(current_list)
            available = [v for v in candidates if v not in current_set]
            if not available:
                continue
            v = random.choice(available)
            new_list = [v if x == u else x for x in current_list]
            new_list.sort()
        elif r < 0.75:
            # Add
            available = [v for v in candidates if v not in current_set]
            if not available:
                continue
            v = random.choice(available)
            new_list = current_list + [v]
            new_list.sort()
        else:
            # Remove
            if not current_list:
                continue
            u = random.choice(current_list)
            new_list = [x for x in current_list if x != u]
        
        new_collisions = _count_collisions(new_list)
        new_score = len(new_list) - 10 * new_collisions
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_list = new_list
            current_set = set(new_list)
            current_collisions = new_collisions
            current_score = new_score
            
            if current_collisions == 0:
                if len(current_list) > best_score:
                    best_score = len(current_list)
                    best_vals = list(current_list)
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    # 3. Local Search Polish
    # Try to insert any missing number into any position
    # This is a greedy hill-climbing on the valid sequence
    improved = True
    while improved:
        improved = False
        current_best = list(best_vals)
        for x in candidates:
            if x in current_best:
                continue
            for i in range(len(current_best) + 1):
                test_list = current_best[:i] + [x] + current_best[i:]
                if _is_valid(test_list):
                    best_vals = test_list
                    improved = True
                    break
            if improved:
                break
                
    return best_vals
