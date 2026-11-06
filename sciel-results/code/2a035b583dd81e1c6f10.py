"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations
    from math import lcm
    
    deadline = time.time() + 80.0  # safety margin before 90s kill
    n = 128
    
    # Precompute lcm table
    lcm_table = {}
    for a in range(1, n+1):
        for b in range(a, n+1):
            lcm_table[(a,b)] = lcm(a,b)
            if a != b:
                lcm_table[(b,a)] = lcm(a,b)
    
    def get_lcm(a, b):
        if a == b:
            return a
        if a > b:
            a, b = b, a
        return lcm_table.get((a,b), lcm(a,b))
    
    # Precompute for each x, the set of pairs (a,b) that would form a bad triple with x
    # x is blocked by pair (a,b) if lcm(a,b) == lcm(a,x) == lcm(b,x)
    blocking_pairs = {}
    for x in range(1, n+1):
        blocking_pairs[x] = set()
        for a in range(1, n+1):
            if a == x:
                continue
            for b in range(a+1, n+1):
                if b == x:
                    continue
                if get_lcm(a,b) == get_lcm(a,x) == get_lcm(b,x):
                    blocking_pairs[x].add((min(a,b), max(a,b)))
    
    def greedy_with_order(order):
        selected = set()
        for x in order:
            if x in selected:
                continue
            # Check if any blocking pair is fully in selected
            can_add = True
            for (a, b) in blocking_pairs[x]:
                if a in selected and b in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(x)
        return selected
    
    # Try multiple orderings
    best = None
    best_size = 0
    
    # 1. Natural order
    order1 = list(range(1, n+1))
    s1 = greedy_with_order(order1)
    if len(s1) > best_size:
        best = s1
        best_size = len(s1)
    
    # 2. Reverse order
    order2 = list(range(n, 0, -1))
    s2 = greedy_with_order(order2)
    if len(s2) > best_size:
        best = s2
        best_size = len(s2)
    
    # 3. Degree-based: process elements with fewer blocking pairs first
    deg_order = sorted(range(1, n+1), key=lambda x: len(blocking_pairs[x]))
    s3 = greedy_with_order(deg_order)
    if len(s3) > best_size:
        best = s3
        best_size = len(s3)
    
    # 4. Reverse degree-based
    deg_order_rev = sorted(range(1, n+1), key=lambda x: -len(blocking_pairs[x]))
    s4 = greedy_with_order(deg_order_rev)
    if len(s4) > best_size:
        best = s4
        best_size = len(s4)
    
    # 5. Random orderings with time budget
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        s = greedy_with_order(order)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # 6. Kick-and-refill local improvement (faster randomized strategy)
    if best and time.time() < deadline:
        current = set(best)
        best_size = len(current)
        
        def can_add_to_set(x, s):
            for (a, b) in blocking_pairs[x]:
                if a in s and b in s:
                    return False
            return True
        
        def refill(s):
            """Greedy refill: add all elements that can be added."""
            changed = True
            while changed:
                changed = False
                for y in range(1, n+1):
                    if y not in s and can_add_to_set(y, s):
                        s.add(y)
                        changed = True
            return s
        
        # Initial refill
        current = refill(set(best))
        best_size = len(current)
        
        # Kick-and-refill iterations
        kick_size = 3  # number of elements to randomly remove
        for trial in range(2000):
            if time.time() > deadline:
                break
            
            # Kick: remove random elements
            kick_candidates = list(current)
            rng.shuffle(kick_candidates)
            kick_set = set(kick_candidates[:kick_size])
            
            # Remove kicked elements
            candidate = current - kick_set
            
            # Refill greedily
            candidate = refill(candidate)
            
            # If we got a larger set, accept
            if len(candidate) > best_size:
                best_size = len(candidate)
                current = candidate
            elif len(candidate) == best_size:
                # Sometimes accept same-size solutions to escape local optima
                if rng.random() < 0.3:
                    current = candidate
    
    if best is None or len(best) == 0:
        return [1]
    return sorted(best)
# EVOLVE-BLOCK-END
