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
    
    # 5b. Try several more structured orderings
    # 5b1. By number of distinct LCM values with other elements
    def lcm_diversity(x):
        lcms = set()
        for y in range(1, n+1):
            if y != x:
                lcms.add(get_lcm(x, y))
        return len(lcms)
    
    div_order = sorted(range(1, n+1), key=lcm_diversity)
    s5 = greedy_with_order(div_order)
    if len(s5) > best_size:
        best = s5
        best_size = len(s5)
    
    div_order_rev = sorted(range(1, n+1), key=lambda x: -lcm_diversity(x))
    s6 = greedy_with_order(div_order_rev)
    if len(s6) > best_size:
        best = s6
        best_size = len(s6)
    
    # 5. Random orderings with time budget
    rng = random.Random(42)
    for trial in range(200):
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
        
        # Precompute for each x, sorted blocking pairs by frequency of co-occurrence
        # This helps prioritize which elements to try adding first
        def can_add_to_set(x, s):
            for (a, b) in blocking_pairs[x]:
                if a in s and b in s:
                    return False
            return True
        
        def refill(s, order=None):
            """Greedy refill: add all elements that can be added, in given order."""
            if order is None:
                order = list(range(1, n+1))
            for y in order:
                if y not in s and can_add_to_set(y, s):
                    s.add(y)
            # Second pass to catch elements that became addable
            changed = True
            while changed:
                changed = False
                for y in order:
                    if y not in s and can_add_to_set(y, s):
                        s.add(y)
                        changed = True
            return s
        
        # Initial refill with degree-based order (fewer blocking pairs first)
        deg_order = sorted(range(1, n+1), key=lambda x: len(blocking_pairs[x]))
        current = refill(set(best), deg_order)
        best_size = len(current)
        
        # Precompute "constraint score" for each element: how many other elements it blocks
        # This helps identify which elements to kick for maximum impact
        element_constraint_score = {}
        for x in range(1, n+1):
            score = 0
            for (a, b) in blocking_pairs[x]:
                # Count how many times this pair appears in other elements' blocking sets
                score += 1
            element_constraint_score[x] = score
        
        # Kick-and-refill iterations with adaptive kick size
        stuck_count = 0
        for trial in range(5000):
            if time.time() > deadline:
                break
            
            # Adaptive kick size: start small, increase if stuck
            if stuck_count < 100:
                kick_size = 2
            elif stuck_count < 300:
                kick_size = 3
            elif stuck_count < 600:
                kick_size = 4
            else:
                kick_size = 5
            
            # Kick strategy: mix of random and constraint-based
            kick_strategy = rng.random()
            kick_set = set()
            
            if kick_strategy < 0.3:
                # Pure random kick
                kick_candidates = list(current)
                rng.shuffle(kick_candidates)
                kick_set = set(kick_candidates[:kick_size])
            elif kick_strategy < 0.6:
                # Kick high-constraint elements (those that block many others)
                kick_candidates = sorted(current, key=lambda x: -element_constraint_score[x])
                kick_set = set(kick_candidates[:kick_size])
            elif kick_strategy < 0.85:
                # Kick low-constraint elements (less disruptive)
                kick_candidates = sorted(current, key=lambda x: element_constraint_score[x])
                kick_set = set(kick_candidates[:kick_size])
            else:
                # Mixed: one random + rest from high-constraint
                kick_candidates = list(current)
                rng.shuffle(kick_candidates)
                kick_set.add(kick_candidates[0])
                remaining = [x for x in current if x != kick_candidates[0]]
                remaining.sort(key=lambda x: -element_constraint_score[x])
                kick_set.update(remaining[:kick_size-1])
            
            # Remove kicked elements
            candidate = current - kick_set
            
            # Refill greedily with degree-based order
            candidate = refill(candidate, deg_order)
            
            # If we got a larger set, accept
            if len(candidate) > best_size:
                best_size = len(candidate)
                current = candidate
                stuck_count = 0
            elif len(candidate) == best_size:
                # Sometimes accept same-size solutions to escape local optima
                if rng.random() < 0.3:
                    current = candidate
                stuck_count += 1
            else:
                stuck_count += 1
    
    if best is None or len(best) == 0:
        return [1]
    return sorted(best)
# EVOLVE-BLOCK-END
