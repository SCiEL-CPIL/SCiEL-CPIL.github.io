"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 80.0  # safety margin before 90s kill
    n = 128
    
    # Precompute lcm table
    lcm_table = {}
    for a in range(1, n+1):
        for b in range(a, n+1):
            lcm_table[(a,b)] = a * b // gcd(a,b)
            if a != b:
                lcm_table[(b,a)] = lcm_table[(a,b)]
    
    def get_lcm(a, b):
        if a == b:
            return a
        if a > b:
            a, b = b, a
        return lcm_table.get((a,b), a * b // gcd(a,b))
    
    # Precompute for each x, the set of pairs (a,b) that would form a bad triple with x
    # x is blocked by pair (a,b) if lcm(a,b) == lcm(a,x) == lcm(b,x)
    blocking_pairs = {}
    for x in range(1, n+1):
        blocking_pairs[x] = set()
        for a in range(1, n+1):
            if a == x:
                continue
            la = get_lcm(a, x)
            for b in range(a+1, n+1):
                if b == x:
                    continue
                if get_lcm(a,b) == la and get_lcm(b,x) == la:
                    blocking_pairs[x].add((a, b))
    
    # Build bitmask representation for fast checking
    # For each x, store bitmask of all elements that appear in any blocking pair with x
    # Actually, we need: for each x, which pairs (a,b) block x
    # Let's precompute for each x a list of (a, b) pairs, and also a set for fast lookup
    
    # Alternative: for each x, precompute the set of "forbidden pairs" as a set of frozensets or tuples
    # And also precompute for each pair (a,b), which x values are blocked by that pair
    blocked_by_pair = {}
    for x in range(1, n+1):
        for (a, b) in blocking_pairs[x]:
            if (a, b) not in blocked_by_pair:
                blocked_by_pair[(a, b)] = set()
            blocked_by_pair[(a, b)].add(x)
    
    def greedy_with_order(order):
        selected = set()
        selected_mask = 0  # bitmask for fast membership
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
                selected_mask |= (1 << (x-1))
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
    
    # 6. Local improvement with kick-and-refill strategy
    if best and time.time() < deadline:
        current = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Kick: remove 1-3 random elements
            kick_size = random.randint(1, 3)
            kick_elements = random.sample(sorted(current), min(kick_size, len(current)))
            candidate = current - set(kick_elements)
            
            # Refill: try to add elements in random order
            refill_order = list(range(1, n+1))
            rng.shuffle(refill_order)
            
            for y in refill_order:
                if y in candidate:
                    continue
                if time.time() > deadline:
                    break
                # Check if y can be added to candidate
                can_add = True
                for (a, b) in blocking_pairs[y]:
                    if a in candidate and b in candidate:
                        can_add = False
                        break
                if can_add:
                    candidate.add(y)
            
            if len(candidate) > len(current):
                current = candidate
                improved = True
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
    
    if best is None or len(best) == 0:
        return [1]
    return sorted(best)
# EVOLVE-BLOCK-END
