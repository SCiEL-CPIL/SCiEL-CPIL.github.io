"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 75.0  # safety margin before 90s kill
    n = 128
    
    # Precompute lcm table
    lcm_table = {}
    for a in range(1, n+1):
        for b in range(a, n+1):
            lcm_table[(a,b)] = a * b // gcd(a,b)
            if a != b:
                lcm_table[(b,a)] = a * b // gcd(a,b)
    
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
            for b in range(a+1, n+1):
                if b == x:
                    continue
                if get_lcm(a,b) == get_lcm(a,x) == get_lcm(b,x):
                    blocking_pairs[x].add((min(a,b), max(a,b)))
    
    # Convert to lists for faster iteration
    blocking_pairs_list = {}
    for x in range(1, n+1):
        blocking_pairs_list[x] = list(blocking_pairs[x])
    
    # Build bitset representation for faster conflict checking
    # For each x, we need to quickly check if any blocking pair (a,b) has both a,b in set S
    # Use a different approach: for each x, precompute a list of "conflict partners"
    # x conflicts with y if there exists z such that {x,y,z} is bad
    # But that's not quite right either.
    
    # Better: use the original check but optimize with early exit
    def can_add_to_set(x, s):
        for (a, b) in blocking_pairs_list[x]:
            if a in s and b in s:
                return False
        return True
    
    def greedy_with_order(order, s=None):
        if s is None:
            s = set()
        for x in order:
            if x in s:
                continue
            if can_add_to_set(x, s):
                s.add(x)
        return s
    
    # Try multiple orderings
    best = None
    best_size = 0
    
    # 1. Natural order
    s1 = greedy_with_order(list(range(1, n+1)))
    if len(s1) > best_size:
        best = s1
        best_size = len(s1)
    
    # 2. Reverse order
    s2 = greedy_with_order(list(range(n, 0, -1)))
    if len(s2) > best_size:
        best = s2
        best_size = len(s2)
    
    # 3. Degree-based: process elements with fewer blocking pairs first
    deg_order = sorted(range(1, n+1), key=lambda x: len(blocking_pairs_list[x]))
    s3 = greedy_with_order(deg_order)
    if len(s3) > best_size:
        best = s3
        best_size = len(s3)
    
    # 4. Reverse degree-based
    deg_order_rev = sorted(range(1, n+1), key=lambda x: -len(blocking_pairs_list[x]))
    s4 = greedy_with_order(deg_order_rev)
    if len(s4) > best_size:
        best = s4
        best_size = len(s4)
    
    # 5. LCM-diversity ordering: prefer elements with more distinct LCM values
    def lcm_diversity(x):
        lcms = set()
        for y in range(1, n+1):
            if y != x:
                lcms.add(get_lcm(x, y))
        return len(lcms)
    
    lcm_div_order = sorted(range(1, n+1), key=lambda x: -lcm_diversity(x))
    s5 = greedy_with_order(lcm_div_order)
    if len(s5) > best_size:
        best = s5
        best_size = len(s5)
    
    # 6. Random orderings with time budget
    rng = random.Random(42)
    for trial in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        s = greedy_with_order(order)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # 7. Kick-and-refill local improvement
    if best and time.time() < deadline:
        current = set(best)
        best_size = len(current)
        
        def refill(s, order=None):
            """Multi-pass greedy refill with different orderings."""
            if order is None:
                order = deg_order
            # Multiple passes with different orderings
            for _ in range(3):
                changed = False
                for y in order:
                    if y not in s and can_add_to_set(y, s):
                        s.add(y)
                        changed = True
                if not changed:
                    break
            # Try reverse order as well
            for y in reversed(order):
                if y not in s and can_add_to_set(y, s):
                    s.add(y)
            return s
        
        # Initial refill
        current = refill(set(best), deg_order)
        best_size = len(current)
        
        # Compute conflict degree: for each x, count how many potential additions are blocked by x
        # An element x blocks addition of y if there exists z in current such that (z, y) blocks x
        # Simplified: count how many pairs (a,b) in blocking_pairs[x] have at least one element in current
        conflict_degree = {}
        for x in range(1, n+1):
            deg = 0
            for (a, b) in blocking_pairs_list[x]:
                if a in current or b in current:
                    deg += 1
            conflict_degree[x] = deg
        
        # Kick-and-refill iterations with simulated annealing
        stuck_count = 0
        temperature = 1.0
        cooling = 0.999
        
        for trial in range(50000):
            if time.time() > deadline:
                break
            
            # Adaptive kick size based on stuck count
            if stuck_count < 30:
                kick_size = 2
            elif stuck_count < 100:
                kick_size = 3
            elif stuck_count < 300:
                kick_size = 4
            else:
                kick_size = rng.choice([3, 4, 5])
            
            # Kick strategy with multiple heuristics
            kick_strategy = rng.random()
            kick_set = set()
            current_list = list(current)
            
            if kick_strategy < 0.3:
                # Random kick
                rng.shuffle(current_list)
                kick_set = set(current_list[:kick_size])
            elif kick_strategy < 0.5:
                # Kick high-conflict elements
                current_list.sort(key=lambda x: -conflict_degree[x])
                kick_set = set(current_list[:kick_size])
            elif kick_strategy < 0.7:
                # Kick elements that block many potential additions
                # For each x in current, count how many y not in current would be blocked if x stayed
                block_potential = {}
                for x in current:
                    count = 0
                    for y in range(1, n+1):
                        if y not in current:
                            # Check if x blocks y: there exists z in current such that (z,y) blocks x
                            # Simplified: check if any blocking pair of x involves y
                            for (a, b) in blocking_pairs_list[x]:
                                if y in (a, b) and (a in current or b in current):
                                    count += 1
                                    break
                    block_potential[x] = count
                current_list.sort(key=lambda x: -block_potential[x])
                kick_set = set(current_list[:kick_size])
            elif kick_strategy < 0.85:
                # Kick low-conflict elements (less disruptive)
                current_list.sort(key=lambda x: conflict_degree[x])
                kick_set = set(current_list[:kick_size])
            else:
                # Hybrid: kick one high-conflict and one low-conflict
                current_list.sort(key=lambda x: -conflict_degree[x])
                kick_set.add(current_list[0])
                current_list.sort(key=lambda x: conflict_degree[x])
                kick_set.add(current_list[0])
                # Fill remaining kick slots randomly
                remaining = list(current - kick_set)
                rng.shuffle(remaining)
                for elem in remaining:
                    if len(kick_set) >= kick_size:
                        break
                    kick_set.add(elem)
            
            # Remove and refill
            candidate = current - kick_set
            candidate = refill(candidate, deg_order)
            
            # Accept if better, or with probability based on temperature if equal
            if len(candidate) > best_size:
                best_size = len(candidate)
                current = candidate
                stuck_count = 0
                temperature = 1.0
            elif len(candidate) == best_size:
                if rng.random() < 0.3 * temperature:
                    current = candidate
                stuck_count += 1
            else:
                stuck_count += 1
            
            # Cool down
            temperature *= cooling
            if temperature < 0.01:
                temperature = 0.01
        
        # Final polish: try to add any remaining elements
        current = refill(current, deg_order)
        current = refill(current, deg_order_rev)
        if len(current) > best_size:
            best_size = len(current)
            best = current
    
    if best is None or len(best) == 0:
        return [1]
    return sorted(best)
# EVOLVE-BLOCK-END
