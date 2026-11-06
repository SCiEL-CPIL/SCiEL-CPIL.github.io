"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 78.0  # safety margin before 90s kill
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
    
    # Precompute for each element x, the set of elements that form a bad triple with x
    # and some other element. This helps with faster conflict detection.
    # For each pair (a,b) that forms a bad triple with x, we record that x conflicts with both a and b
    # when the other is already present.
    
    # Better: for each x, create a dictionary mapping a -> set of b such that (a,b) blocks x
    # This allows O(1) lookup: if a is in current set, check if any b in blocking_map[x][a] is also in current set
    blocking_map = {}
    for x in range(1, n+1):
        blocking_map[x] = {}
        for a in range(1, n+1):
            if a == x:
                continue
            for b in range(a+1, n+1):
                if b == x:
                    continue
                if get_lcm(a,b) == get_lcm(a,x) == get_lcm(b,x):
                    blocking_map[x].setdefault(a, set()).add(b)
                    blocking_map[x].setdefault(b, set()).add(a)
    
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
    
    # Bitset-based conflict tracking for O(1) feasibility checks
    # For each x, create a bitset of elements that conflict with x
    # x conflicts with y if there exists z such that {x,y,z} is a bad triple
    # Actually, we need: x cannot be added if there exist a,b in current set
    # such that (a,b) is in blocking_pairs[x]
    
    # Alternative: for each x, precompute which pairs block it
    # Use bitsets for fast intersection checks
    
    # For each x, create a list of (a, b) pairs that block x
    # To check if x can be added to set S:
    #   For each (a,b) in blocking_pairs[x]: if a in S and b in S, return False
    
    # Faster approach: for each x, create a bitset where bit i is set if
    # x conflicts with element i (i.e., there exists some pair involving i that blocks x)
    # This doesn't directly work because the blocking is pairwise
    
    # Better: use the original approach but with optimized data structures
    # For each x, store blocking pairs as a list for faster iteration
    
    blocking_pairs_list = {}
    for x in range(1, n+1):
        blocking_pairs_list[x] = list(blocking_pairs[x])
    
    def can_add_to_set(x, s):
        # Use the blocking_map for faster checks
        for a, b_set in blocking_map[x].items():
            if a in s:
                # Check if any b in b_set is also in s
                for b in b_set:
                    if b in s:
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
    
    # Precompute a faster feasibility check using bitsets
    # For each x, create a bitset of all elements that appear in any blocking pair for x
    # Then for each such element a, we need to check if its paired elements are in s
    
    # Actually, let's use a different approach: maintain for each element y in the current set,
    # the count of how many other elements z in the set would form a bad triple with y and some x.
    # This is complex. Instead, let's optimize the refill by caching results.
    
    # For the refill, we can precompute for each candidate x which elements in the current set
    # are "dangerous" (i.e., if x were added, which pairs involving those elements would block x)
    
    # Let's try a different optimization: instead of checking all blocking pairs,
    # only check pairs where at least one element is in the current set
    def can_add_to_set_fast(x, s):
        # Only check blocking pairs where at least one element is in s
        for (a, b) in blocking_pairs_list[x]:
            if a in s or b in s:
                if a in s and b in s:
                    return False
        return True
    
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
    
    # 5. Random orderings with time budget
    rng = random.Random(42)
    # Increase the number of random trials since we have more time after optimization
    for trial in range(500):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        s = greedy_with_order(order)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # 6. Kick-and-refill local improvement with optimized refill
    if best and time.time() < deadline:
        current = set(best)
        best_size = len(current)
        
        def refill(s, order=None):
            """Optimized greedy refill: add all elements that can be added."""
            if order is None:
                order = deg_order
            # Multiple passes to ensure we add everything possible
            changed = True
            passes = 0
            while changed and passes < 5:
                changed = False
                for y in order:
                    if y not in s and can_add_to_set(y, s):
                        s.add(y)
                        changed = True
                passes += 1
            return s
        
        # Initial refill
        current = refill(set(best), deg_order)
        best_size = len(current)
        
        # Precompute conflict scores: for each x in current, count how many
        # other elements in current would be blocked if x were removed
        # Actually, we want: for each x, how many potential additions are blocked by x
        
        # Compute "block count" for each element: number of (y, z) pairs where
        # y is not in current, z is not in current, and (y,z) blocks x
        # This is complex; use a simpler heuristic
        
        # For kick strategy: kick elements that are "central" in the conflict graph
        # An element x is central if many other elements conflict with it
        
        # Compute degree in conflict graph: for each x, count how many y
        # such that x and y cannot both be in an independent set together
        # (i.e., there exists z such that {x,y,z} is a bad triple)
        
        conflict_degree = {}
        for x in range(1, n+1):
            deg = 0
            for (a, b) in blocking_pairs_list[x]:
                # If a is in current, then b conflicts with x (if b not in current)
                # Actually, this is getting complex. Use a simpler metric.
                deg += 1
            conflict_degree[x] = deg
        
        # Kick-and-refill iterations
        stuck_count = 0
        max_trials = 50000  # Increase trials since we have more time
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Adaptive kick size
            if stuck_count < 50:
                kick_size = 2
            elif stuck_count < 200:
                kick_size = 3
            elif stuck_count < 500:
                kick_size = 4
            else:
                kick_size = 5
            
            # Kick strategy
            kick_strategy = rng.random()
            kick_set = set()
            current_list = list(current)
            
            if kick_strategy < 0.4:
                # Random kick
                rng.shuffle(current_list)
                kick_set = set(current_list[:kick_size])
            elif kick_strategy < 0.7:
                # Kick high-conflict elements
                current_list.sort(key=lambda x: -conflict_degree[x])
                kick_set = set(current_list[:kick_size])
            else:
                # Kick low-conflict elements (less disruptive)
                current_list.sort(key=lambda x: conflict_degree[x])
                kick_set = set(current_list[:kick_size])
            
            # Remove and refill
            candidate = current - kick_set
            candidate = refill(candidate, deg_order)
            
            # Accept if better or equal (with some probability)
            if len(candidate) > best_size:
                best_size = len(candidate)
                current = candidate
                stuck_count = 0
            elif len(candidate) == best_size:
                if rng.random() < 0.2:
                    current = candidate
                stuck_count += 1
            else:
                stuck_count += 1
    
    if best is None or len(best) == 0:
        return [1]
    return sorted(best)
# EVOLVE-BLOCK-END
