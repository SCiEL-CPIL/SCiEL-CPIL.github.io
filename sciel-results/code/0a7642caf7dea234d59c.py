def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    N = 180

    # Precompute all forbidden triples (a, b, c) with a < b < c, distinct, 1/a = 1/b + 1/c
    # a = bc/(b+c), so b+c must divide bc
    forbidden_triples = []
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if a < b and a >= 1:
                    forbidden_triples.append((a, b, c))

    # Build conflict structure: for each x, list of (y, z) such that {x,y,z} is forbidden
    conflicts = {i: [] for i in range(1, N + 1)}
    for (a, b, c) in forbidden_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # Also build: for each pair (y, z), which x's conflict with that pair
    # This helps with a different perspective
    pair_conflicts = {}
    for (a, b, c) in forbidden_triples:
        for pair in [(a, b), (a, c), (b, c)]:
            key = (min(pair), max(pair))
            if key not in pair_conflicts:
                pair_conflicts[key] = set()
            pair_conflicts[key].add(a if key == (b, c) else (b if key == (a, c) else c))

    def can_add(x, s):
        """Check if x can be added to set s without creating forbidden triple."""
        for (y, z) in conflicts[x]:
            if y in s and z in s:
                return False
        return True

    def is_valid(s):
        """Check if set s has no forbidden triple."""
        s_list = sorted(s)
        s_set = set(s_list)
        for (a, b, c) in forbidden_triples:
            if a in s_set and b in s_set and c in s_set:
                return False
        return True

    # Strategy: Start with a good base set, then use branch-and-bound / greedy with backtracking
    
    # Good initial heuristic: take numbers in [91, 180] (all > 90)
    # For any triple (a,b,c) with a<b<c and 1/a=1/b+1/c, we have a < b < c
    # If b >= 91 and c >= 92, then a = bc/(b+c) < b. 
    # Actually if we take all of [91,180], no two elements can form a triple with a third in [91,180]
    # because the smallest element a = bc/(b+c) < b, so if b >= 91, then a < 91, not in set.
    # Wait, that's not quite right. Let me think again.
    # If a, b, c are all in [91, 180] with a < b < c, then 1/a = 1/b + 1/c.
    # Since a < b, 1/a > 1/b, so 1/c = 1/a - 1/b = (b-a)/(ab) > 0.
    # c = ab/(b-a). Since a >= 91, b >= 92, b-a >= 1.
    # c = ab/(b-a). If b-a=1, c = ab/(1) = ab which is way > 180.
    # If b-a=2, c = ab/2. With a=91,b=93, c=91*93/2 = 4228.5, not integer.
    # Generally c = ab/(b-a) is large when b-a is small.
    # For c <= 180, we need ab/(b-a) <= 180, so ab <= 180(b-a), so ab <= 180b - 180a,
    # a(b+180) <= 180b, a <= 180b/(b+180).
    # For b=91: a <= 180*91/271 = 16380/271 ≈ 60.4. So a <= 60, but a >= 91. Contradiction.
    # For b=180: a <= 180*180/360 = 90. So a <= 90, but a >= 91. Contradiction.
    # So [91,180] is indeed valid! Size = 90.
    
    base = set(range(91, N + 1))
    best = set(base)
    best_size = len(best)

    # Now greedily try to add elements from 1 to 90
    # Use multiple orderings and keep the best
    
    orderings = [
        list(range(1, 91)),           # ascending
        list(range(90, 0, -1)),       # descending
        list(range(1, 91, 2)) + list(range(89, 0, -2)),  # odds then evens
        list(range(90, 0, -2)) + list(range(1, 91, 2)),  # evens desc then odds asc
        list(range(90, 0, -1, 2)) + list(range(89, 0, -1, 2)),  # evens desc, odds desc
    ]
    
    for ordering in orderings:
        if time.time() - start_time > DEADLINE - 10:
            break
        trial = set(base)
        for x in ordering:
            if x in trial:
                continue
            if can_add(x, trial):
                trial.add(x)
        if len(trial) > best_size:
            best = set(trial)
            best_size = len(trial)

    # Local search: iterative improvement
    # Try to add elements, and if stuck, try remove-one-add-two swaps
    improved = True
    while improved and time.time() - start_time < DEADLINE - 5:
        improved = False
        
        # Try adding any remaining element
        remaining = [x for x in range(1, N + 1) if x not in best]
        for x in remaining:
            if time.time() - start_time > DEADLINE - 5:
                break
            if can_add(x, best):
                best.add(x)
                improved = True
                best_size = len(best)
        
        # If no single add worked, try: remove one, add two
        if not improved:
            for y in list(best):
                if time.time() - start_time > DEADLINE - 5:
                    break
                trial = set(best)
                trial.discard(y)
                added = 0
                for x in range(1, N + 1):
                    if x in trial:
                        continue
                    if can_add(x, trial):
                        trial.add(x)
                        added += 1
                if len(trial) > len(best):
                    best = trial
                    best_size = len(best)
                    improved = True

    # Final verification
    if not is_valid(best):
        # Fallback to safe set
        best = set(range(91, N + 1))

    return list(best)
