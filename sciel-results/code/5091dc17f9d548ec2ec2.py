def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c, distinct, 1/a = 1/b + 1/c
    # a = bc/(b+c), so b+c must divide bc
    # For efficiency: iterate over b, c and check if a = bc/(b+c) is integer and a < b
    forbidden = set()  # store frozensets or sorted tuples
    forbidden_triples = []
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if a < b and a >= 1:
                    # a, b, c are distinct since a < b < c
                    forbidden_triples.append((a, b, c))
                    forbidden.add((a, b, c))
    
    # Build conflict map: for each element x, which other elements conflict with it
    # conflict[x] = set of (u, v) pairs such that (x, u, v) is a forbidden triple
    # Better: for each x, build set of "conflict pairs"
    # Actually, let's build: for each x, the set of y such that there exists z with {x,y,z} forbidden
    # And more precisely: for each x, dict from y to list of z
    
    # For greedy: when adding x, check if any (x, u, v) in forbidden with u,v in current set
    # Or (u, x, v) or (u, v, x) - but since we store sorted (a<b<c), 
    # x can be the smallest, middle, or largest
    
    # Let's precompute for each x, the list of (y, z) such that {x,y,z} is forbidden
    conflicts = {i: [] for i in range(1, N + 1)}
    for (a, b, c) in forbidden_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Start with base set [91..180]
    base = set(range(91, N + 1))
    current = set(base)
    
    # Greedily add numbers from 1 to 90
    # Try each number, check if adding it creates any forbidden triple
    def can_add(x, s):
        """Check if x can be added to set s without creating forbidden triple."""
        for (y, z) in conflicts[x]:
            if y in s and z in s:
                return False
        return True
    
    # Try adding from 1 to 90
    for x in range(1, 91):
        if time.time() - start_time > DEADLINE:
            break
        if can_add(x, current):
            current.add(x)
    
    # Local search: try to add more by removing one and adding two
    # Or try different orderings
    best = set(current)
    best_size = len(best)
    
    # Try greedy with different orderings
    orderings = [
        list(range(1, 91)),           # ascending
        list(range(90, 0, -1)),       # descending
        list(range(1, 91, 2)) + list(range(89, 0, -2)),  # odds then evens
        list(range(90, 0, -2)) + list(range(1, 91, 2)),  # evens desc then odds asc
    ]
    
    for ordering in orderings:
        if time.time() - start_time > DEADLINE:
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
    
    # Local search: try to improve by swapping
    # For each x not in best, try to find y in best such that removing y and adding x keeps validity
    # and doesn't reduce size (it's a swap, so same size) - but we want to ADD, so try:
    # Remove one element, then try to add two new elements
    
    # Actually, let's do a more aggressive local search:
    # Try to add elements one by one, and if stuck, try removing one to unblock
    
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        # Try adding any remaining element
        remaining = [x for x in range(1, N + 1) if x not in best]
        for x in remaining:
            if time.time() - start_time > DEADLINE:
                break
            if can_add(x, best):
                best.add(x)
                improved = True
        
        # If no single add worked, try: remove one, add two
        if not improved:
            # Try removing each element and see if we can add more
            for y in list(best):
                if time.time() - start_time > DEADLINE:
                    break
                trial = set(best)
                trial.discard(y)
                added_count = 0
                for x in range(1, N + 1):
                    if x in trial:
                        continue
                    if can_add(x, trial):
                        trial.add(x)
                        added_count += 1
                if len(trial) > len(best):
                    best = trial
                    improved = True
    
    if time.time() - start_time > DEADLINE:
        pass  # Return what we have
    
    return list(best)
