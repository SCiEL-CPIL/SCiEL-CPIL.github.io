<<<<<<< SEARCH
def construct():
    N = 128
    deadline = time.time() + 80.0  # 10s safety margin
    
    # Precompute LCM table
    lcm = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            lcm[i][j] = i * j // math.gcd(i, j)
    
    # Precompute for each x, the set of pairs (a,b) with a<b, a!=x, b!=x
    # such that lcm(a,b) == lcm(a,x) == lcm(b,x)
    bad_pairs = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if lcm[a][b] == lcm[a][x] == lcm[b][x]:
                    bad_pairs[x].add((a, b))
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without forming a bad triple."""
        for (a, b) in bad_pairs[x]:
            if a in selected_set and b in selected_set:
                return False
        return True
    
    def greedy_construct(order, selected_set=None):
        if selected_set is None:
            selected_set = set()
        for x in order:
            if x not in selected_set and can_add(selected_set, x):
                selected_set.add(x)
        return selected_set
    
    # Multiple greedy starts with different orderings
    best_set = set()
    best_size = 0
    
    # 1. Ascending
    s = greedy_construct(range(1, N + 1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
    
    # 2. Descending
    s = greedy_construct(range(N, 0, -1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
    
    # 3. Randomized greedy starts
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # Local search: try to add elements, then try swaps
    current_set = best_set
    current_size = len(current_set)
    
    all_nums = set(range(1, N + 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any missing element
        missing = list(all_nums - current_set)
        rng.shuffle(missing)
        for v in missing:
            if can_add(current_set, v):
                current_set.add(v)
                current_size += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-swap: remove one, try to add more
        removal_candidates = list(current_set)
        rng.shuffle(removal_candidates)
        for u in removal_candidates:
            if time.time() > deadline:
                break
            
            temp_set = current_set - {u}
            added = 0
            new_elems = []
            missing_for_temp = list(all_nums - temp_set)
            rng.shuffle(missing_for_temp)
            
            for v in missing_for_temp:
                if v == u:
                    continue
                if can_add(temp_set, v):
                    temp_set.add(v)
                    added += 1
                    new_elems.append(v)
            
            if added > 1:
                current_set = temp_set
                current_size = len(current_set)
                improved = True
                break
    
    if not current_set:
        current_set = {1}
    
    return list(current_set)
=======
def construct():
    N = 128
    deadline = time.time() + 80.0  # 10s safety margin
    
    # Precompute LCM table
    lcm = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            lcm[i][j] = i * j // math.gcd(i, j)
    
    # Precompute bad pairs
    bad_pairs = [set() for _ in range(N + 1)]
    conflict_degree = [0] * (N + 1)
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if lcm[a][b] == lcm[a][x] == lcm[b][x]:
                    bad_pairs[x].add((a, b))
        conflict_degree[x] = len(bad_pairs[x])
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without forming a bad triple."""
        for (a, b) in bad_pairs[x]:
            if a in selected_set and b in selected_set:
                return False
        return True

    def greedy_construct(order, initial_set=None):
        if initial_set is None:
            selected_set = set()
        else:
            selected_set = set(initial_set)
        for x in order:
            if x not in selected_set and can_add(selected_set, x):
                selected_set.add(x)
        return selected_set

    # Initial Greedy: Sort by conflict degree (ascending)
    order_by_degree = sorted(range(1, N + 1), key=lambda x: conflict_degree[x])
    order_by_degree_rev = sorted(range(1, N + 1), key=lambda x: -conflict_degree[x])
    
    best_set = set()
    best_size = 0
    
    # 1. Greedy by degree
    s = greedy_construct(order_by_degree)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
        
    # 2. Greedy by reverse degree
    s = greedy_construct(order_by_degree_rev)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)

    # 3. Randomized greedy starts
    rng = random.Random(42)
    for trial in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)

    all_nums = set(range(1, N + 1))
    
    def local_search(s):
        """Improve s using addition, 1-swap, and 2-swap."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = list(all_nums - s)
            rng.shuffle(missing)
            
            # 1. Try adding
            for v in missing:
                if can_add(s, v):
                    s.add(v)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Try 1-swap (remove u, add as many as possible)
            removal_candidates = list(s)
            rng.shuffle(removal_candidates)
            found_swap = False
            for u in removal_candidates:
                if time.time() > deadline:
                    return s
                temp_set = set(s)
                temp_set.remove(u)
                added = 0
                candidates = list(all_nums - temp_set)
                rng.shuffle(candidates)
                for v in candidates:
                    if v == u:
                        continue
                    if can_add(temp_set, v):
                        temp_set.add(v)
                        added += 1
                
                if added > 1: # Net gain
                    s = temp_set
                    improved = True
                    found_swap = True
                    break
        
            if improved:
                continue

            # 3. Try 2-swap (remove u, v, add as many as possible)
            removals = list(s)
            rng.shuffle(removals)
            # Try a few random pairs
            for i in range(min(10, len(removals) - 1)):
                if time.time() > deadline:
                    return s
                u = removals[i]
                v = removals[i+1]
                if u == v: continue
                temp_set = set(s)
                temp_set.remove(u)
                temp_set.remove(v)
                added = 0
                candidates = list(all_nums - temp_set)
                rng.shuffle(candidates)
                for w in candidates:
                    if w == u or w == v:
                        continue
                    if can_add(temp_set, w):
                        temp_set.add(w)
                        added += 1
                    
                    if added > 2: # Net gain
                        s = temp_set
                        improved = True
                        break
                if improved:
                    break

        return s

    # Local Search on best initial
    current_set = local_search(best_set)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)

    # ILS Loop
    while time.time() < deadline:
        # Perturb current_set
        perturb_set = set(current_set)
        if len(perturb_set) >= 3:
            to_remove = rng.sample(list(perturb_set), 3)
            for u in to_remove:
                perturb_set.remove(u)
        
        # Greedily add back
        missing = list(all_nums - perturb_set)
        missing.sort(key=lambda x: conflict_degree[x])
        for v in missing:
            if can_add(perturb_set, v):
                perturb_set.add(v)
        
        # Then local search
        perturb_set = local_search(perturb_set)
        
        if len(perturb_set) > best_size:
            best_size = len(perturb_set)
            best_set = set(perturb_set)
        
        # Update current_set to the perturbed version for next iteration
        # Even if it's not the best, it provides diversity
        current_set = perturb_set

    if not best_set:
        best_set = {1}
        
    return list(best_set)
>>>>>>> REPLACE
