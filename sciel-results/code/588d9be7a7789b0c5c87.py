def construct():
    import time
    import random
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s limit

    N = 180

    # Precompute all forbidden triples: (a, b, c) with a < b < c and a*(b+c) == b*c
    # For each element, store its conflict pairs
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            diff = b - a
            if diff == 0:
                continue
            num = a * b
            if num % diff != 0:
                continue
            c = num // diff
            if c > b and c <= N:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))

    def can_add(x, in_set):
        """Check if x can be added to the set without creating a forbidden triple."""
        for (u, v) in conflict_pairs[x]:
            if in_set[u] and in_set[v]:
                return False
        return True

    def greedy_add(in_set, result, order):
        """Add elements in given order if possible."""
        for x in order:
            if can_add(x, in_set):
                in_set[x] = True
                result.add(x)

    def try_add_all(in_set, result, candidates):
        """Try to add all possible candidates repeatedly until no more can be added."""
        changed = True
        while changed:
            changed = False
            for y in candidates:
                if in_set[y]:
                    continue
                if can_add(y, in_set):
                    in_set[y] = True
                    result.add(y)
                    changed = True

    def local_search_1swap(in_set, result, max_iterations=5000):
        """Try to increase size by removing 1 and adding 2+."""
        improved = True
        iters = 0
        while improved and iters < max_iterations and time.time() - start_time < DEADLINE:
            improved = False
            iters += 1
            for x in list(result):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Remove x
                in_set[x] = False
                result.discard(x)
                
                # Try to add as many as possible
                added = []
                for y in range(1, N + 1):
                    if in_set[y]:
                        continue
                    if can_add(y, in_set):
                        in_set[y] = True
                        result.add(y)
                        added.append(y)
                
                if len(added) >= 1:
                    # Keep the improvement
                    improved = True
                else:
                    # No improvement, restore x
                    in_set[x] = True
                    result.add(x)

    def simulated_annealing(in_set, result, iters_per_temp=200):
        """Simulated annealing to improve the set."""
        current_size = len(result)
        best_size = current_size
        best_in_set = in_set[:]
        best_result = set(result)
        
        temp = 10.0
        cooling = 0.95
        
        while temp > 0.1 and time.time() - start_time < DEADLINE:
            for _ in range(iters_per_temp):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Random move: remove one element and add another
                if len(result) < 2:
                    break
                    
                # Remove a random element
                to_remove = random.choice(list(result))
                in_set[to_remove] = False
                result.discard(to_remove)
                
                # Try to add a random element
                candidates = [y for y in range(1, N + 1) if not in_set[y]]
                if not candidates:
                    in_set[to_remove] = True
                    result.add(to_remove)
                    continue
                    
                to_add = random.choice(candidates)
                if can_add(to_add, in_set):
                    in_set[to_add] = True
                    result.add(to_add)
                    
                    if len(result) > best_size:
                        best_size = len(result)
                        best_in_set = in_set[:]
                        best_result = set(result)
                else:
                    # Accept with probability exp(-delta/temp)
                    # delta is the "cost" of not being able to add
                    # Here we just restore
                    in_set[to_remove] = True
                    result.add(to_remove)
            
            temp *= cooling
        
        # Restore best solution
        for i in range(1, N + 1):
            in_set[i] = best_in_set[i]
        result.clear()
        result.update(best_result)
        return best_size

    # Multiple greedy strategies
    best_size = 0
    best_result = None
    best_in_set = None

    def evaluate_and_update(in_set, result):
        nonlocal best_size, best_result, best_in_set
        if len(result) > best_size:
            best_size = len(result)
            best_result = set(result)
            best_in_set = in_set[:]

    # Strategy 1: Start with high range, add lower
    in_set = [False] * (N + 1)
    result = set()
    for i in range(91, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(90, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    evaluate_and_update(in_set, result)

    # Strategy 2: Start with high range, add lower in ascending order
    in_set = [False] * (N + 1)
    result = set()
    for i in range(91, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(1, 91))
    try_add_all(in_set, result, range(1, N + 1))
    evaluate_and_update(in_set, result)

    # Strategy 3: Pure greedy from high to low
    in_set = [False] * (N + 1)
    result = set()
    greedy_add(in_set, result, range(N, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    evaluate_and_update(in_set, result)

    # Strategy 4: Pure greedy from low to high
    in_set = [False] * (N + 1)
    result = set()
    greedy_add(in_set, result, range(1, N + 1))
    try_add_all(in_set, result, range(1, N + 1))
    evaluate_and_update(in_set, result)

    # Strategy 5: Start with multiples of some structure
    in_set = [False] * (N + 1)
    result = set()
    for i in range(121, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(120, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    evaluate_and_update(in_set, result)

    # Use the best result for local search
    in_set = best_in_set[:]
    result = set(best_result)

    # Local search: try to increase size
    local_search_1swap(in_set, result, max_iterations=3000)
    evaluate_and_update(in_set, result)

    # Simulated annealing for further improvement
    if time.time() - start_time < DEADLINE - 5:
        simulated_annealing(in_set, result, iters_per_temp=100)
        evaluate_and_update(in_set, result)

    # Final polish: try to add any remaining
    for y in range(1, N + 1):
        if not in_set[y]:
            if can_add(y, in_set):
                in_set[y] = True
                result.add(y)

    return list(result)
