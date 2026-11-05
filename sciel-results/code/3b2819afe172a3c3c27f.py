def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s limit

    N = 180

    # Precompute all forbidden triples: (a, b, c) with a < b < c and a*(b+c) == b*c
    # 1/a = 1/b + 1/c  =>  a*(b+c) = b*c  =>  c = a*b/(b-a)
    # For each element x, store the set of pairs (u,v) that conflict with x
    # i.e., if x is in the set, we cannot have both u and v in the set

    # For each x, conflict_pairs[x] = list of (u, v) such that {x, u, v} forms a forbidden triple
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    all_triples = []

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
                all_triples.append((a, b, c))

    # Convert to sets for faster membership testing
    conflict_sets = {i: set() for i in range(1, N + 1)}
    for i in range(1, N + 1):
        for pair in conflict_pairs[i]:
            conflict_sets[i].add(pair)

    # Degree: number of conflicts for each element
    degree = {i: len(conflict_pairs[i]) for i in range(1, N + 1)}

    # For fast checking: for each x, precompute a list of (u, v) pairs
    # and also build an adjacency: adj[x][y] = set of z such that {x,y,z} is forbidden
    adj = {i: {j: set() for j in range(1, N + 1)} for i in range(1, N + 1)}
    for a, b, c in all_triples:
        adj[a][b].add(c)
        adj[b][a].add(c)
        adj[a][c].add(b)
        adj[c][a].add(b)
        adj[b][c].add(a)
        adj[c][b].add(a)

    def can_add(x, in_set):
        """Check if x can be added to the current set."""
        for (u, v) in conflict_pairs[x]:
            if in_set[u] and in_set[v]:
                return False
        return True

    def can_add_fast(x, in_set):
        """Faster check using adjacency."""
        # For each pair (u,v) in conflict_pairs[x], check if both are in set
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

    def try_add_all(in_set, result, candidates=None):
        """Try to add all possible candidates repeatedly until no more can be added."""
        if candidates is None:
            candidates = range(1, N + 1)
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

    def local_search_1_for_2(in_set, result, deadline):
        """Local search: try removing one element and adding two or more."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Sort by conflict count descending - elements with more conflicts are better to remove
            sorted_result = sorted(result, key=lambda x: degree[x], reverse=True)
            for x in sorted_result:
                if time.time() > deadline:
                    break
                
                # Remove x
                in_set[x] = False
                result.discard(x)
                
                # Try to add as many as possible, ordered by lowest degree first
                candidates = [y for y in range(1, N + 1) if not in_set[y]]
                candidates.sort(key=lambda y: degree[y])
                added = []
                for y in candidates:
                    if can_add(y, in_set):
                        in_set[y] = True
                        result.add(y)
                        added.append(y)
                
                if len(added) > 1:
                    improved = True
                elif len(added) == 0:
                    # No improvement, restore x
                    in_set[x] = True
                    result.add(x)
                else:
                    # Exactly 1 added: swap. Keep the new state
                    improved = True

    def local_search_2_for_3(in_set, result, deadline):
        """Local search: try removing two elements and adding three or more."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            result_list = sorted(result, key=lambda x: degree[x], reverse=True)
            n = len(result_list)
            for i in range(min(n, 15)):
                for j in range(i + 1, min(n, 15)):
                    if time.time() > deadline:
                        return
                    x1, x2 = result_list[i], result_list[j]
                    
                    # Remove x1 and x2
                    in_set[x1] = False
                    in_set[x2] = False
                    result.discard(x1)
                    result.discard(x2)
                    
                    # Try to add as many as possible
                    candidates = [y for y in range(1, N + 1) if not in_set[y]]
                    candidates.sort(key=lambda y: degree[y])
                    added = []
                    for y in candidates:
                        if can_add(y, in_set):
                            in_set[y] = True
                            result.add(y)
                            added.append(y)
                    
                    if len(added) > 2:
                        improved = True
                    elif len(added) == 0:
                        in_set[x1] = True
                        in_set[x2] = True
                        result.add(x1)
                        result.add(x2)
                    else:
                        # Keep the new state (swap)
                        improved = True

    def shake_and_recover(in_set, result, k, deadline):
        """Remove k random elements, then greedily add back what fits."""
        if len(result) < k + 2:
            return False
        # Remove elements with highest conflict counts first
        sorted_result = sorted(result, key=lambda x: degree[x], reverse=True)
        removed = random.sample(sorted_result, min(k, len(sorted_result)))
        for x in removed:
            in_set[x] = False
            result.discard(x)
        
        # Greedily add back in order of lowest conflict count first
        candidates = [y for y in range(1, N + 1) if not in_set[y]]
        candidates.sort(key=lambda x: degree[x])
        greedy_add(in_set, result, candidates)
        try_add_all(in_set, result, range(1, N + 1))
        return True

    # Multiple greedy strategies
    best_size = 0
    best_result = None
    best_in_set = None

    def update_best(in_set, result):
        nonlocal best_size, best_result, best_in_set
        if len(result) > best_size:
            best_size = len(result)
            best_result = set(result)
            best_in_set = in_set[:]

    # Strategy 1: Start with high range, add lower descending
    in_set = [False] * (N + 1)
    result = set()
    for i in range(91, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(90, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 2: Start with high range, add lower ascending
    in_set = [False] * (N + 1)
    result = set()
    for i in range(91, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(1, 91))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 3: Pure greedy from high to low
    in_set = [False] * (N + 1)
    result = set()
    greedy_add(in_set, result, range(N, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 4: Pure greedy from low to high
    in_set = [False] * (N + 1)
    result = set()
    greedy_add(in_set, result, range(1, N + 1))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 5: Start with top 60
    in_set = [False] * (N + 1)
    result = set()
    for i in range(121, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(120, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 6: Greedy ordered by degree (lowest first)
    in_set = [False] * (N + 1)
    result = set()
    order = sorted(range(1, N + 1), key=lambda x: degree[x])
    greedy_add(in_set, result, order)
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 7: Greedy ordered by degree (highest first)
    in_set = [False] * (N + 1)
    result = set()
    order = sorted(range(1, N + 1), key=lambda x: degree[x], reverse=True)
    greedy_add(in_set, result, order)
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Randomized restarts
    random.seed(42)
    num_restarts = 0
    max_restarts = 300

    while time.time() < DEADLINE - 15 and num_restarts < max_restarts:
        num_restarts += 1
        in_set = [False] * (N + 1)
        result = set()
        
        # Randomized: pick seed elements biased toward larger numbers
        seed_size = random.randint(30, 80)
        weights = [1.0 / i for i in range(1, N + 1)]
        total_w = sum(weights)
        seeds = set()
        remaining = list(range(1, N + 1))
        for _ in range(seed_size):
            if not remaining:
                break
            w_rem = [weights[i - 1] for i in remaining]
            total = sum(w_rem)
            r = random.random() * total
            cum = 0
            idx = 0
            for j, w in enumerate(w_rem):
                cum += w
                if r <= cum:
                    idx = j
                    break
            seeds.add(remaining.pop(idx))
        
        for i in seeds:
            in_set[i] = True
            result.add(i)
        
        # Greedily add remaining in random order
        candidates = [i for i in range(1, N + 1) if not in_set[i]]
        random.shuffle(candidates)
        greedy_add(in_set, result, candidates)
        try_add_all(in_set, result, range(1, N + 1))
        update_best(in_set, result)
        
        # Also try: start empty, random greedy
        if time.time() < DEADLINE - 15:
            in_set2 = [False] * (N + 1)
            result2 = set()
            candidates2 = list(range(1, N + 1))
            random.shuffle(candidates2)
            greedy_add(in_set2, result2, candidates2)
            try_add_all(in_set2, result2, range(1, N + 1))
            update_best(in_set2, result2)

    # Use the best result for local search
    in_set = best_in_set[:]
    result = set(best_result)

    # Local search: try to increase size by removing 1 and adding 2+
    local_search_1_for_2(in_set, result, DEADLINE - 5)
    update_best(in_set, result)

    # Local search: try removing 2 and adding 3+
    local_search_2_for_3(in_set, result, DEADLINE - 5)
    update_best(in_set, result)

    # Shake and recover to escape local optima
    for k in [1, 2, 3, 4]:
        max_shakes = 80
        shakes_done = 0
        stuck_count = 0
        while time.time() < DEADLINE - 5 and shakes_done < max_shakes:
            old_size = len(result)
            shake_and_recover(in_set, result, k, DEADLINE - 5)
            if len(result) > old_size:
                update_best(in_set, result)
                stuck_count = 0
            else:
                stuck_count += 1
            # Try local search after shake
            local_search_1_for_2(in_set, result, DEADLINE - 5)
            update_best(in_set, result)
            shakes_done += 1
            if stuck_count >= 10:
                break

    # Final polish: try to add any remaining in multiple orders
    for _ in range(10):
        candidates = [y for y in range(1, N + 1) if not in_set[y]]
        candidates.sort(key=lambda x: degree[x])
        for y in candidates:
            if can_add(y, in_set):
                in_set[y] = True
                result.add(y)
        if time.time() < DEADLINE - 3:
            in_set2 = in_set[:]
            result2 = set(result)
            candidates = [y for y in range(1, N + 1) if not in_set2[y]]
            random.shuffle(candidates)
            greedy_add(in_set2, result2, candidates)
            try_add_all(in_set2, result2, range(1, N + 1))
            if len(result2) > len(result):
                in_set = in_set2
                result = result2
                update_best(in_set, result)

    return list(best_result)
