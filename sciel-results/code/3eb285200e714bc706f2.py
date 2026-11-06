def construct():
    import time
    import random
    from math import gcd
    from itertools import combinations

    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute gcd matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            G[i][j] = g
            G[j][i] = g

    def is_bad_triple(a, b, c):
        return G[a][b] == G[a][c] == G[b][c]

    def can_add(x, S_list, S_set):
        # S_list contains elements in S_set
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                g_ab = G[a][b]
                if G[x][a] == g_ab and G[x][b] == g_ab:
                    return False
        return True

    def is_valid(S_list):
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                g_ab = G[a][b]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if G[a][c] == g_ab and G[b][c] == g_ab:
                        return False
        return True

    # Calculate conflict degree: number of pairs (u,v) such that (u,v,x) is a bad triple
    # This helps in ordering for B&B
    conflict_degree = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if is_bad_triple(a, b, c):
                    conflict_degree[a] += 1
                    conflict_degree[b] += 1
                    conflict_degree[c] += 1

    # Phase 1: Greedy Heuristic to get a strong initial bound
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    # Strategy 1: Sort by conflict degree descending (hardest first)
    order1 = sorted(nums, key=lambda x: -conflict_degree[x])
    # Strategy 2: Sort by conflict degree ascending (easiest first)
    order2 = sorted(nums, key=lambda x: conflict_degree[x])
    # Strategy 3: Random shuffles
    orders = [order1, order2]
    for _ in range(20):
        o = list(nums)
        rng.shuffle(o)
        orders.append(o)

    def greedy_solve(order):
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            if can_add(x, S, S_set):
                S.append(x)
                S_set.add(x)
        return S

    for order in orders:
        S = greedy_solve(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Phase 2: Branch and Bound
    # Sort by conflict degree descending to prune early?
    # Or ascending? 
    # Usually, placing high-degree nodes first allows earlier detection of conflicts.
    b_b_order = sorted(nums, key=lambda x: -conflict_degree[x])
    # Map original numbers to their index in b_b_order? 
    # We will iterate through b_b_order.
    
    start_time = time.time()
    time_limit = 50.0 # Leave 10s for overhead
    
    # We need a fast upper bound.
    # Upper bound: best_size is the current max.
    # In DFS, if current_size + remaining_count <= best_size, prune.
    
    def dfs(idx, current_set, current_list, current_size):
        nonlocal best_size, best_set, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False # Signal timeout
            
        # Pruning
        if current_size + (N - idx) <= best_size:
            return True
            
        if idx == N:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_list)
            return True

        x = b_b_order[idx]
        
        # Option 1: Include x
        if can_add(x, current_list, current_set):
            current_set.add(x)
            current_list.append(x)
            dfs(idx + 1, current_set, current_list, current_size + 1)
            current_list.pop()
            current_set.remove(x)
            
        # Option 2: Exclude x
        dfs(idx + 1, current_set, current_list, current_size)
        
        return True

    # Initialize B&B with empty set? 
    # It's better to start with the best set found in Phase 1?
    # B&B usually explores the whole space. 
    # However, starting with empty set and relying on pruning is standard.
    # But if the space is huge, we might timeout without finding the optimum.
    # The greedy phase gives a lower bound `best_size`.
    # The B&B will try to beat it.
    
    # To speed up B&B, we can use a "forward checking" or "constraint propagation" if possible, 
    # but for 3-uniform hypergraphs it's complex.
    
    # Let's run B&B.
    # We use a list for current_list to pass to can_add.
    # We use a set for current_set for O(1) membership if needed (though can_add uses list).
    
    dfs(0, set(), [], 0)

    # If B&B timed out, best_set is the best found so far.
    # Ensure validity
    if not is_valid(best_set):
        # Fallback to repair or just use a known valid small set
        # Since B&B only adds valid elements, best_set should be valid unless there's a bug.
        # But if it timed out in the middle of a recursion, best_set is from a completed leaf or initial.
        # The initial best_set from greedy is valid.
        pass

    if not best_set:
        best_set = [2]

    return best_set
