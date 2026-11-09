def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 55  # safety margin
    
    n = 128
    # Precompute gcd table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute for each element x, and each pair (a,b), whether adding x creates a bad triple
    # conflict[x][a][b] = True if gcd(a,b) == gcd(a,x) == gcd(b,x)
    # But that's too much memory. Instead, for each x, precompute which pairs (a,b) conflict with x.
    
    # Alternative: for each x, precompute a bitmask of "conflicting pairs" - but pairs are O(n^2)
    
    # Better approach: for a given set S (as bitmask), check if adding x is valid by checking
    # all pairs in S. But we can optimize by precomputing for each x, for each a, the set of b
    # such that gcd(a,b) == gcd(a,x) == gcd(b,x).
    
    # Actually, let's precompute: for each x and each a (x != a), the set of b such that
    # gcd(a,b) == gcd(a,x) == gcd(b,x). Call this conflict[x][a] as a bitmask.
    
    # This is O(n^3) precomputation but with bitmasks it's manageable.
    
    # conflict[x][a] = bitmask of b where gcd(a,b) == gcd(a,x) == gcd(b,x)
    # When checking if we can add x to set S: for each a in S, if (S & conflict[x][a]) != 0, 
    # then there exists b in S such that (a,b,x) is a bad triple.
    
    # Wait, we need b != a and b != x and a != x. Since we're adding x to S, x is not in S yet.
    # And b must be in S, b != a.
    
    # Let me precompute this.
    
    # conflict[x][a] = bitmask of b (b != a, b != x) such that gcd(a,b) == gcd(a,x) == gcd(b,x)
    
    # But actually, we just need: for each pair (a,b) in S, check if gcd(a,b) == gcd(a,x) == gcd(b,x)
    # This is equivalent to: for each a in S, check if there exists b in S (b != a) such that
    # gcd(a,b) == gcd(a,x) AND gcd(b,x) == gcd(a,x)
    # i.e., gcd(a,b) == gcd(a,x) AND gcd(a,x) == gcd(b,x)
    
    # So for fixed x and a, the conflicting b's are those where gcd(a,b) == gcd(a,x) AND gcd(b,x) == gcd(a,x)
    
    # Let's precompute this as bitmasks.
    
    # For each x in 1..128, for each a in 1..128, a != x:
    #   g_ax = gcd(a, x)
    #   conflict[x][a] = bitmask of b where b != a, b != x, gcd(a,b) == g_ax, gcd(b,x) == g_ax
    
    # This is O(n^3) but n=128, so 128^3 = 2M operations, which is fine.
    
    conflict = [[0]*(n+1) for _ in range(n+1)]
    for x in range(1, n+1):
        for a in range(1, n+1):
            if a == x:
                continue
            g_ax = gcd_table[a][x]
            mask = 0
            for b in range(1, n+1):
                if b == a or b == x:
                    continue
                if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                    mask |= (1 << b)
            conflict[x][a] = mask
    
    def can_add(x, S):
        """Check if we can add x to set S (S is a bitmask)."""
        # For each a in S, check if conflict[x][a] intersects with S
        # (excluding a itself, but conflict[x][a] already excludes a and x)
        temp = S
        while temp:
            lb = temp & (-temp)
            a = lb.bit_length() - 1
            if conflict[x][a] & S:
                return False
            temp ^= lb
        return True
    
    def is_valid(S):
        """Check if set S is valid (no bad triples)."""
        # For each triple, check... but that's O(n^3). Let's use the conflict structure.
        # A set is valid iff for every x in S, and every a in S, conflict[x][a] & S == 0
        # Wait, that's not quite right. The conflict[x][a] tells us which b would form a bad triple
        # with a and x. So S is valid iff for all x in S, for all a in S, conflict[x][a] & S == 0.
        # But this double-counts. Let's just check all pairs.
        temp = S
        while temp:
            lb_x = temp & (-temp)
            x = lb_x.bit_length() - 1
            temp ^= lb_x
            temp2 = S
            while temp2:
                lb_a = temp2 & (-temp2)
                a = lb_a.bit_length() - 1
                temp2 ^= lb_a
                if a == x:
                    continue
                if conflict[x][a] & S:
                    return False
        return True
    
    # Greedy build with random ordering
    def greedy_build(order):
        S = 0
        for x in order:
            if can_add(x, S):
                S |= (1 << x)
        return S
    
    # Local search: try to add elements, or swap
    def local_search(S, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add any element
            for x in range(1, n+1):
                if (S >> x) & 1:
                    continue
                if can_add(x, S):
                    S |= (1 << x)
                    improved = True
                    break
            if improved:
                continue
            # Try swap: remove one, add two different ones
            # First, try removing one and adding one
            for a in range(1, n+1):
                if not ((S >> a) & 1):
                    continue
                S2 = S & ~(1 << a)
                for x in range(1, n+1):
                    if (S2 >> x) & 1:
                        continue
                    if can_add(x, S2):
                        S = S2 | (1 << x)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try remove one, add two
            for a in range(1, n+1):
                if not ((S >> a) & 1):
                    continue
                S2 = S & ~(1 << a)
                # Try to add two elements to S2
                added = 0
                for x in range(1, n+1):
                    if (S2 >> x) & 1:
                        continue
                    if can_add(x, S2):
                        S2 |= (1 << x)
                        added += 1
                        if added == 2:
                            break
                if added >= 2:
                    S = S2
                    improved = True
                    break
        return S
    
    # Run multiple greedy builds with different random orderings
    best_S = 0
    best_size = 0
    
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        S = greedy_build(order)
        if S.bit_count() > best_size:
            best_S = S
            best_size = S.bit_count()
    
    # Local search on the best solution
    if best_S and time.time() < deadline - 5:
        best_S = local_search(best_S, deadline - 2)
    
    # Convert bitmask to list
    result = [i for i in range(1, n+1) if (best_S >> i) & 1]
    
    if not result:
        result = [1]
    
    return result
