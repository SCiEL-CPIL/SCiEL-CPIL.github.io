def construct():
    import time
    import random
    from collections import defaultdict

    deadline = time.time() + 145.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_list = list(S)
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S_set:
                    return False
        return True

    def solve_mis_bitset(n, adj_bits, time_limit, start_time):
        """
        Exact Maximum Independent Set using branch-and-bound with bitsets.
        n: number of vertices
        adj_bits[i]: bitmask of neighbors of vertex i
        Returns the size of the maximum independent set and the set itself.
        """
        import sys
        sys.setrecursionlimit(100000)

        # Upper bound: use greedy coloring on the subgraph for tighter bounds
        # We'll use a simple greedy coloring (first-fit) for the upper bound

        def greedy_color_bound(vertices_mask):
            """
            Compute a greedy coloring upper bound for the subgraph induced by vertices_mask.
            Returns (upper_bound, color_classes) where color_classes is a list of masks.
            """
            if vertices_mask == 0:
                return 0, []

            remaining = vertices_mask
            color_classes = []
            colors = 0

            while remaining:
                # Pick a vertex with max degree in remaining subgraph
                best_v = -1
                best_deg = -1
                temp = remaining
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    deg = (adj_bits[v] & remaining).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v

                # Assign this color to all non-adjacent vertices in remaining
                color_mask = 0
                available = remaining & ~adj_bits[best_v]
                while available:
                    v = (available & -available).bit_length() - 1
                    available &= available - 1
                    color_mask |= (1 << v)
                    available &= ~adj_bits[v]

                color_classes.append(color_mask)
                remaining &= ~color_mask
                colors += 1

            return colors, color_classes

        # State: (vertices_mask, current_set_size)
        # We use iterative deepening or simple recursion with pruning

        best_size = 0
        best_set_mask = 0

        def dfs(vertices_mask, current_size, current_mask):
            nonlocal best_size, best_set_mask

            if time.time() - start_time > time_limit:
                return

            if vertices_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set_mask = current_mask
                return

            # Upper bound: current_size + number of vertices in subgraph
            # Better: use greedy coloring bound
            if vertices_mask.bit_count() + current_size <= best_size:
                return

            # Get greedy coloring bound
            ub, _ = greedy_color_bound(vertices_mask)
            if current_size + ub <= best_size:
                return

            # Pick a pivot vertex (one with max degree in the subgraph)
            pivot = -1
            max_deg = -1
            temp = vertices_mask
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj_bits[v] & vertices_mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = v

            if pivot == -1:
                # All isolated vertices
                if current_size + vertices_mask.bit_count() > best_size:
                    best_size = current_size + vertices_mask.bit_count()
                    best_set_mask = current_mask | vertices_mask
                return

            # Branch 1: Include pivot
            # If we include pivot, we cannot include any of its neighbors
            new_mask = vertices_mask & ~adj_bits[pivot] & ~(1 << pivot)
            dfs(new_mask, current_size + 1, current_mask | (1 << pivot))

            # Branch 2: Exclude pivot
            dfs(vertices_mask & ~(1 << pivot), current_size, current_mask)

        dfs((1 << n) - 1, 0, 0)
        return best_size, best_set_mask

    def solve_instance(vals, time_limit=40.0):
        n = len(vals)
        vals_set = set(vals)
        start_time = time.time()

        # Build conflict graph: edge between i and j if i+j is in vals_set,
        # or if there exists k in vals_set such that i+k=j or j+k=i
        # Actually: i and j conflict if i+j is in vals, or i-j is in vals, or j-i is in vals
        # Because: if i+j=c and c is in set, then {i,j,c} violates sum-free
        #           if i-j=c and c is in set, then {c,j,i} violates (c+j=i)
        #           if j-i=c and c is in set, then {c,i,j} violates (c+i=j)

        # For each pair (i, j), check if they conflict
        adj_bits = [0] * n
        val_to_idx = {v: i for i, v in enumerate(vals)}

        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = vals[i], vals[j]
                # Check if vi + vj is in set
                if vi + vj in vals_set:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    continue
                # Check if vi - vj is in set (then vi = vj + (vi-vj))
                if vi - vj in vals_set and vi != vj:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    continue
                # Check if vj - vi is in set (then vj = vi + (vj-vi))
                if vj - vi in vals_set and vi != vj:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)

        # Also check self-loops: if 2*vi is in set, then vi conflicts with itself
        # But in independent set, we just don't include vi if 2*vi is also included
        # Actually, the condition is: no a+b=c with a,b,c in set (a=b allowed)
        # So if 2*vi = vk for some k, then vi and vk conflict
        # This is already handled above when i != j and vi + vi = vk
        # But when i == j, we need: if 2*vi is in set, then vi cannot be in the set
        # along with 2*vi. But that's a conflict between vi and vk where vk=2*vi
        # Which is handled by the pair (i, k) check

        # Run MIS solver
        try:
            best_size, best_mask = solve_mis_bitset(n, adj_bits, time_limit * 0.8, start_time)
            if best_mask != 0:
                result = set()
                for i in range(n):
                    if best_mask & (1 << i):
                        result.add(vals[i])
                if is_sum_free(result):
                    return result
        except Exception:
            pass

        # Fallback: use odds
        odds = set(x for x in vals if x % 2 != 0)
        if is_sum_free(odds):
            return odds

        # Fallback: greedy
        S = set()
        for x in sorted(vals, key=lambda x: -abs(x)):
            ok = True
            if 2 * x in S:
                ok = False
            else:
                for y in S:
                    if x + y in S or x - y in S:
                        ok = False
                        break
            if ok:
                S.add(x)
        return S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            break
        vals = instance(size, seed)

        if size == 60:
            t_lim = 30.0
        elif size == 120:
            t_lim = 50.0
        else:
            t_lim = 60.0

        S = solve_instance(vals, t_lim)

        if not is_sum_free(S):
            S = set(x for x in vals if x % 2 != 0)

        results.append(sorted(S))

    while len(results) < 3:
        results.append([])

    return results[:3]
