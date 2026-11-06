    import time
    import random
    import sys

    def construct():
        start_time = time.time()
        deadline = start_time + 115  # Safety margin
        
        n = 40
        full_mask = (1 << n) - 1
        
        # adj[i] is a bitmask of neighbors of i
        adj = [0] * n
        
        def add_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        def would_create_c6(u, v):
            # Check if path of length 4 exists between u and v
            # u - x - y - z - v
            # x in N(u), z in N(v), x != z, N(x) & N(z) != 0
            
            # Optimization: 
            # Let A = N(u) & ~(1<<v)
            # Let B = N(v) & ~(1<<u)
            # We need x in A, z in B, x != z, and (N(x) & N(z)) != 0.
            
            A = adj[u] & ~(1 << v)
            B = adj[v] & ~(1 << u)
            
            # If A or B is empty, no path of length 4 starting with u-x or v-z
            if A == 0 or B == 0:
                return False
            
            # Iterate through x in A
            # For each x, check if there is any z in B (z != x) such that N(x) & N(z) != 0
            # To speed up, we can precompute union of neighbors? No, depends on z.
            
            # Bitset trick:
            # For a fixed x, we want to know if (N(x) & (Union of N(z) for z in B, z!=x)) != 0.
            # Let UnionB = 0
            # for z in bits(B):
            #     if z != x:
            #         UnionB |= adj[z]
            # if N(x) & UnionB != 0: return True
            
            # This is O(|B|) per x, so O(|A|*|B|) per check.
            # With n=40, |A|,|B| <= 38. 38*38 = 1444 ops.
            # Number of candidate edges ~ 800.
            # 800 * 1444 ~ 1M ops per pass.
            # Python is slow. 1M ops * 10 restarts * 100 passes? Might be too slow.
            
            # Better check:
            # A C6 is u-x-y-z-v.
            # y is in N(x) ∩ N(z).
            # So for each y, let X_y = N(y) ∩ A (neighbors of y in A)
            # Let Z_y = N(y) ∩ B (neighbors of y in B)
            # If there exists y such that X_y and Z_y are not disjoint (and not just {x} if x=z?), 
            # wait, if x=z, then u-x-y-x-v is not a simple cycle (u-x-y-x-v has repeated x).
            # The cycle must be simple: u, x, y, z, v distinct.
            # So x != z.
            # If X_y ∩ Z_y is non-empty, let w be the intersection.
            # If w is in X_y and w is in Z_y.
            # If w == x and w == z, then x=z.
            # If w == x but w != z, then x != z. Valid.
            # If w != x and w == z, then x != z. Valid.
            # If w != x and w != z, then x != z. Valid.
            
            # So we just need: Exists y such that (N(y) ∩ A) ∩ (N(y) ∩ B) is not empty 
            # AND it's not the case that the intersection is exactly {x} where x is also the only element in B? 
            # No, if intersection is {w}, and w is in A and w is in B.
            # If w is in A, w is a candidate for x.
            # If w is in B, w is a candidate for z.
            # If we pick x=w and z=w, then x=z.
            # But we can pick another x' in A? No, the path is u-x-y-z-v.
            # The nodes x and z are specific.
            # If N(y) ∩ A = {w} and N(y) ∩ B = {w}, then x must be w and z must be w.
            # Then x=z. This is not a simple C6.
            # So if (N(y) ∩ A) == {w} and (N(y) ∩ B) == {w}, it doesn't form a C6 via this y.
            # However, if (N(y) ∩ A) has size > 1, or (N(y) ∩ B) has size > 1, or they share multiple, we are safe?
            # Actually, if (N(y) ∩ A) ∩ (N(y) ∩ B) is non-empty, let S be that set.
            # If |S| >= 2, we can pick x, z in S with x!=z. C6 exists.
            # If |S| == 1, let S={w}.
            #   If w is the ONLY neighbor in A (for this y) and ONLY neighbor in B (for this y), then x=z=w. No C6.
            #   But if w has another neighbor in A (i.e., N(y)∩A has size > 1), we can pick x' != w. Then x' != z=w. C6 exists.
            #   So condition: For any y, if (N(y) ∩ A) ∩ (N(y) ∩ B) is not empty:
            #      Let I = (N(y) ∩ A) ∩ (N(y) ∩ B)
            #      If len(I) > 1: C6
            #      If len(I) == 1:
            #          Let w = list(I)[0]
            #          If (N(y) ∩ A) != {w} OR (N(y) ∩ B) != {w}: C6
            #          Else: No C6 from this y.
            
            # This check is O(n) per y, and we iterate y from 0 to n-1?
            # Or iterate y in N2(u) ∪ N2(v)?
            # y must be distance 2 from u and v.
            # y in N2(u) means y in N(x) for x in A.
            # y in N2(v) means y in N(z) for z in B.
            
            # Let's just iterate all y in 0..n-1. n=40.
            # For each y:
            #   Ay = adj[y] & A
            #   By = adj[y] & B
            #   I = Ay & By
            #   if I:
            #       # Count bits
            #       if I & (I - 1): # more than 1 bit
            #           return True
            #       else:
            #           # exactly 1 bit
            #           if Ay != I or By != I:
            #               return True
            # return False
            
            for y in range(n):
                Ay = adj[y] & A
                By = adj[y] & B
                I = Ay & By
                if I:
                    if I & (I - 1): # popcount > 1
                        return True
                    else:
                        if Ay != I or By != I:
                            return True
            return False

        # Greedy construction
        best_count = 0
        best_adj = None
        
        # Strategy: Multiple restarts
        # 1. Random dense start, remove C6s? (Slow to find all C6s)
        # 2. Greedy add from sparse start.
        
        def greedy_construct(p_init):
            adj_local = [0] * n
            # Initialize with random edges
            if p_init > 0:
                for u in range(n):
                    for v in range(u+1, n):
                        if random.random() < p_init:
                            add_edge_to_local(adj_local, u, v)
            
            # Greedy add remaining edges
            # To make it deterministic/fast, we shuffle the list of all non-edges once
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj_local[u] & (1 << v)):
                        candidates.append((u, v))
            
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > deadline:
                    return None
                if not (adj_local[u] & (1 << v)):
                    if not would_create_c6_local(adj_local, u, v):
                        add_edge_to_local(adj_local, u, v)
            
            # Count edges
            count = sum(bin(adj_local[i]).count("1") for i in range(n)) // 2
            return count, adj_local

        def add_edge_to_local(adj_local, u, v):
            adj_local[u] |= (1 << v)
            adj_local[v] |= (1 << u)

        def would_create_c6_local(adj_local, u, v):
            A = adj_local[u] & ~(1 << v)
            B = adj_local[v] & ~(1 << u)
            if A == 0 or B == 0:
                return False
            
            for y in range(n):
                Ay = adj_local[y] & A
                By = adj_local[y] & B
                I = Ay & By
                if I:
                    if I & (I - 1):
                        return True
                    else:
                        if Ay != I or By != I:
                            return True
            return False

        restarts = 0
        while time.time() < deadline:
            restarts += 1
            # Vary initial density
            p = 0.05 + (restarts % 5) * 0.05
            res = greedy_construct(p)
            if res is None:
                break
            count, adj_local = res
            if count > best_count:
                best_count = count
                best_adj = adj_local[:]
            
            # If we hit 105, we can break or try to polish
            if best_count >= 105:
                # Try to add more? The greedy is already maximal for that order.
                # We can try a different order or local search (swap edges)
                pass

        # Convert best_adj to list of edges
        if best_adj is None:
            # Fallback: Empty graph or K5s
            edges = []
            for k in range(8):
                for a in range(5):
                    for b in range(a+1, 5):
                        u = 5*k + a
                        v = 5*k + b
                        edges.append([u, v])
            return edges

        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if best_adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    