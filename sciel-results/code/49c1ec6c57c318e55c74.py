import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 118.0   # safety margin below 120s

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_incompatible(edges, adj):
        m = len(edges)
        incompatible = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
                    continue
                if (u2 in adj[u1] or v2 in adj[u1] or
                        u2 in adj[v1] or v2 in adj[v1]):
                    incompatible[i].add(j)
                    incompatible[j].add(i)
        return incompatible

    def count_colors(colors):
        return max(colors) + 1 if colors else 0

    def dsatur_greedy(incompatible, m, order):
        """Standard DSATUR greedy coloring."""
        colors = [-1] * m
        saturation = [set() for _ in range(m)]
        degree = [len(incompatible[i]) for i in range(m)]
        order_idx = {v: i for i, v in enumerate(order)}
        colored = [False] * m
        for _ in range(m):
            best_v = -1
            best_sat = -1
            best_deg = -1
            best_ord = 10**9
            for v in range(m):
                if not colored[v]:
                    sat = len(saturation[v])
                    if (sat > best_sat or
                            (sat == best_sat and degree[v] > best_deg) or
                            (sat == best_sat and degree[v] == best_deg and order_idx[v] < best_ord)):
                        best_v = v
                        best_sat = sat
                        best_deg = degree[v]
                        best_ord = order_idx[v]
            if best_v == -1:
                break
            used = set(saturation[best_v])
            c = 0
            while c in used:
                c += 1
            colors[best_v] = c
            colored[best_v] = True
            for u in incompatible[best_v]:
                if not colored[u]:
                    saturation[u].add(c)
        return colors

    def solve_exact(incompatible, m, time_limit, initial_k, rng):
        """
        Branch and Bound for Graph Coloring.
        Returns the best coloring found within time_limit.
        """
        # Precompute degrees
        degree = [len(incompatible[i]) for i in range(m)]
        
        # State for B&B
        colors = [-1] * m
        saturation = [set() for _ in range(m)]
        colored_count = 0
        best_colors = None
        best_k = initial_k
        nodes_explored = 0
        start = time.time()

        # Order vertices by descending degree (static ordering for B&B is simpler to implement correctly with pruning)
        # Actually, dynamic DSATUR ordering is much stronger. 
        # We will implement a recursive DSATUR-like B&B.
        
        # To avoid deep recursion issues and overhead, we use a stack-based approach or careful recursion.
        # Given m <= 200, recursion depth is manageable.
        
        # Helper to get next vertex to color (DSATUR rule)
        def get_next_vertex(colored_mask, sat_sizes):
            # This is O(m^2) if done naively. For m=200, 40000 ops per node is too slow for many nodes.
            # We need a faster way to select the next vertex.
            # Optimization: Maintain a list of uncolored vertices and pick the one with max saturation.
            # For simplicity and speed in Python, we will iterate through uncolored vertices.
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(m):
                if not (colored_mask >> v) & 1:
                    sat = len(saturation[v])
                    if sat > best_sat or (sat == best_sat and degree[v] > best_deg):
                        best_v = v
                        best_sat = sat
                        best_deg = degree[v]
            return best_v

        def search(colored_mask, k_used):
            nonlocal best_colors, best_k, nodes_explored, time_limit, start
            
            if time.time() - start > time_limit:
                return False
            
            nodes_explored += 1
            
            if colored_mask == (1 << m) - 1:
                if k_used < best_k:
                    best_k = k_used
                    best_colors = colors[:]
                return True
            
            # Pruning: if k_used + (some lower bound on remaining) >= best_k, prune
            # Simple LB: 0. Better: count uncolored vertices that have all k_used colors in their saturation.
            # If a vertex has saturation size == k_used, it MUST use a new color (k_used + 1).
            # Let's check if we can prune based on "forced" new colors.
            
            v = get_next_vertex(colored_mask, saturation)
            if v == -1:
                return True
            
            # Try colors 0..k_used (if k_used < best_k - 1, we might need new color)
            # Max color we can try: min(k_used, best_k - 1)
            # If we must use a new color, try k_used.
            
            max_c = k_used
            if k_used < best_k - 1:
                max_c = k_used
            else:
                max_c = k_used # Can't exceed best_k - 1 if we want to improve, but we can equal it? 
                # Actually, if k_used == best_k - 1, trying color k_used (new) would make total best_k, which is not an improvement.
                # So we only try colors 0..k_used-1 if k_used == best_k - 1? 
                # No, if we use a new color, the count becomes k_used + 1.
                # If k_used + 1 >= best_k, we can prune this branch UNLESS we are looking for a solution with exactly best_k?
                # We want to find k < best_k.
                # So if k_used + 1 >= best_k, we should NOT try assigning a new color (index k_used) because it won't improve best_k.
                # We can only try existing colors 0..k_used-1.
                
                # Correction:
                # Current used colors: 0..k_used-1.
                # If we assign color c < k_used, total colors remains k_used.
                # If we assign color c == k_used, total colors becomes k_used + 1.
                # If k_used + 1 >= best_k, then assigning a new color cannot lead to a solution with < best_k colors.
                # So we only try c in 0..k_used-1 in that case.
                
                if k_used >= best_k - 1:
                    max_c = k_used - 1
                else:
                    max_c = k_used

            # Sort colors by preference? Usually 0..k_used-1 is fine.
            for c in range(max_c + 1):
                # Check if c is valid for v
                valid = True
                for u in incompatible[v]:
                    if (colored_mask >> u) & 1 and colors[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign c to v
                colors[v] = c
                new_mask = colored_mask | (1 << v)
                new_k = k_used
                if c == k_used:
                    new_k = k_used + 1
                
                # Update saturation for neighbors
                touched = []
                for u in incompatible[v]:
                    if not (new_mask >> u) & 1:
                        if c not in saturation[u]:
                            saturation[u].add(c)
                            touched.append(u)
                
                if search(new_mask, new_k):
                    # Backtrack saturation
                    for u in touched:
                        saturation[u].discard(c)
                    colors[v] = -1
                    return True # Found a better solution? No, search returns True if it pruned or found solution. 
                    # We need to continue searching to find even better? 
                    # B&B usually continues to prove optimality or find better.
                    # If we found a solution with k < best_k, we update best_k.
                    # We should continue searching to see if we can go lower.
                    # So we don't return immediately.
                
                # Backtrack saturation
                for u in touched:
                    saturation[u].discard(c)
                colors[v] = -1
            
            return False

        # Initial setup
        # Use greedy to get initial best_k
        # The function takes initial_k.
        
        # Run search
        search(0, 0)
        
        if best_colors:
            return best_colors, best_k
        else:
            return None, initial_k

    def solve_instance(edges, adj, time_budget):
        m = len(edges)
        if m == 0:
            return []
        incompatible = build_incompatible(edges, adj)
        if m == 0:
            return []

        rng = random.Random(7)
        base_order = list(range(m))
        degree = [len(incompatible[i]) for i in range(m)]

        # 1. Get initial coloring with DSATUR
        colors_init = dsatur_greedy(incompatible, m, base_order)
        best_k = count_colors(colors_init)
        best_colors = colors_init[:]

        # 2. Try to improve with Exact B&B
        # Allocate 80% of time to B&B
        bb_time = time_budget * 0.8
        res_colors, res_k = solve_exact(incompatible, m, bb_time, best_k, rng)
        
        if res_colors and res_k < best_k:
            best_k = res_k
            best_colors = res_colors

        # 3. If time remains, do a quick local search on the result
        remaining = time_budget - (time.time() - (start_time - time_budget)) # This logic is tricky with nested time. 
        # Better: track time in solve_instance
        t_start = time.time()
        # We already used some time in solve_exact.
        # Let's just check if we have 2s left for LS.
        if time.time() - t_start < time_budget - 2.0:
            # Implement a simple LS here or reuse previous? 
            # To keep code small, we skip complex LS and rely on B&B.
            # Or just do a few random restarts of DSATUR if B&B didn't finish.
            pass

        # Renumber colors compactly 0..K-1
        remap = {}
        nxt = 0
        out = [0] * m
        for i in range(m):
            c = best_colors[i]
            if c not in remap:
                remap[c] = nxt
                nxt += 1
            out[i] = remap[c]
        return out

    out = []
    for idx, spec in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        remaining = deadline - time.time()
        if remaining <= 2:
            # Fallback: quick DSATUR
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            incompatible = build_incompatible(edges, adj)
            m = len(edges)
            colors = dsatur_greedy(incompatible, m, list(range(m))) if m else []
            out.append(colors)
            continue

        # Allocate more time to later (harder) instances
        if idx == 0:
            time_budget = remaining * 0.30
        elif idx == 1:
            time_budget = remaining * 0.35
        else:
            time_budget = remaining * 0.55

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        colors = solve_instance(edges, adj, time_budget)
        out.append(colors)

    return out
