import time
import math
from random import Random

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build original adjacency
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build conflict graph
        # conflict[i] is a set of j != i that conflict with i
        # Conflict if share a vertex or are at distance 2
        conflict = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                # Share vertex?
                if a == c or a == dd or b == c or b == dd:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    # Joined by an edge?
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        conflict[i].add(j)
                        conflict[j].add(i)

        conflict_list = [list(s) for s in conflict]
        
        # Precompute neighbor bits for faster checks? 
        # m is small enough for sets/lists.
        
        def dsatur_color(rng, time_limit):
            colors = [-1] * m
            uncolored = list(range(m))
            # For efficiency, maintain a list of uncolored vertices
            # and their saturation degrees.
            
            # We will use a simple loop to find the best vertex each time.
            # Since m is small (< 300), O(m^2) per step is acceptable if steps are not too many.
            # But m steps * m^2 = m^3 = 27M operations, which is fine in Python if optimized.
            
            # To speed up DSATUR, we can maintain a priority queue or just scan.
            # Scanning is simple and m is small.
            
            while uncolored:
                if time.time() > time_limit:
                    break
                
                best_v = -1
                best_sat = -1
                best_deg = -1
                
                for v in uncolored:
                    # Compute saturation
                    neighbor_colors = set()
                    for u in conflict_list[v]:
                        if colors[u] != -1:
                            neighbor_colors.add(colors[u])
                    sat = len(neighbor_colors)
                    
                    # Degree in uncolored subgraph
                    deg = 0
                    for u in conflict_list[v]:
                        if colors[u] == -1:
                            deg += 1
                            
                    # Tie-breaking: random
                    if sat > best_sat or (sat == best_sat and (deg > best_deg or (deg == best_deg and rng.random() < 0.5))):
                        best_sat = sat
                        best_deg = deg
                        best_v = v
                        
                if best_v == -1:
                    break
                    
                # Assign smallest available color
                used = set()
                for u in conflict_list[best_v]:
                    if colors[u] != -1:
                        used.add(colors[u])
                
                c = 0
                while c in used:
                    c += 1
                    
                colors[best_v] = c
                uncolored.remove(best_v)
            
            # If time ran out, assign new colors to remaining
            if -1 in colors:
                max_c = max(colors) if colors else -1
                for i in range(m):
                    if colors[i] == -1:
                        max_c += 1
                        colors[i] = max_c
                        
            return colors

        def local_search_improve(colors, time_limit):
            """
            Try to reduce the number of colors by moving vertices to lower color indices.
            """
            colors = colors[:]
            max_c = max(colors) if colors else 0
            
            # Build color classes
            color_classes = {}
            for i, c in enumerate(colors):
                if c not in color_classes:
                    color_classes[c] = set()
                color_classes[c].add(i)
                
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                # Try to move each vertex to the lowest possible color
                for i in range(m):
                    if time.time() > time_limit:
                        break
                    current_c = colors[i]
                    # Try colors 0 to current_c - 1
                    for c in range(current_c):
                        # Check if i can be recolored to c
                        # i.e., no neighbor of i has color c
                        can_move = True
                        for u in conflict_list[i]:
                            if colors[u] == c:
                                can_move = False
                                break
                        if can_move:
                            # Move i to color c
                            colors[i] = c
                            color_classes[current_c].remove(i)
                            if c not in color_classes:
                                color_classes[c] = set()
                            color_classes[c].add(i)
                            improved = True
                            break
                            
            return compress(colors)

        rng = Random(seed * 7919 + 42)
        best_coloring = None
        best_num_colors = float('inf')
        
        start_time = time.time()
        
        # Run multiple DSATUR instances with different random seeds for tie-breaking
        # and then apply local search.
        # Budget allocation:
        # We have ~150s total for 3 instances. ~50s per instance.
        # Let's use 20s for DSATUR restarts and 30s for local search on the best one.
        
        dsatur_deadline = start_time + 15.0
        ls_deadline = start_time + 45.0
        
        num_attempts = 10
        time_per_attempt = (dsatur_deadline - start_time) / num_attempts
        
        for i in range(num_attempts):
            if time.time() > dsatur_deadline:
                break
            attempt_rng = Random(rng.random())
            attempt_deadline = min(dsatur_deadline, time.time() + time_per_attempt)
            coloring = dsatur_color(attempt_rng, attempt_deadline)
            
            if coloring is None or -1 in coloring:
                continue
                
            # Compress and count
            compressed = compress(coloring)
            num_colors = len(set(compressed)) if compressed else 0
            
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = compressed[:]
                
        # Apply local search to the best coloring found
        if best_coloring is not None:
            # We can run local search multiple times or just once with a long time limit
            # Since local search is deterministic given the input, running it once with a long limit is fine.
            # But maybe we can try to "shake" it?
            # For now, simple local search.
            best_coloring = local_search_improve(best_coloring, ls_deadline)
            # Recompress after local search
            best_coloring = compress(best_coloring)
            
        if best_coloring is None:
            # Fallback: simple greedy
            active = set(range(m))
            colors = [-1] * m
            c = 0
            while active:
                v = next(iter(active))
                colors[v] = c
                active.remove(v)
                for u in conflict_list[v]:
                    if u in active:
                        active.remove(u)
                c += 1
            best_coloring = compress(colors)
            
        return best_coloring

    try:
        start = time.time()
        total_budget = 140.0
        per_instance = total_budget / len(specs)

        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = start + total_budget - time.time()
            share = min(per_instance, remaining / (len(specs) - idx))
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))

        return out
    except Exception:
        # Fallback: simple greedy
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            if not edges:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colour[e] = c
            out.append([colour[e] for e in edges])
        return out
