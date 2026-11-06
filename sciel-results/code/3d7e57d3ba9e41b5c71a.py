import time
import math
import numpy as np
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

        # Build conflict graph: conflict[i] is a set of j != i that conflict with i
        # Conflict if share a vertex or are at distance 2 (joined by an edge)
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
                    # Joined by an edge? i.e., c in adj[a] or c in adj[b] or d in adj[a] or d in adj[b]
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        conflict[i].add(j)
                        conflict[j].add(i)

        # Convert to list for faster iteration
        conflict_list = [list(s) for s in conflict]
        # Also build a bitset for fast "is it a neighbor?" checks
        # Since m is small (max ~200), we can use Python ints as bitsets
        conflict_bits = [0] * m
        for i in range(m):
            bits = 0
            for j in conflict_list[i]:
                bits |= (1 << j)
            conflict_bits[i] = bits

        def dsatur_color(rng, time_limit):
            """
            DSATUR algorithm for coloring the conflict graph.
            Returns a list of colors for each edge (vertex in conflict graph).
            """
            colors = [-1] * m
            # color_count[c] = number of vertices colored with color c
            color_count = {}
            # For each vertex, the set of colors used by its neighbors
            # We'll maintain this dynamically
            # To speed up, we can precompute the list of neighbors
            # But DSATUR typically uses a priority queue. Here we use a simple loop.
            
            # To make it faster, we can use a list of uncolored vertices and pick the one with highest saturation degree.
            # Saturation degree = number of distinct colors in its neighborhood.
            # Tie-break by degree in the uncolored subgraph.
            
            uncolored = set(range(m))
            current_color = 0
            
            while uncolored:
                if time.time() > time_limit:
                    break
                
                # Find the vertex with highest saturation degree
                best_v = None
                best_sat = -1
                best_deg = -1
                
                for v in uncolored:
                    # Compute saturation: distinct colors in neighbors
                    neighbor_colors = set()
                    for u in conflict_list[v]:
                        if colors[u] != -1:
                            neighbor_colors.add(colors[u])
                    sat = len(neighbor_colors)
                    deg = 0
                    for u in conflict_list[v]:
                        if colors[u] == -1:
                            deg += 1
                    
                    # Random tie-break
                    if sat > best_sat or (sat == best_sat and (deg > best_deg or (deg == best_deg and rng.random() < 0.5))):
                        best_sat = sat
                        best_deg = deg
                        best_v = v
                
                if best_v is None:
                    break
                    
                # Assign the smallest available color
                available_colors = set(range(current_color + 1))  # We only consider colors up to current_color
                # Actually, we can use any color, but DSATUR typically reuses the smallest available.
                # Let's find the smallest color not in neighbor_colors
                used = set()
                for u in conflict_list[best_v]:
                    if colors[u] != -1:
                        used.add(colors[u])
                
                c = 0
                while c in used:
                    c += 1
                
                colors[best_v] = c
                uncolored.remove(best_v)
                if c > current_color:
                    current_color = c
            
            # If we broke early, assign remaining arbitrarily (with new colors) to ensure validity
            if -1 in colors:
                next_c = current_color + 1
                for i in range(m):
                    if colors[i] == -1:
                        colors[i] = next_c
                        next_c += 1
            
            return colors

        def local_search(coloring, rng, time_limit):
            """
            Try to reduce the number of colors by moving vertices to lower colors or swapping.
            """
            colors = coloring[:]
            # Count colors
            max_c = max(colors) if colors else 0
            # color_classes[c] = set of vertices with color c
            color_classes = {}
            for i, c in enumerate(colors):
                if c not in color_classes:
                    color_classes[c] = set()
                color_classes[c].add(i)
            
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                # Try to move a vertex to a lower color
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
                    if improved:
                        # Restart the loop? Or continue?
                        # Continue, but we might want to restart to catch cascading improvements.
                        # For simplicity, we break out of the inner loop and continue the outer.
                        pass
                
                # Try to swap colors between two vertices to free up a color?
                # This is more complex. Let's stick to simple moves for now.
                # Or try to recolor a vertex to a new color if it reduces the max? No, that increases.
                
                # Try to "merge" two color classes if possible?
                # That would require recoloring all vertices in one class to the other, which is hard.
            
            # Compress colors to 0..k-1
            return compress(colors)

        rng = Random(seed * 7919 + 42)
        
        best_coloring = None
        best_num_colors = float('inf')
        
        # Run multiple DSATUR instances
        num_attempts = 20
        # We can adjust num_attempts based on time, but let's start with a fixed number.
        # If time is tight, we can reduce.
        
        start_time = time.time()
        time_per_attempt = (deadline - start_time - 1.0) / num_attempts  # Leave 1s for local search
        
        for i in range(num_attempts):
            if time.time() > deadline - 1.0:
                break
            attempt_rng = Random(rng.random())
            attempt_deadline = min(deadline - 1.0, time.time() + time_per_attempt)
            coloring = dsatur_color(attempt_rng, attempt_deadline)
            
            if coloring is None or -1 in coloring:
                continue
                
            # Compress and count
            compressed = compress(coloring)
            num_colors = len(set(compressed)) if compressed else 0
            
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = compressed[:]
                
            # If we hit the theoretical bound, we can stop early?
            en_bound = int(5 * d * d / 4)
            if best_num_colors <= en_bound:
                break
        
        # Apply local search to the best coloring
        if best_coloring is not None:
            final_deadline = deadline - 0.5
            best_coloring = local_search(best_coloring, rng, final_deadline)
        
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
