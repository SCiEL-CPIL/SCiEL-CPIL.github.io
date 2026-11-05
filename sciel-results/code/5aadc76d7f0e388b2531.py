import time
from random import Random

def construct():
    deadline = time.time() + 140
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        # Generate graph
        # Increase attempts to ensure we get a graph close to max degree if possible, 
        # but the specific graph structure is fixed by the seed.
        # The original code used 4*n*d attempts. We keep this to ensure same graph.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict_graph(edges, n):
        m = len(edges)
        if m == 0:
            return []
        
        # Map vertices to incident edges
        incident = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            incident[u].append(i)
            incident[v].append(i)
            
        # Adjacency list for vertices (to check "joined by an edge")
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        conflict_graph = [[] for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                
                # Check if they share a vertex
                if u1 in (u2, v2) or v1 in (u2, v2):
                    conflict_graph[i].append(j)
                    conflict_graph[j].append(i)
                    continue
                
                # Check if they are joined by an edge
                # (u1, u2) in E?
                if u2 in adj[u1]:
                    conflict_graph[i].append(j)
                    conflict_graph[j].append(i)
                    continue
                if v2 in adj[u1]:
                    conflict_graph[i].append(j)
                    conflict_graph[j].append(i)
                    continue
                if u2 in adj[v1]:
                    conflict_graph[i].append(j)
                    conflict_graph[j].append(i)
                    continue
                if v2 in adj[v1]:
                    conflict_graph[i].append(j)
                    conflict_graph[j].append(i)
                    continue
                    
        return conflict_graph

    def greedy_color(conflict_graph, order, rng=None):
        m = len(conflict_graph)
        color = [-1] * m
        # For each color, keep a set of vertices (edge indices) having that color
        # And for each vertex, keep a set of colors used by its neighbors
        # To speed up, we can just check neighbors when assigning.
        
        # To make it faster, we maintain `used_colors` for each vertex?
        # Or just check neighbors.
        # m is small (~200-300). Checking neighbors is O(deg).
        
        for v in order:
            # Find smallest color not used by neighbors
            used = set()
            for nb in conflict_graph[v]:
                if color[nb] != -1:
                    used.add(color[nb])
            
            c = 0
            while c in used:
                c += 1
            color[v] = c
            
        return color

    def try_merge(color, conflict_graph):
        """
        Try to merge color classes.
        Returns new color list and num colors.
        """
        # Group edges by color
        groups = {}
        for i, c in enumerate(color):
            if c not in groups:
                groups[c] = []
            groups[c].append(i)
            
        colors = list(groups.keys())
        
        # We try to merge pairs of colors if the union of their edge sets is an independent set in conflict_graph.
        # Two edges in the union conflict if they are in the same original color (impossible, they were independent) 
        # or if one is from color A and one from color B and they conflict.
        # So we just need to check cross-conflicts.
        
        changed = True
        while changed:
            changed = False
            colors = list(groups.keys())
            for idx_a in range(len(colors)):
                ca = colors[idx_a]
                if ca not in groups: continue
                group_a = groups[ca]
                
                for idx_b in range(idx_a + 1, len(colors)):
                    cb = colors[idx_b]
                    if cb not in groups: continue
                    group_b = groups[cb]
                    
                    # Check if any edge in A conflicts with any edge in B
                    mergeable = True
                    for e_a in group_a:
                        for e_b in group_b:
                            if e_b in conflict_graph[e_a]:
                                mergeable = False
                                break
                        if not mergeable:
                            break
                            
                    if mergeable:
                        # Merge B into A
                        for e in group_b:
                            color[e] = ca
                        del groups[cb]
                        changed = True
                        break
                if changed:
                    break
                    
        # Compress colors to 0..k-1
        if not groups:
            return color, 0
        mapping = {}
        new_color = [0] * len(color)
        next_c = 0
        for c in sorted(groups.keys()):
            mapping[c] = next_c
            next_c += 1
        for i in range(len(color)):
            new_color[i] = mapping[color[i]]
        return new_color, next_c

    def local_search(color, conflict_graph, rng, time_limit):
        """
        Simple local search: pick a random edge, try to move it to a lower color if possible, or swap.
        """
        m = len(color)
        best_color = color[:]
        best_nc = len(set(color))
        
        while time.time() < time_limit:
            v = rng.randrange(m)
            c_old = color[v]
            
            # Try to move to any color < c_old
            for c_new in range(c_old):
                # Check if v can take c_new
                valid = True
                for nb in conflict_graph[v]:
                    if color[nb] == c_new:
                        valid = False
                        break
                if valid:
                    color[v] = c_new
                    # Try to merge
                    new_c, nc = try_merge(color, conflict_graph)
                    if nc < best_nc:
                        best_nc = nc
                        best_color = new_c[:]
                        color = new_c
                    break
            
            # If no improvement, maybe try a random valid move to escape local min?
            # Or just stop if no progress for a while? 
            # For now, just do simple moves.
            
        return best_color, best_nc

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() > deadline - 5:
            # Fill remaining with empty or best effort?
            # If we run out of time, we must return something.
            # The task says "invalid result... scores 0". 
            # So we must ensure we return a valid coloring for all 3.
            # If we are close to deadline, we do a very fast greedy for remaining.
            pass
            
        edges = instance(*spec)
        n = spec[0]
        m = len(edges)
        
        if m == 0:
            out.append([])
            continue
            
        conflict_graph = build_conflict_graph(edges, n)
        
        best_color = None
        best_nc = float('inf')
        
        # Calculate degrees in conflict graph for ordering
        conf_degrees = [len(nei) for nei in conflict_graph]
        
        # Base order: highest conflict degree first
        base_order = sorted(range(m), key=lambda x: -conf_degrees[x])
        
        # Run several restarts with perturbed orders
        rng = Random(42)
        start_time = time.time()
        # Budget per instance: roughly 40s
        instance_budget = 40
        
        # 1. Deterministic Greedy
        color = greedy_color(conflict_graph, base_order)
        color, nc = try_merge(color, conflict_graph)
        if nc < best_nc:
            best_nc = nc
            best_color = color[:]
            
        # 2. Randomized Greedy + Local Search
        remaining_time = instance_budget - (time.time() - start_time)
        # Do a few random restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Perturb order: swap some random elements in base_order
            order = base_order[:]
            for _ in range(10):
                i, j = rng.randrange(m), rng.randrange(m)
                order[i], order[j] = order[j], order[i]
                
            color = greedy_color(conflict_graph, order)
            color, nc = try_merge(color, conflict_graph)
            
            if nc < best_nc:
                best_nc = nc
                best_color = color[:]
                
            # Small local search budget
            ls_time = min(1.0, remaining_time / 10)
            if ls_time > 0.1:
                ls_color, ls_nc = local_search(color, conflict_graph, rng, time.time() + ls_time)
                if ls_nc < best_nc:
                    best_nc = ls_nc
                    best_color = ls_color[:]
                    
        # If we still have time and best_nc is high, do more local search on best
        if time.time() < deadline - 10 and best_color is not None:
             ls_color, ls_nc = local_search(best_color, conflict_graph, rng, time.time() + 10)
             if ls_nc < best_nc:
                 best_nc = ls_nc
                 best_color = ls_color[:]

        out.append(best_color)
        
    # Ensure we return 3 lists
    while len(out) < 3:
        # Fallback: generate a trivial valid coloring (each edge unique color)
        # This is valid but high score. Better than crashing.
        spec = specs[len(out)]
        edges = instance(*spec)
        out.append(list(range(len(edges))))
        
    return out
