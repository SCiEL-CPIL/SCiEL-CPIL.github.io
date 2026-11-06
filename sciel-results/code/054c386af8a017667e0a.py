import time
from random import Random

def construct():
    deadline = time.time() + 110  # safety margin
    
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
    
    def build_conflict_graph(edges, n):
        """Build conflict graph: vertex for each edge, edge if they conflict."""
        num_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        adj = [set() for _ in range(num_edges)]
        
        # Build adjacency of original graph
        orig_adj = [set() for _ in range(n)]
        for u, v in edges:
            orig_adj[u].add(v)
            orig_adj[v].add(u)
        
        for i in range(num_edges):
            u1, v1 = edges[i]
            for j in range(i + 1, num_edges):
                u2, v2 = edges[j]
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    adj[i].add(j)
                    adj[j].add(i)
                    continue
                # Check if they are joined by an edge
                if u2 in orig_adj[u1] or v2 in orig_adj[u1] or u2 in orig_adj[v1] or v2 in orig_adj[v1]:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return adj
    
    def dsatur_color(adj, time_limit):
        """DSATUR greedy coloring."""
        n = len(adj)
        if n == 0:
            return [], 0
        
        colors = [-1] * n
        color_count = 0
        
        # Precompute degrees
        degrees = [len(adj[i]) for i in range(n)]
        
        # Saturation: for each vertex, track which colors are in its neighborhood
        sat = [set() for _ in range(n)]
        
        uncolored = list(range(n))
        
        while uncolored:
            # Pick vertex with highest saturation, tie-break by degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                s = len(sat[v])
                d = degrees[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_v = v
                    best_sat = s
                    best_deg = d
            
            # Assign lowest available color
            available = 0
            while available in sat[best_v]:
                available += 1
            
            colors[best_v] = available
            if available == color_count:
                color_count += 1
            
            # Update saturation of neighbors
            for nb in adj[best_v]:
                sat[nb].add(available)
            
            uncolored.remove(best_v)
        
        return colors, color_count
    
    def try_reduce_colors(adj, colors, k, time_limit):
        """Try to reduce coloring from k to k-1 colors. Returns new colors if successful."""
        n = len(adj)
        # Find vertices with color k-1 (0-indexed, so the last color)
        last_color = k - 1
        vertices_with_last = [v for v in range(n) if colors[v] == last_color]
        
        if not vertices_with_last:
            return None
        
        # Try to reassign each vertex with last color to another color
        # Use backtracking
        new_colors = colors[:]
        
        def backtrack(idx):
            nonlocal new_colors
            if time.time() > time_limit:
                return False
            if idx == len(vertices_with_last):
                return True
            v = vertices_with_last[idx]
            # Try colors 0 to k-2
            for c in range(k - 1):
                # Check if c is available for v
                conflict = False
                for nb in adj[v]:
                    if new_colors[nb] == c:
                        conflict = True
                        break
                if not conflict:
                    new_colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    new_colors[v] = last_color  # restore
            return False
        
        if backtrack(0):
            return new_colors
        return None
    
    def color_graph(adj, time_limit):
        """Color the conflict graph, trying to minimize colors."""
        n = len(adj)
        if n == 0:
            return []
        
        # Initial DSATUR coloring
        colors, k = dsatur_color(adj, time_limit)
        
        # Try to reduce colors
        while k > 1 and time.time() < time_limit:
            new_colors = try_reduce_colors(adj, colors, k, time_limit)
            if new_colors is not None:
                colors = new_colors
                k -= 1
            else:
                break
        
        return colors
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 10:
            # Not enough time, use simple greedy
            edges = instance(*spec)
            n = spec[0]
            adj_orig = [set() for _ in range(n)]
            for u, v in edges:
                adj_orig[u].add(v)
                adj_orig[v].add(u)
            colour = {}
            for e in edges:
                near = set(e)
                for x in e:
                    near.update(adj_orig[x])
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        adj = build_conflict_graph(edges, n)
        colors = color_graph(adj, deadline - 5)
        out.append(colors)
    
    return out
