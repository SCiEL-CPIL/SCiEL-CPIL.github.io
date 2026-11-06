"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    def instance(n, d, seed):
        from random import Random
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

    def build_conflict_graph(n, edges):
        m = len(edges)
        if m == 0:
            return []
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        adj_g = defaultdict(set)
        for u, v in edges:
            adj_g[u].add(v)
            adj_g[v].add(u)
            
        h_adj = [set() for _ in range(m)]
        
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                if (u == x or u == y or v == x or v == y):
                    h_adj[i].add(j)
                    h_adj[j].add(i)
                else:
                    if (u, x) in edge_to_idx or (u, y) in edge_to_idx or \
                       (v, x) in edge_to_idx or (v, y) in edge_to_idx:
                        h_adj[i].add(j)
                        h_adj[j].add(i)
        
        return h_adj

    def dsatur_coloring(h_adj, rng):
        m = len(h_adj)
        if m == 0:
            return []
        
        colors = [-1] * m
        neighbor_colors = [set() for _ in range(m)]
        
        for step in range(m):
            best_v = -1
            best_sat = -1
            best_deg = -1
            candidates = []
            
            for v in range(m):
                if colors[v] != -1:
                    continue
                sat = len(neighbor_colors[v])
                deg = len(h_adj[v])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    candidates = [v]
                elif sat == best_sat and deg == best_deg:
                    candidates.append(v)
            
            if best_v == -1 and not candidates:
                break
                
            if len(candidates) > 1:
                best_v = rng.choice(candidates)
            else:
                best_v = candidates[0]
                
            c = 0
            while c in neighbor_colors[best_v]:
                c += 1
            
            colors[best_v] = c
            
            for u in h_adj[best_v]:
                if colors[u] == -1:
                    neighbor_colors[u].add(c)
                    
        return colors

    def merge_colors(colors, h_adj):
        """
        Greedily merge color classes if no conflicts exist between them.
        """
        m = len(colors)
        if m == 0:
            return colors
        
        # Find max color
        max_c = max(colors)
        
        # Group vertices by color
        color_classes = defaultdict(list)
        for i, c in enumerate(colors):
            color_classes[c].append(i)
            
        merged = True
        # We iterate until no more merges are possible
        # To be efficient, we can just check all pairs of current color classes
        while merged:
            merged = False
            current_colors = sorted([c for c in color_classes if color_classes[c]])
            
            for i_idx, c1 in enumerate(current_colors):
                if not color_classes[c1]:
                    continue
                for c2 in current_colors[i_idx+1:]:
                    if not color_classes[c2]:
                        continue
                    
                    # Check if any vertex in c1 is adjacent to any vertex in c2 in H
                    conflict = False
                    for v in color_classes[c1]:
                        if any(u in h_adj[v] for u in color_classes[c2]):
                            conflict = True
                            break
                    if not conflict:
                        # Merge c2 into c1
                        for v in color_classes[c2]:
                            colors[v] = c1
                        color_classes[c1].extend(color_classes[c2])
                        color_classes[c2] = []
                        merged = True
                        # Break inner loop to restart pair checking? 
                        # Or just continue, but the list of colors changed.
                        # Since we are modifying the structure, it's safer to break out
                        # and restart the while loop.
                        break
                if merged:
                    break
                    
        # Remap colors to be 0..K-1
        if not colors:
            return []
        
        unique_colors = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        return [mapping[c] for c in colors]

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        if not edges:
            return []
        
        h_adj = build_conflict_graph(n, edges)
        
        best_colors = None
        best_count = float('inf')
        
        # Run multiple randomized DSATUR instances
        # Time budget per instance is small, so we can run many
        from random import Random
        rng = Random(seed + 1000)
        
        # Estimate time per DSATUR run. It's very fast.
        # Let's run 20 restarts.
        for _ in range(20):
            colors = dsatur_coloring(h_adj, rng)
            colors = merge_colors(colors, h_adj)
            count = len(set(colors)) if colors else 0
            if count < best_count:
                best_count = count
                best_colors = colors[:]
                
            if time.time() > DEADLINE:
                break
                
        if best_colors is None:
            # Fallback to simple greedy if DSATUR failed (shouldn't happen)
            best_colors = [0] * len(edges)
            
        return best_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE:
            # If we run out of time, we must still return valid results.
            # We can use the previous best or a very fast greedy.
            # But we should have finished.
            pass
            
        colors = solve_instance(*spec)
        out.append(colors)
        
    return out
# EVOLVE-BLOCK-END
