import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        # Precompute adjacency and distance-2 neighbors for conflict checking
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # For each edge, precompute the set of vertices that conflict with it
        # An edge (u,v) conflicts with another edge (x,y) if:
        # 1. They share a vertex (x==u, x==v, y==u, or y==v)
        # 2. They are joined by an edge (u-x, u-y, v-x, or v-y is an edge)
        # So for edge (u,v), the "forbidden" vertices are:
        # - u, v (shared vertex)
        # - neighbors of u (except v)
        # - neighbors of v (except u)
        # If another edge has any endpoint in this forbidden set, it conflicts.
        
        edge_vertex_conflicts = []
        for u, v in edges:
            forbidden = set()
            forbidden.add(u)
            forbidden.add(v)
            for w in adj[u]:
                if w != v:
                    forbidden.add(w)
            for w in adj[v]:
                if w != u:
                    forbidden.add(w)
            edge_vertex_conflicts.append(forbidden)
        
        # Initial coloring: greedy with random tie-breaking
        rng = random.Random(seed + 12345)
        colors = [0] * m
        color_used = [0] * m  # How many edges use each color
        max_color = 0
        
        for i in range(m):
            # Find a valid color for edge i
            # A color c is invalid if any edge with color c conflicts with edge i
            taken = set()
            for j in range(i):
                c = colors[j]
                # Check if edge j conflicts with edge i
                uj, vj = edges[j]
                ui, vi = edges[i]
                # Shared vertex?
                if uj == ui or uj == vi or vj == ui or vj == vi:
                    taken.add(c)
                else:
                    # Distance 2? Check if any endpoint of (ui,vi) is adjacent to any endpoint of (uj,vj)
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        taken.add(c)
            
            # Assign smallest available color
            c = 0
            while c in taken:
                c += 1
            colors[i] = c
            color_used[c] += 1
            max_color = max(max_color, c)
        
        # Simulated annealing
        def energy(cols):
            # Count conflicts: for each pair of edges with same color, check if they conflict
            # This is O(m^2) which is too slow. Instead, for each edge, check conflicts with same-color edges.
            # Better: for each color, for each pair of edges with that color, check conflict.
            # Even better: for each edge i, for each edge j < i with same color, check conflict.
            # Let's use a faster method: for each color, group edges, then for each pair in group, check conflict.
            # But m is small (<= 240), so O(m^2) per evaluation is ~57600 ops, which is fine.
            # Actually, let's precompute a conflict matrix.
            pass
        
        # Precompute conflict matrix: conflict[i][j] = 1 if edges i and j conflict
        conflict = [[0] * m for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                # Shared vertex?
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i][j] = 1
                    conflict[j][i] = 1
                else:
                    # Distance 2?
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict[i][j] = 1
                        conflict[j][i] = 1
        
        def compute_energy(cols):
            # Energy = number of conflicting pairs with same color + penalty for number of colors used
            # We want to minimize both conflicts and number of colors.
            # Let's use: energy = conflicts * 1000 + num_colors
            # This prioritizes validity over color count.
            conflicts = 0
            for i in range(m):
                ci = cols[i]
                for j in range(i + 1, m):
                    if cols[j] == ci and conflict[i][j]:
                        conflicts += 1
            num_colors = max(cols) + 1 if m > 0 else 0
            return conflicts * 1000 + num_colors
        
        current_energy = compute_energy(colors)
        best_colors = colors[:]
        best_energy = current_energy
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9999
        
        while time.time() < time_limit:
            if T < T_min:
                break
            
            # Pick a random edge and change its color
            i = rng.randrange(m)
            old_c = colors[i]
            # Try a random new color in [0, max_color]
            new_c = rng.randint(0, max_color)
            if new_c == old_c:
                continue
            
            colors[i] = new_c
            new_energy = compute_energy(colors)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    # Reduce max_color if possible
                    # Actually, max_color is just an upper bound for new colors, we can keep it
            else:
                colors[i] = old_c
            
            T *= cooling
        
        # Return the best coloring found
        # Need to renumber colors to be 0..k-1
        color_map = {}
        next_c = 0
        final_colors = []
        for c in best_colors:
            if c not in color_map:
                color_map[c] = next_c
                next_c += 1
            final_colors.append(color_map[c])
        
        return final_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            # Return best effort for remaining instances
            # Fallback: greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colors = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colors.items() if set(f) & near}
                c = 0
                while c in taken:
                    c += 1
                colors[e] = c
            results.append([colors[e] for e in edges])
            continue
        
        time_limit = max(1.0, remaining - 1.0)  # Leave 1s for other instances
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)
    
    return results
