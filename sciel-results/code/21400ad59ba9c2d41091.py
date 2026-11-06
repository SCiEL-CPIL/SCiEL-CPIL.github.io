import time
import sys
from random import Random
from collections import deque

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

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

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

    def solve_instance(n, d, seed, time_budget, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict graph
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
            
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)
                        
        def dsatur_coloring(rng):
            color = [-1] * m
            uncolored = list(range(m))
            
            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    neigh_colors = set()
                    for u in neigh[v]:
                        if color[u] != -1:
                            neigh_colors.add(color[u])
                    sat = len(neigh_colors)
                    deg = len(neigh[v])
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_v = v
                
                if best_v == -1:
                    break
                    
                forbidden = set()
                for u in neigh[best_v]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                
                c = 0
                while c in forbidden:
                    c += 1
                color[best_v] = c
                uncolored.remove(best_v)
            
            return color

        def try_reduce_color(color, K, rng, deadline):
            target_vertices = [i for i in range(m) if color[i] == K-1]
            if not target_vertices:
                return color, True
            
            target_vertices.sort(key=lambda x: len(neigh[x]))
            
            def backtrack(idx, current_color):
                if time.time() > deadline:
                    return None
                
                if idx == len(target_vertices):
                    return current_color[:]
                
                v = target_vertices[idx]
                colors_to_try = list(range(K-1))
                rng.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    valid = True
                    for u in neigh[v]:
                        if current_color[u] == c:
                            valid = False
                            break
                    
                    if valid:
                        current_color[v] = c
                        result = backtrack(idx + 1, current_color)
                        if result is not None:
                            return result
                        current_color[v] = K-1
            
            temp_color = color[:]
            result = backtrack(0, temp_color)
            if result:
                return result, True
            return color, False

        def random_greedy_coloring(rng):
            color = [-1] * m
            order = list(range(m))
            rng.shuffle(order)
            for v in order:
                forbidden = set()
                for u in neigh[v]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                c = 0
                while c in forbidden:
                    c += 1
                color[v] = c
            return color

        rng = Random(seed * 12345 + 7)
        
        best_color = dsatur_coloring(rng)
        best_cost = max(best_color) + 1
        
        while time.time() < deadline - 0.5:
            new_color, success = try_reduce_color(best_color, best_cost, rng, deadline - 0.5)
            
            if success:
                best_color = new_color
                best_cost -= 1
            else:
                new_col = random_greedy_coloring(rng)
                new_K = max(new_col) + 1
                
                if new_K < best_cost:
                    best_color = new_col
                    best_cost = new_K
                else:
                    if new_K == best_cost:
                        test_color, test_success = try_reduce_color(new_col, new_K, rng, deadline - 0.5)
                        if test_success:
                            best_color = test_color
                            best_cost -= 1
        
        return compress(best_color)

    try:
        start = time.time()
        deadline = start + 145.0
        out = []
        # Allocate time: 20s, 30s, 90s
        budgets = [20.0, 30.0, 90.0]
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = min(budgets[idx], remaining - 0.5)
            share = max(share, 0.5)
            out.append(solve_instance(n, d, seed, share, deadline))
        return out
    except Exception:
        return fallback()
