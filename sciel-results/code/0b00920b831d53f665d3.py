"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
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

    def solve_instance(edges, time_limit):
        if not edges:
            return []
        n = max(u for u, v in edges) + 1
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        m = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        
        conflicts = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i+1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                else:
                    # Check if joined by an edge
                    if (min(u1, u2), max(u1, u2)) in edge_index or \
                       (min(u1, v2), max(u1, v2)) in edge_index or \
                       (min(v1, u2), max(v1, u2)) in edge_index or \
                       (min(v1, v2), max(v1, v2)) in edge_index:
                         conflicts[i].add(j)
                         conflicts[j].add(i)

        deg = [0]*n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        def greedy_color(order):
            color = [0] * m
            used_colors = set()
            for i in order:
                forbidden = set()
                for j in conflicts[i]:
                    # In greedy, we process in order, so previous ones are colored
                    # But conflicts are symmetric. We just check all neighbors.
                    # Since we assign colors sequentially, unassigned neighbors have color 0 initially?
                    # No, we should only check assigned neighbors.
                    # But in this implementation, `color` is initialized to 0.
                    # If we process i, and j is not processed yet, color[j] is 0.
                    # This is a bug in simple greedy if we don't track assignment.
                    # Let's fix greedy to track assigned.
                    pass
                
            # Corrected Greedy
            color = [-1] * m
            used_colors = set()
            for i in order:
                forbidden = set()
                for j in conflicts[i]:
                    if color[j] != -1:
                        forbidden.add(color[j])
                
                c = 0
                while c in forbidden:
                    c += 1
                color[i] = c
                used_colors.add(c)
            return color, max(used_colors) + 1 if used_colors else 0

        best_color = None
        best_k = float('inf')
        
        # Deterministic Orders
        order1 = list(range(m))
        c1, k1 = greedy_color(order1)
        if k1 < best_k:
            best_k = k1
            best_color = c1
            
        order2 = sorted(range(m), key=lambda i: deg[edges[i][0]] + deg[edges[i][1]], reverse=True)
        c2, k2 = greedy_color(order2)
        if k2 < best_k:
            best_k = k2
            best_color = c2

        # Random Orders
        rng = Random(42)
        for _ in range(50):
            if time.time() > time_limit:
                break
            order = list(range(m))
            rng.shuffle(order)
            c, k = greedy_color(order)
            if k < best_k:
                best_k = k
                best_color = c

        # Local Search
        current_color = best_color
        current_k = best_k
        
        def try_reduce(color, k):
             while k > 0:
                if time.time() > time_limit:
                    break
                target_color = k - 1
                edges_with_target = [i for i in range(m) if color[i] == target_color]
                
                if not edges_with_target:
                    k -= 1
                    continue
                
                assigned = {}
                def can_recolor(idx):
                    if idx == len(edges_with_target):
                        return True
                    e_idx = edges_with_target[idx]
                    forbidden = set()
                    for j in conflicts[e_idx]:
                        if j in assigned:
                            forbidden.add(assigned[j])
                        else:
                            # Only fixed colors < target_color matter
                            if color[j] < target_color:
                                forbidden.add(color[j])
                    
                    for c in range(target_color):
                        if c not in forbidden:
                            assigned[e_idx] = c
                            if can_recolor(idx + 1):
                                return True
                            del assigned[e_idx]
                    return False

                if can_recolor(0):
                    for i in edges_with_target:
                        color[i] = assigned[i]
                    k -= 1
                else:
                    break
             return color, k

        current_color, current_k = try_reduce(current_color, current_k)
        return current_color

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() > deadline:
             # Fallback
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
             continue

        edges = instance(*spec)
        colors = solve_instance(edges, deadline)
        out.append(colors)
        
    return out
# EVOLVE-BLOCK-END
