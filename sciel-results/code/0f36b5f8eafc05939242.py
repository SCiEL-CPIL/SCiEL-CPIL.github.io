import time
import sys
import math
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

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict graph using adjacency sets for O(1) lookups
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
            
        # For each edge i=(a,b), conflicts with edge j=(c,d) if:
        # - shares a vertex: a==c, a==d, b==c, or b==d
        # - or one endpoint of j is adjacent to an endpoint of i (but not the shared one)
        neigh = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].add(j)
                    neigh[j].add(i)
                else:
                    # Check if any endpoint of j is adjacent to any endpoint of i
                    if (c in adj_orig[a] or dd in adj_orig[a] or 
                        c in adj_orig[b] or dd in adj_orig[b]):
                        neigh[i].add(j)
                        neigh[j].add(i)
                        
        # Convert to lists for faster iteration
        neigh_list = [list(s) for s in neigh]
        deg = [len(neigh_list[i]) for i in range(m)]
        
        # DSATUR greedy coloring
        def dsatur_coloring(order=None):
            if order is None:
                order = list(range(m))
            color = [-1] * m
            order = sorted(order, key=lambda x: deg[x], reverse=True)
            
            for v in order:
                forbidden = set()
                for u in neigh_list[v]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                c = 0
                while c in forbidden:
                    c += 1
                color[v] = c
            
            return color, max(color) + 1
        
        # Try multiple random orderings to get good initial coloring
        rng = Random(seed * 7919 + 13)
        best_color = None
        best_cost = m
        
        # Deterministic DSATUR
        c0, k0 = dsatur_coloring()
        best_color = c0
        best_cost = k0
        
        # Random restarts
        for restart in range(30):
            if time.time() > deadline - 10.0:
                break
            perm = list(range(m))
            rng.shuffle(perm)
            c, k = dsatur_coloring(perm)
            if k < best_cost:
                best_color = c
                best_cost = k
        
        # Local search: try to merge color classes
        def try_merge_colors(color, K):
            """Try to merge color classes to reduce K. Returns (new_color, new_K) or None."""
            if time.time() > deadline - 5.0:
                return None
            
            # Build color class membership
            color_classes = [[] for _ in range(K)]
            for i in range(m):
                color_classes[color[i]].append(i)
            
            # For each pair of color classes, check if they can be merged
            # Two color classes can be merged if no edge in one conflicts with an edge in the other
            # i.e., for all e1 in class_a and e2 in class_b, e1 and e2 are not in conflict
            
            best_merge = None
            best_size = 0
            
            for a in range(K):
                if time.time() > deadline - 5.0:
                    return None
                for b in range(a + 1, K):
                    # Check if classes a and b can be merged
                    can_merge = True
                    for e1 in color_classes[a]:
                        for e2 in color_classes[b]:
                            if e2 in neigh[e1]:
                                can_merge = False
                                break
                        if not can_merge:
                            break
                    if can_merge:
                        # Prefer merging larger classes
                        size = len(color_classes[a]) + len(color_classes[b])
                        if size > best_size:
                            best_size = size
                            best_merge = (a, b)
            
            if best_merge is None:
                return None
            
            a, b = best_merge
            new_color = color[:]
            for i in color_classes[b]:
                new_color[i] = a
            return new_color, K - 1
        
        def local_search_reduce(color, K):
            """Iteratively try to merge colors until no more merges possible."""
            current_color = color[:]
            current_K = K
            while True:
                if time.time() > deadline - 5.0:
                    break
                result = try_merge_colors(current_color, current_K)
                if result is None:
                    break
                current_color, current_K = result
            return current_color, current_K
        
        # Apply local search reduction
        current_color, current_K = local_search_reduce(best_color, best_cost)
        
        # Simulated annealing for further reduction
        def simulated_annealing(color, K, iterations=5000):
            """Try to reduce colors using simulated annealing."""
            if time.time() > deadline - 8.0:
                return color, K
            
            current_color = color[:]
            current_K = K
            best_color = current_color[:]
            best_K = current_K
            
            temperature = 1.0
            cooling = 0.9995
            
            for it in range(iterations):
                if time.time() > deadline - 8.0:
                    break
                
                # Random move: pick a random edge and try to recolor it
                v = rng.randint(0, m - 1)
                old_c = current_color[v]
                
                # Try a new color
                if current_K > 1:
                    new_c = rng.randint(0, current_K - 1)
                    if new_c == old_c:
                        continue
                    
                    # Check if new color is valid for v
                    valid = True
                    for u in neigh_list[v]:
                        if current_color[u] == new_c:
                            valid = False
                            break
                    
                    if valid:
                        # Accept the move (always accept valid moves)
                        current_color[v] = new_c
                        
                        # Check if we can reduce K
                        # If no edge uses color current_K-1, we can reduce
                        uses_last = any(current_color[i] == current_K - 1 for i in range(m))
                        if not uses_last and current_K > 1:
                            # Relabel colors to compact
                            for i in range(m):
                                if current_color[i] > current_K - 2:
                                    current_color[i] = current_K - 2
                            current_K -= 1
                            if current_K < best_K:
                                best_K = current_K
                                best_color = current_color[:]
                    else:
                        # Invalid move: reject
                        pass
                else:
                    continue
                
                temperature *= cooling
            
            return best_color, best_K
        
        # Run simulated annealing with multiple restarts
        for sa_restart in range(5):
            if time.time() > deadline - 10.0:
                break
            # Perturb current solution
            perturbed_color = current_color[:]
            perturbed_K = current_K
            for _ in range(m // 10 + 1):
                v = rng.randint(0, m - 1)
                if perturbed_K > 1:
                    new_c = rng.randint(0, perturbed_K - 1)
                    # Check validity
                    valid = True
                    for u in neigh_list[v]:
                        if perturbed_color[u] == new_c:
                            valid = False
                            break
                    if valid:
                        perturbed_color[v] = new_c
            
            # Run SA from perturbed state
            sa_color, sa_K = simulated_annealing(perturbed_color, perturbed_K, iterations=3000)
            if sa_K < current_K:
                current_color = sa_color
                current_K = sa_K
            
            # Also try merging after SA
            current_color, current_K = local_search_reduce(current_color, current_K)
        
        return compress(current_color)

    try:
        start = time.time()
        deadline = start + 140.0  # Safety margin before 150s kill
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            if remaining < 5.0:
                # Not enough time, use fallback for remaining instances
                for rem_idx in range(idx, len(specs)):
                    edges = instance(*specs[rem_idx])
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
                break
            # Allocate time: give more to larger instances
            weights = [40, 60, 80]  # proportional to n
            total_weight = sum(weights[idx:])
            share = remaining * weights[idx] / total_weight
            share = min(share, 50.0)  # Cap per instance
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
