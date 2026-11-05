"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    def build_conflict_graph(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
        n_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(n_edges)]
        
        for i, (u, v) in enumerate(edges):
            # Edges sharing vertex u or v
            for nb in adj[u]:
                if nb > u:  # edge (u, nb)
                    e = (u, nb)
                    if e in edge_to_idx:
                        j = edge_to_idx[e]
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            for nb in adj[v]:
                if nb > v:  # edge (v, nb)
                    e = (v, nb)
                    if e in edge_to_idx:
                        j = edge_to_idx[e]
                        conflicts[i].add(j)
                        conflicts[j].add(i)
                elif nb < v:
                    e = (nb, v)
                    if e in edge_to_idx:
                        j = edge_to_idx[e]
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            # Edges joined by an edge: (u,w) or (v,w) for w in adj[u] or adj[v]
            # Actually, we need: for edge (a,b), it conflicts with (c,d) if 
            # {a,b} & {c,d} != empty OR (a,c) in E OR (a,d) in E OR (b,c) in E OR (b,d) in E
            # The above handles shared vertices. Now handle "joined by an edge":
            for nb in adj[u]:
                if nb != v:
                    for nb2 in adj[nb]:
                        if nb2 != u and nb2 != v:
                            e2 = (min(nb, nb2), max(nb, nb2))
                            if e2 in edge_to_idx:
                                j = edge_to_idx[e2]
                                if j != i:
                                    conflicts[i].add(j)
                                    conflicts[j].add(i)
            for nb in adj[v]:
                if nb != u:
                    for nb2 in adj[nb]:
                        if nb2 != u and nb2 != v:
                            e2 = (min(nb, nb2), max(nb, nb2))
                            if e2 in edge_to_idx:
                                j = edge_to_idx[e2]
                                if j != i:
                                    conflicts[i].add(j)
                                    conflicts[j].add(i)
        
        return conflicts

    def greedy_color(conflicts, order):
        """Greedy colouring following the given order of edge indices."""
        n = len(conflicts)
        colour = [-1] * n
        used_colours = set()
        
        for idx in order:
            # Find smallest colour not used by any conflicting edge
            blocked = set()
            for j in conflicts[idx]:
                if colour[j] != -1:
                    blocked.add(colour[j])
            c = 0
            while c in blocked:
                c += 1
            colour[idx] = c
            used_colours.add(c)
        
        return colour, len(used_colours)

    def color_instance(n, d, seed):
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflicts = build_conflict_graph(edges, adj)
        n_edges = len(edges)
        
        best_colour = None
        best_count = n_edges  # worst case: all different colours
        
        # Try multiple orderings
        rng = random.Random(42)
        
        # Ordering 1: descending degree in conflict graph
        deg_order = sorted(range(n_edges), key=lambda i: len(conflicts[i]), reverse=True)
        colour, count = greedy_color(conflicts, deg_order)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
        
        # Ordering 2: ascending degree
        asc_order = sorted(range(n_edges), key=lambda i: len(conflicts[i]))
        colour, count = greedy_color(conflicts, asc_order)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
        
        # Try random orderings
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            order = list(range(n_edges))
            rng.shuffle(order)
            colour, count = greedy_color(conflicts, order)
            if count < best_count:
                best_count = count
                best_colour = colour[:]
        
        # Local search: try swapping colours to reduce count
        if best_colour is not None and time.time() < deadline - 5:
            # Try to recolor some edges to free up a colour
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                # Find the highest colour used
                max_c = max(best_colour)
                # Try to remove colour max_c by recoloring edges with that colour
                edges_with_max = [i for i in range(n_edges) if best_colour[i] == max_c]
                if not edges_with_max:
                    break
                # Try to recolor one edge from max_c to a lower colour
                improved = False
                for idx in edges_with_max:
                    blocked = set()
                    for j in conflicts[idx]:
                        if best_colour[j] != -1 and best_colour[j] != max_c:
                            blocked.add(best_colour[j])
                    # Find a colour < max_c not in blocked
                    for c in range(max_c):
                        if c not in blocked:
                            # Check if this is valid
                            valid = True
                            for j in conflicts[idx]:
                                if best_colour[j] == c:
                                    valid = False
                                    break
                            if valid:
                                best_colour[idx] = c
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    break
        
        return [best_colour[i] for i in range(n_edges)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 20:
            # Not enough time, use simple greedy for remaining
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        else:
            out.append(color_instance(*spec))
    
    return out
# EVOLVE-BLOCK-END
