"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

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

    def build_conflict_graph(edges, edge_set):
        """Build conflict graph on edges. Two edges conflict if they share a vertex or are joined by an edge."""
        m = len(edges)
        conflict = [set() for _ in range(m)]
        
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, d = edges[j]
                # Check if share a vertex
                if a == c or a == d or b == c or b == d:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    # Check if joined by an edge
                    if (a, c) in edge_set or (a, d) in edge_set or (b, c) in edge_set or (b, d) in edge_set:
                        conflict[i].add(j)
                        conflict[j].add(i)
        
        return conflict

    def greedy_color(order, conflict, m):
        """Greedy coloring with given order."""
        color = [-1] * m
        color_edges = {}  # color -> list of edge indices
        max_color = -1
        
        for e in order:
            placed = False
            for c in range(max_color + 1):
                if c not in color_edges:
                    color_edges[c] = [e]
                    color[e] = c
                    placed = True
                    break
                # Check conflict with all edges in this color
                ok = True
                for other in color_edges[c]:
                    if other in conflict[e]:
                        ok = False
                        break
                if ok:
                    color_edges[c].append(e)
                    color[e] = c
                    placed = True
                    break
            if not placed:
                c = max_color + 1
                color_edges[c] = [e]
                color[e] = c
                max_color = c
        
        return color, max_color + 1

    out = []
    start_time = time.time()
    deadline = start_time + 140
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            while len(out) < 3:
                out.append([])
            break
        
        edges = instance(*spec)
        m = len(edges)
        edge_set = set(edges)
        
        # Build conflict graph
        conflict = build_conflict_graph(edges, edge_set)
        
        # Try multiple orderings
        best_color = None
        best_count = m + 1
        
        # Ordering 1: descending conflict degree
        degs = [len(conflict[i]) for i in range(m)]
        order1 = sorted(range(m), key=lambda i: -degs[i])
        color1, count1 = greedy_color(order1, conflict, m)
        if count1 < best_count:
            best_count = count1
            best_color = color1
        
        # Ordering 2: multiple random shuffles
        rng = Random(42)
        for trial in range(50):
            if time.time() > deadline:
                break
            order2 = list(range(m))
            rng.shuffle(order2)
            color2, count2 = greedy_color(order2, conflict, m)
            if count2 < best_count:
                best_count = count2
                best_color = color2
        
        # Ordering 3: ascending conflict degree
        order3 = sorted(range(m), key=lambda i: degs[i])
        color3, count3 = greedy_color(order3, conflict, m)
        if count3 < best_count:
            best_count = count3
            best_color = color3
        
        if best_color is None:
            best_color = color1
        
        out.append([best_color[e] for e in range(m)])
    
    while len(out) < 3:
        out.append([])
    
    return out
# EVOLVE-BLOCK-END
