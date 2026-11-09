"""Strong edge colouring of three fixed bounded-degree graphs."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill

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

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_conflict(e1, e2, adj):
        """Check if two edges are in conflict for strong coloring."""
        # They share a vertex
        if e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]:
            return True
        # They are joined by an edge
        u1, v1 = e1
        u2, v2 = e2
        if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
            return True
        return False

    def greedy_color(edges, adj, order):
        """Greedy coloring with a given edge order."""
        color = {}
        for e in order:
            u, v = e
            # Neighbourhood: vertices in e and their neighbours
            near = {u, v} | adj[u] | adj[v]
            taken = set()
            for f, c in color.items():
                fu, fv = f
                if fu in near or fv in near:
                    taken.add(c)
            c = 0
            while c in taken:
                c += 1
            color[e] = c
        return [color[e] for e in edges], max(color.values()) + 1 if color else 0

    def local_search_improve(coloring, edges, adj, max_iters=500):
        """Try to reduce number of colors by reassigning edges."""
        n_colors = max(coloring) + 1 if coloring else 0
        # Build color classes
        classes = [[] for _ in range(n_colors)]
        for i, c in enumerate(coloring):
            classes[c].append(edges[i])
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for c in range(n_colors - 1, -1, -1):
                if not classes[c]:
                    continue
                # Try to move each edge in class c to a lower class
                for e in list(classes[c]):
                    u, v = e
                    near = {u, v} | adj[u] | adj[v]
                    for c2 in range(c):
                        # Check if e can go into class c2
                        conflict = False
                        for f in classes[c2]:
                            fu, fv = f
                            if fu in near or fv in near:
                                conflict = True
                                break
                        if not conflict:
                            classes[c].remove(e)
                            classes[c2].append(e)
                            improved = True
                            break
                if not classes[c]:
                    n_colors -= 1
        # Rebuild coloring
        new_coloring = [0] * len(edges)
        for c, cls in enumerate(classes):
            if c >= n_colors:
                break
            for e in cls:
                idx = edges.index(e)
                new_coloring[idx] = c
        return new_coloring, n_colors

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        m = len(edges)
        
        best_coloring = None
        best_num_colors = m  # upper bound
        
        # Try multiple orderings
        orders = []
        
        # Original order
        orders.append(list(range(m)))
        
        # Degree-based ordering (higher degree vertices first)
        deg = [len(adj[v]) for v in range(n)]
        order_deg = sorted(range(m), key=lambda i: (deg[edges[i][0]] + deg[edges[i][1]], edges[i]))
        orders.append(order_deg)
        
        # Random orderings
        rng = random.Random(seed + 999)
        for _ in range(20):
            order_rand = list(range(m))
            rng.shuffle(order_rand)
            orders.append(order_rand)
        
        for order in orders:
            if time.time() >= deadline:
                break
            ordered_edges = [edges[i] for i in order]
            coloring, nc = greedy_color(ordered_edges, adj, ordered_edges)
            # Map back to original edge order
            # coloring is in the order of ordered_edges
            # We need coloring in the order of edges
            # Actually greedy_color returns coloring for the given edge list
            # So coloring[i] is the color of ordered_edges[i]
            # We need to map back: for each original edge index, find its color
            
            # Build a map from edge to color
            edge_to_color = {}
            for i, e in enumerate(ordered_edges):
                edge_to_color[e] = coloring[i]
            
            # Rebuild in original edge order
            final_coloring = [edge_to_color[e] for e in edges]
            
            # Try local search
            if nc < best_num_colors:
                improved_coloring, nc2 = local_search_improve(final_coloring, edges, adj)
                if nc2 < best_num_colors:
                    best_coloring = improved_coloring
                    best_num_colors = nc2
                elif nc < best_num_colors:
                    best_coloring = final_coloring
                    best_num_colors = nc
        
        if best_coloring is None:
            # Fallback: simple greedy
            coloring, nc = greedy_color(edges, adj, edges)
            best_coloring = coloring
            best_num_colors = nc
        
        return best_coloring

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() >= deadline:
            # Emergency fallback
            n, d, seed = spec
            edges = instance(n, d, seed)
            adj = build_adj(edges, n)
            coloring, _ = greedy_color(edges, adj, edges)
            out.append(coloring)
            continue
        coloring = solve_instance(*spec)
        out.append(coloring)
    
    return out

# EVOLVE-BLOCK-END
