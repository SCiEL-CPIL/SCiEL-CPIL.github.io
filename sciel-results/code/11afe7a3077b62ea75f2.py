"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Try many times to build a bounded-degree graph
        for _ in range(20 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_edge_coloring(edges, deadline):
        if not edges:
            return []

        m = len(edges)
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0

        # Build vertex adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute for each edge the set of vertices it "touches"
        # Edge e=(u,v) touches u, v, and all neighbors of u and v
        # Two edges conflict if they share a vertex or are joined by an edge
        # i.e., if any endpoint of one is in the closed neighborhood of any endpoint of the other

        # For each edge i, compute the set of edges that conflict with it
        # Edge i=(u,v) conflicts with edge j=(w,x) if:
        # - They share a vertex: u==w, u==x, v==w, or v==x
        # - They are joined by an edge: u-w, u-x, v-w, or v-x is an edge in the graph

        edge_to_vertices = {}
        for i, (u, v) in enumerate(edges):
            edge_to_vertices[i] = (u, v)

        # For each edge, find all conflicting edges
        # Build a map from vertex to list of edges incident to it
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)

        # conflict[i] = set of edges that conflict with edge i
        conflict = [set() for _ in range(m)]

        for i, (u, v) in enumerate(edges):
            # Edges sharing u or v
            for j in vertex_to_edges[u]:
                if j != i:
                    conflict[i].add(j)
            for j in vertex_to_edges[v]:
                if j != i:
                    conflict[i].add(j)
            # Edges joined by an edge from u or v
            for w in adj[u]:
                for j in vertex_to_edges[w]:
                    if j != i:
                        conflict[i].add(j)
            for w in adj[v]:
                for j in vertex_to_edges[w]:
                    if j != i:
                        conflict[i].add(j)

        # Convert to bitmask for efficiency
        conflict_mask = [0] * m
        for i in range(m):
            mask = 0
            for j in conflict[i]:
                mask |= (1 << j)
            conflict_mask[i] = mask

        # Greedy coloring with multiple orderings, then local search
        def greedy_color(order):
            color = [-1] * m
            color_masks = []  # bitmask of edges per color
            max_color = -1

            for idx in order:
                # Find the smallest color that doesn't conflict
                taken = set()
                for c, mask in enumerate(color_masks):
                    if conflict_mask[idx] & mask:
                        taken.add(c)

                c = 0
                while c in taken:
                    c += 1

                color[idx] = c
                if c >= len(color_masks):
                    color_masks.append(0)
                color_masks[c] |= (1 << idx)
                if c > max_color:
                    max_color = c

            return color, max_color + 1

        # Local search: try to recolor edges to reduce color count
        def local_search(color, num_colors, deadline_local):
            color = color[:]
            current_colors = num_colors
            improved = True

            while improved and time.time() < deadline_local:
                improved = False
                # Try to reduce the highest color
                # Find edges with the highest color
                highest = current_colors - 1
                edges_at_highest = [i for i in range(m) if color[i] == highest]

                for i in edges_at_highest:
                    # Try to recolor edge i to a lower color
                    for c in range(highest):
                        # Check if color c is available for edge i
                        # Get all edges with color c
                        mask_c = 0
                        for j in range(m):
                            if color[j] == c:
                                mask_c |= (1 << j)
                        if not (conflict_mask[i] & mask_c):
                            # Recolor
                            color[i] = c
                            improved = True
                            break

                # Update current_colors
                used = set(color)
                current_colors = len(used) if used else 0

                # If we removed the highest color, continue
                if not any(color[i] == highest for i in range(m)):
                    continue

            # Renumber colors to be compact
            if any(color[i] >= 0 for i in range(m)):
                used = sorted(set(color))
                remap = {c: i for i, c in enumerate(used)}
                color = [remap[c] for c in color]
                return color, len(used)
            else:
                return color, 0

        # Generate different orderings
        orderings = []

        # Order by degree (number of conflicts)
        deg = [len(conflict[i]) for i in range(m)]
        orderings.append(sorted(range(m), key=lambda i: deg[i], reverse=True))
        orderings.append(sorted(range(m), key=lambda i: deg[i]))

        # Random orderings
        rng = random.Random(12345)
        for _ in range(10):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)

        # Order by vertex degree
        vert_deg = [len(adj.get(u, set())) + len(adj.get(v, set())) for u, v in edges]
        orderings.append(sorted(range(m), key=lambda i: vert_deg[i], reverse=True))

        best_color = None
        best_num_colors = m + 1

        dl = min(deadline, start_time + 135.0)

        # Try greedy colorings with different orderings
        for order in orderings:
            if time.time() > dl:
                break
            color, nc = greedy_color(order)
            if nc < best_num_colors:
                best_num_colors = nc
                best_color = color[:]

        # Apply local search to the best coloring
        if best_color is not None and time.time() < dl:
            color, nc = local_search(best_color, best_num_colors, dl)
            if nc < best_num_colors:
                best_num_colors = nc
                best_color = color[:]

        # If we still have time, try more random orderings
        if best_color is not None and time.time() < dl:
            for _ in range(20):
                if time.time() > dl:
                    break
                order = list(range(m))
                rng.shuffle(order)
                color, nc = greedy_color(order)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_color = color[:]
                # Quick local search
                if nc <= best_num_colors and time.time() < dl:
                    color2, nc2 = local_search(color, nc, min(dl, time.time() + 2.0))
                    if nc2 < best_num_colors:
                        best_num_colors = nc2
                        best_color = color2[:]

        if best_color is None:
            # Fallback: simple greedy
            color, nc = greedy_color(list(range(m)))
            best_color = color
            best_num_colors = nc

        return best_color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE:
            # Fallback simple greedy
            edges = instance(*spec)
            if not edges:
                out.append([])
                continue
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            vertex_to_edges = defaultdict(list)
            for i, (u, v) in enumerate(edges):
                vertex_to_edges[u].append(i)
                vertex_to_edges[v].append(i)
            color = [0] * len(edges)
            for i, (u, v) in enumerate(edges):
                taken = set()
                for j in vertex_to_edges[u]:
                    if j < i:
                        taken.add(color[j])
                for j in vertex_to_edges[v]:
                    if j < i:
                        taken.add(color[j])
                for w in adj[u]:
                    for j in vertex_to_edges[w]:
                        if j < i:
                            taken.add(color[j])
                for w in adj[v]:
                    for j in vertex_to_edges[w]:
                        if j < i:
                            taken.add(color[j])
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
            out.append(color)
            continue

        edges = instance(*spec)
        colors = strong_edge_coloring(edges, DEADLINE)
        out.append(colors)

    return out
# EVOLVE-BLOCK-END
