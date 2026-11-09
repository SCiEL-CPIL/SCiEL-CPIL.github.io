"""Strong edge colourings of three fixed bounded-degree graphs."""
import time

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

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

    def build_conflict_bitmasks(n, edges):
        """Build for each edge a bitmask of edges that conflict with it (share vertex or joined by edge)."""
        m = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        # adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # For each edge i, compute conflict mask
        conflict = [0] * m
        for i, (u, v) in enumerate(edges):
            # Edges sharing vertex u or v
            for w in adj[u] | adj[v]:
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    e2 = (min(u, w), max(u, w)) if w != u else (min(v, w), max(v, w))
                    # Actually let's be more careful
                    pass
            # Simpler: for each edge, find all edges that share a vertex or are joined
            # Two edges conflict if:
            # 1. They share a vertex (i.e., not a matching)
            # 2. There is an edge connecting one endpoint of one to one endpoint of the other (induced matching violation)
            
        # Let me redo this more carefully
        # Edge i = (u, v), Edge j = (a, b)
        # Conflict if:
        #   {u,v} & {a,b} != empty (share vertex)
        #   OR (u,a) in edge_set OR (u,b) in edge_set OR (v,a) in edge_set OR (v,b) in edge_set
        #   but NOT if they share a vertex already (that's already a conflict)
        
        edge_set = set(edges)
        for i, (u, v) in enumerate(edges):
            mask = 0
            for j, (a, b) in enumerate(edges):
                if i == j:
                    continue
                # Check if they share a vertex
                if u == a or u == b or v == a or v == b:
                    mask |= (1 << j)
                    continue
                # Check if joined by an edge
                if (u, a) in edge_set or (min(u,a), max(u,a)) in edge_set:
                    mask |= (1 << j)
                    continue
                if (u, b) in edge_set or (min(u,b), max(u,b)) in edge_set:
                    mask |= (1 << j)
                    continue
                if (v, a) in edge_set or (min(v,a), max(v,a)) in edge_set:
                    mask |= (1 << j)
                    continue
                if (v, b) in edge_set or (min(v,b), max(v,b)) in edge_set:
                    mask |= (1 << j)
                    continue
            conflict[i] = mask
        return conflict

    def greedy_color(order, conflict, m):
        """Greedy coloring: assign each edge the smallest available color."""
        color = [-1] * m
        color_used = [0] * (m + 1)  # color_used[c] = bitmask of edges colored with color c
        for idx in order:
            # Find forbidden colors
            forb = conflict[idx]
            # A color c is forbidden if any edge with color c conflicts with idx
            # i.e., if color_used[c] & forb != 0
            for c in range(m + 1):
                if color_used[c] & forb == 0:
                    color[idx] = c
                    color_used[c] |= (1 << idx)
                    break
        return color

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict masks
        conflict = build_conflict_bitmasks(n, edges)
        
        # Try multiple orderings and keep the best
        best_color = None
        best_num_colors = m + 1
        
        # Ordering 1: conflict-degree sorted (most constrained first)
        conf_deg = [bin(conflict[i]).count('1') for i in range(m)]
        order1 = sorted(range(m), key=lambda i: -conf_deg[i])
        
        # Ordering 2: original order
        order2 = list(range(m))
        
        # Ordering 3: reverse original
        order3 = list(range(m - 1, -1, -1))
        
        # Ordering 4: degree-based (edges with higher-degree endpoints first)
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        order4 = sorted(range(m), key=lambda i: -(deg[edges[i][0]] + deg[edges[i][1]]))
        
        # Ordering 5: random with fixed seed
        from random import Random
        rng = Random(42)
        order5 = list(range(m))
        rng.shuffle(order5)
        
        # Ordering 6: random with another seed
        rng2 = Random(123)
        order6 = list(range(m))
        rng2.shuffle(order6)
        
        # Ordering 7: conflict-degree with tie-breaking by degree
        order7 = sorted(range(m), key=lambda i: (-conf_deg[i], -(deg[edges[i][0]] + deg[edges[i][1]])))
        
        # Ordering 8: LPT-like by endpoint degree product
        order8 = sorted(range(m), key=lambda i: -(deg[edges[i][0]] * deg[edges[i][1]]))
        
        orders = [order1, order2, order3, order4, order5, order6, order7, order8]
        
        for order in orders:
            if time.time() > DEADLINE:
                break
            color = greedy_color(order, conflict, m)
            num_colors = max(color) + 1
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_color = color
        
        # Additional random orderings if time permits
        extra_rng = Random(999)
        for trial in range(20):
            if time.time() > DEADLINE:
                break
            order = list(range(m))
            extra_rng.shuffle(order)
            color = greedy_color(order, conflict, m)
            num_colors = max(color) + 1
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_color = color
        
        if best_color is None:
            # Fallback: simple greedy in original order
            best_color = greedy_color(list(range(m)), conflict, m)
        
        # Return colors in sorted edge order
        return [best_color[i] for i in range(m)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE:
            # If we're out of time, use a quick greedy for remaining instances
            edges = instance(*spec)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            conflict = build_conflict_bitmasks(spec[0], edges)
            color = greedy_color(list(range(m)), conflict, m)
            out.append([color[i] for i in range(m)])
            continue
        out.append(solve_instance(*spec))
    
    return out
