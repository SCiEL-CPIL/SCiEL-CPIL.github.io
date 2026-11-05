import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin

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

    def build_conflict(edges, adj):
        """Build conflict sets: two edges conflict if they share a vertex or are joined by an edge."""
        n_edges = len(edges)
        edge_set = set(edges)
        # For each edge, find all conflicting edges
        conflicts = [set() for _ in range(n_edges)]
        edge_index = {e: i for i, e in enumerate(edges)}
        
        for i, (u, v) in enumerate(edges):
            # Edges sharing u or v
            for w in adj[u]:
                e1 = (min(u, w), max(u, w))
                if e1 in edge_index:
                    j = edge_index[e1]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            for w in adj[v]:
                e1 = (min(v, w), max(v, w))
                if e1 in edge_index:
                    j = edge_index[e1]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            # Edges joined by an edge: for each neighbor of u, check if connected to v, and vice versa
            # Actually, "joined by an edge" means there exists an edge (x,y) where x in {u,v} and y in {x',y'} 
            # Wait, let me re-read: "no two edges share a vertex or are joined by an edge"
            # Two edges e1=(u,v) and e2=(x,y) are "joined by an edge" if there's an edge in the graph
            # connecting one endpoint of e1 to one endpoint of e2.
            # So for edge (u,v), check all neighbors of u and all neighbors of v.
            # If neighbor w of u is in edge (x,y), then (u,v) and (x,y) are joined by edge (u,w) if w is x or y.
            # Actually more precisely: edges (u,v) and (x,y) conflict if:
            #   - they share a vertex (u==x, u==y, v==x, or v==y), OR
            #   - there is an edge in the graph between {u,v} and {x,y}, i.e., (u,x) in E, (u,y) in E, (v,x) in E, or (v,y) in E
            # The first case is already handled. For the second case:
            # For edge (u,v), for each neighbor w of u, if w is part of some edge (w,z), then (u,v) and (w,z) are joined by (u,w).
            # Similarly for neighbors of v.
            # But wait, this is the same as the first case! If w is a neighbor of u, then edge (u,w) exists, and any edge containing w...
            # Hmm, let me think again. Two edges (a,b) and (c,d) are at distance 1 in the line graph if they share a vertex.
            # They are "joined by an edge" if there's a path of length 2 in the original graph between them, i.e., 
            # there exists a vertex x such that x is adjacent to one endpoint of (a,b) and one endpoint of (c,d).
            # Actually, I think "joined by an edge" means the two edges are connected by a single edge in the graph.
            # So edge (u,v) and edge (x,y) are joined by an edge if (u,x) in E, (u,y) in E, (v,x) in E, or (v,y) in E.
            # This is different from sharing a vertex.
            
            # Let me redo: for edge (u,v), the conflicting edges are:
            # 1. All edges sharing u or v (already done above)
            # 2. All edges (x,y) such that (u,x) in E or (u,y) in E or (v,x) in E or (v,y) in E
            #    But we need to exclude edges already counted in case 1.
            
            # For case 2: for each neighbor w of u, look at all edges containing w. For each such edge (w,z), 
            # if z != v and z != u (to avoid double-counting with case 1), then (u,v) and (w,z) conflict.
            # Similarly for neighbors of v.
            
            for w in adj[u]:
                for z in adj[w]:
                    if z == u or z == v:
                        continue
                    e2 = (min(w, z), max(w, z))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
            for w in adj[v]:
                for z in adj[w]:
                    if z == u or z == v:
                        continue
                    e2 = (min(w, z), max(w, z))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
        
        return conflicts

    def greedy_color(order, conflicts, n_edges):
        """Color edges in given order using first-fit."""
        color = [-1] * n_edges
        color_conflicts = []  # color_conflicts[c] = set of edge indices with color c
        for idx in order:
            used = set()
            for c in range(len(color_conflicts)):
                if color_conflicts[c] & conflicts[idx]:
                    used.add(c)
            # Find smallest available color
            c = 0
            while c in used:
                c += 1
            color[idx] = c
            if c >= len(color_conflicts):
                color_conflicts.append(set())
            color_conflicts[c].add(idx)
        return color, len(color_conflicts)

    def try_improve(color, conflicts, n_edges, max_iter=5000):
        """Try to reduce number of colors by reassigning edges."""
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors
        
        # Build color classes
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)
        
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # Try to remove a color class by moving its edges to other classes
            for c in range(n_colors):
                if len(classes[c]) == 0:
                    continue
                # Try to move all edges in class c to other classes
                moved = True
                temp_assign = {}
                for e in sorted(classes[c]):
                    # Find a color class (not c) where e can go
                    placed = False
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        # Check if e conflicts with any edge in class c2
                        if not (conflicts[e] & classes[c2]):
                            temp_assign[e] = c2
                            placed = True
                            break
                    if not placed:
                        moved = False
                        break
                if moved:
                    # Apply the move
                    for e, c2 in temp_assign.items():
                        classes[c].discard(e)
                        classes[c2].add(e)
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    break
        
        return color, n_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n_edges = len(edges)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflicts = build_conflict(edges, adj)
        
        # Compute conflict degrees
        conf_deg = [len(conflicts[i]) for i in range(n_edges)]
        
        best_color = None
        best_n = float('inf')
        
        # Strategy 1: Sort by descending conflict degree
        order1 = sorted(range(n_edges), key=lambda i: -conf_deg[i])
        color1, n1 = greedy_color(order1, conflicts, n_edges)
        color1, n1 = try_improve(color1, conflicts, n_edges)
        if n1 < best_n:
            best_n = n1
            best_color = color1[:]
        
        # Strategy 2: Random orderings
        rng = random.Random(42)
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            order = list(range(n_edges))
            rng.shuffle(order)
            color, n = greedy_color(order, conflicts, n_edges)
            color, n = try_improve(color, conflicts, n_edges, max_iter=2000)
            if n < best_n:
                best_n = n
                best_color = color[:]
        
        # Strategy 3: Multiple passes with different heuristics
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Weighted random: higher conflict degree = higher priority
            weights = [conf_deg[i] + 1 for i in range(n_edges)]
            total_w = sum(weights)
            order = []
            remaining = set(range(n_edges))
            while remaining:
                # Pick with probability proportional to weight
                r = rng.random() * total_w
                cum = 0
                chosen = None
                for i in sorted(remaining, key=lambda x: -conf_deg[x]):
                    cum += weights[i]
                    if r <= cum:
                        chosen = i
                        break
                if chosen is None:
                    chosen = sorted(remaining, key=lambda x: -conf_deg[x])[0]
                order.append(chosen)
                remaining.remove(chosen)
                total_w -= weights[chosen]
            
            color, n = greedy_color(order, conflicts, n_edges)
            color, n = try_improve(color, conflicts, n_edges, max_iter=2000)
            if n < best_n:
                best_n = n
                best_color = color[:]
        
        if best_color is None:
            # Fallback
            best_color = list(range(n_edges))
            best_n = n_edges
        
        results.append([best_color[i] for i in range(n_edges)])
    
    return results
