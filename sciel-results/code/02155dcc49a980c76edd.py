import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

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
        m = len(edges)
        if m == 0:
            return []
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            # For each neighbor x of u, all edges incident to x conflict with (u,v)
            # because (u,x) is an edge (joined by edge condition) or they share x.
            # Actually, if e2 = (x, y), it conflicts with e1=(u,v) if:
            # - x==u or x==v or y==u or y==v (share vertex)
            # - (u,x) in E or (u,y) in E or (v,x) in E or (v,y) in E (joined by edge)
            
            # Efficient approach:
            # 1. All edges incident to u or v conflict (share vertex).
            # 2. All edges (x,y) where x in N(u) and y in N(v) conflict (joined by edge u-x or v-y? No.)
            #    Wait, the condition is: e1=(u,v), e2=(x,y).
            #    Joined by edge means there is an edge connecting an endpoint of e1 to an endpoint of e2.
            #    So if (u,x) in E, conflict. If (u,y) in E, conflict. If (v,x) in E, conflict. If (v,y) in E, conflict.
            
            # Let's collect candidates:
            # Candidates for endpoints of conflicting edges are N(u) U N(v) U {u, v}.
            # But we can just iterate:
            # For each x in N(u):
            #   For each y in N(x):
            #     e2 = (min(x,y), max(x,y))
            #     This covers all edges incident to any neighbor of u.
            #     Why? If e2=(a,b) conflicts with e1=(u,v) because (u,a) in E, then a is in N(u).
            #     Then b is in N(a). So when x=a, y=b, we find e2.
            #     Similarly if (u,b) in E, b in N(u), a in N(b).
            #     If (v,a) in E, a in N(v). We need to also check neighbors of v.
            
            # So:
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            for x in adj[v]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
                        
            conflicts[i].discard(i)
            
        return conflicts

    def greedy_color(edges, conflicts, order):
        """Color edges in the given order using smallest available color."""
        m = len(edges)
        color_of_edge = [-1] * m
        used_colors = 0
        
        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if color_of_edge[j] != -1:
                    forbidden.add(color_of_edge[j])
            
            c = 0
            while c in forbidden:
                c += 1
            
            color_of_edge[i] = c
            if c + 1 > used_colors:
                used_colors = c + 1
        
        return color_of_edge, used_colors

    def local_search(edges, conflicts, initial_colors, max_steps=2000):
        """Try to improve coloring via recoloring, swaps, and color-merging."""
        m = len(edges)
        colors = initial_colors[:]
        
        conflict_lists = [list(c) for c in conflicts]
        rng = random.Random(42)
        
        def get_forbidden(i, exclude_edge=None):
            """Get set of colors used by neighbors of edge i."""
            forbidden = set()
            for j in conflict_lists[i]:
                if j != exclude_edge:
                    forbidden.add(colors[j])
            return forbidden
        
        def try_recolor(i):
            """Try to reduce color of edge i. Returns True if changed."""
            current_color = colors[i]
            if current_color == 0:
                return False
            forbidden = get_forbidden(i)
            for c in range(current_color):
                if c not in forbidden:
                    colors[i] = c
                    return True
            return False
        
        def try_swap(i, j):
            """Try to swap colors of edges i and j. Returns True if changed."""
            if i == j:
                return False
            ci, cj = colors[i], colors[j]
            if ci == cj:
                return False
            # Check if swap is valid
            # After swap: i gets cj, j gets ci
            # For i: all neighbors except j must not have color cj
            # For j: all neighbors except i must not have color ci
            for k in conflict_lists[i]:
                if k != j and colors[k] == cj:
                    return False
            for k in conflict_lists[j]:
                if k != i and colors[k] == ci:
                    return False
            colors[i], colors[j] = cj, ci
            return True
        
        def try_move(i, target_color):
            """Try to move edge i to target_color. Returns True if changed."""
            if colors[i] == target_color:
                return False
            forbidden = get_forbidden(i)
            if target_color not in forbidden:
                colors[i] = target_color
                return True
            return False
        
        def try_merge_color_class(max_color):
            """Try to merge color class (max_color-1) into max_color or vice versa.
            Actually, try to eliminate color class with fewest edges by moving its edges."""
            # Find color class with fewest edges (among colors > 0)
            color_to_edges = defaultdict(list)
            for i in range(m):
                color_to_edges[colors[i]].append(i)
            
            if len(color_to_edges) <= 1:
                return False
            
            # Try to eliminate the color class with the fewest edges
            min_color = min(color_to_edges.keys(), key=lambda c: len(color_to_edges[c]))
            if min_color == 0:
                return False
            
            edges_in_class = color_to_edges[min_color]
            moved_any = False
            for i in edges_in_class:
                # Try to move edge i to any other color
                forbidden = get_forbidden(i)
                for c in range(min_color):
                    if c not in forbidden:
                        colors[i] = c
                        moved_any = True
                        break
                else:
                    # Try higher colors
                    for c in range(min_color + 1, max_color + 1):
                        if c not in forbidden:
                            colors[i] = c
                            moved_any = True
                            break
            
            return moved_any
        
        improved = True
        steps = 0
        no_improve_rounds = 0
        
        while improved and steps < max_steps:
            improved = False
            no_improve_rounds += 1
            
            # Pass 1: Simple recoloring
            order = list(range(m))
            rng.shuffle(order)
            for i in order:
                if try_recolor(i):
                    improved = True
                    steps += 1
                    if steps >= max_steps:
                        return colors, max(colors) + 1
            
            # Pass 2: Try swaps
            if no_improve_rounds < 5:  # Only do swaps for first few rounds
                for _ in range(50):
                    i = rng.randrange(m)
                    j = rng.randrange(m)
                    if try_swap(i, j):
                        improved = True
                        steps += 1
                        if steps >= max_steps:
                            return colors, max(colors) + 1
            
            # Pass 3: Try to merge color classes (eliminate sparse colors)
            if no_improve_rounds >= 2:  # Start merging after some recoloring
                current_max = max(colors)
                if try_merge_color_class(current_max):
                    improved = True
                    steps += 1
                    if steps >= max_steps:
                        return colors, max(colors) + 1
            
            if not improved:
                # Try random restarts with different shuffles
                for _ in range(20):
                    order = list(range(m))
                    rng.shuffle(order)
                    for i in order:
                        if try_recolor(i):
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    # Try more aggressive: random recoloring to break local optimum
                    for _ in range(10):
                        i = rng.randrange(m)
                        old_color = colors[i]
                        # Try all colors
                        forbidden = get_forbidden(i)
                        available = [c for c in range(max(colors) + 2) if c not in forbidden]
                        if available and available != [old_color]:
                            colors[i] = rng.choice(available)
                            improved = True
                            break
                    if not improved:
                        break
        
        return colors, max(colors) + 1

    def color_graph(edges, adj, seed_offset=0):
        m = len(edges)
        if m == 0:
            return [], 0
        
        conflicts = build_conflict_graph(edges, adj)
        
        # Strategy 1: Greedy with various orderings
        best_colors = None
        best_score = float('inf')
        
        # Ordering 1: Descending conflict degree
        order1 = list(range(m))
        order1.sort(key=lambda i: len(conflicts[i]), reverse=True)
        colors1, score1 = greedy_color(edges, conflicts, order1)
        if score1 < best_score:
            best_score = score1
            best_colors = colors1[:]
        
        # Ordering 2: Ascending conflict degree
        order2 = list(range(m))
        order2.sort(key=lambda i: len(conflicts[i]))
        colors2, score2 = greedy_color(edges, conflicts, order2)
        if score2 < best_score:
            best_score = score2
            best_colors = colors2[:]
        
        # Ordering 3: Random shuffle
        rng = random.Random(seed_offset)
        for _ in range(5):
            order3 = list(range(m))
            rng.shuffle(order3)
            colors3, score3 = greedy_color(edges, conflicts, order3)
            if score3 < best_score:
                best_score = score3
                best_colors = colors3[:]
        
        # Local search from best greedy solution
        if best_colors is not None:
            ls_colors, ls_score = local_search(edges, conflicts, best_colors, max_steps=500)
            if ls_score < best_score:
                best_score = ls_score
                best_colors = ls_colors
        
        return best_colors, best_score

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    # Pre-generate graphs
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        graphs.append((edges, adj))
        
    best_results = []
    best_scores = [float('inf')] * len(graphs)
    
    # Initial pass: run multiple random seeds for each graph
    # We have ~130s. Let's allocate time per graph.
    # Graphs are small, so we can do many iterations.
    
    for idx, (edges, adj) in enumerate(graphs):
        # Run with different seed offsets
        for offset in range(20):
            if time.time() > deadline - 10.0:
                break
            colors, score = color_graph(edges, adj, seed_offset=offset)
            if score < best_scores[idx]:
                best_scores[idx] = score
                best_results.append(colors) if len(best_results) <= idx else None
                while len(best_results) < len(graphs):
                    best_results.append([])
                best_results[idx] = colors
        
        # If we ran out of time, break outer loop
        if time.time() > deadline - 10.0:
            break
    
    # If any graph didn't get a result, fill with a quick greedy
    for i in range(len(graphs)):
        if not best_results[i]:
            colors, _ = color_graph(graphs[i][0], graphs[i][1], seed_offset=0)
            best_results[i] = colors

    return best_results
