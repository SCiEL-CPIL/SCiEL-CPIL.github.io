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

    def local_search(edges, conflicts, initial_colors, max_steps=1000, time_budget=30.0):
        """Try to improve coloring by recoloring and swapping colors."""
        m = len(edges)
        colors = initial_colors[:]
        current_cost = max(colors) + 1 if m > 0 else 0
        
        # Precompute conflict lists as sorted lists for faster iteration
        conflict_lists = [list(c) for c in conflicts]
        
        rng = random.Random(42)
        start = time.time()
        improved = True
        steps = 0
        
        while improved and steps < max_steps and (time.time() - start) < time_budget:
            improved = False
            # Shuffle order to break symmetry
            order = list(range(m))
            rng.shuffle(order)
            
            for i in order:
                if (time.time() - start) > time_budget:
                    break
                # Try to reduce the color of edge i
                current_color = colors[i]
                
                # Find the smallest color available for edge i
                forbidden = set()
                for j in conflict_lists[i]:
                    forbidden.add(colors[j])
                
                # Try colors from 0 up to current_color - 1
                new_color = None
                for c in range(current_color):
                    if c not in forbidden:
                        new_color = c
                        break
                
                if new_color is not None:
                    colors[i] = new_color
                    new_cost = max(colors) + 1
                    if new_cost < current_cost:
                        current_cost = new_cost
                    improved = True
                    steps += 1
            
            # Try color swaps: for each pair of colors (a, b), try to swap them
            # to reduce the maximum color used
            if current_cost > 2:
                max_color = current_cost - 1
                # Try to eliminate the highest color by swapping edges
                for i in order:
                    if (time.time() - start) > time_budget:
                        break
                    if colors[i] != max_color:
                        continue
                    # Edge i has max_color, try to move it to a lower color
                    # or swap with another edge
                    forbidden = set()
                    for j in conflict_lists[i]:
                        forbidden.add(colors[j])
                    
                    # Try to recolor edge i to a lower color
                    for c in range(max_color):
                        if c not in forbidden:
                            colors[i] = c
                            new_cost = max(colors) + 1
                            if new_cost < current_cost:
                                current_cost = new_cost
                            improved = True
                            steps += 1
                            break
                    else:
                        # Can't recolor directly, try a swap: find edge k with color c
                        # such that swapping colors[i] and colors[k] is valid
                        for c in range(max_color):
                            if c in forbidden:
                                continue
                            # Find an edge k with color c that doesn't conflict with i
                            # after the swap
                            for k in order:
                                if k == i or colors[k] != c:
                                    continue
                                # Check if swapping is valid
                                # After swap: i gets c, k gets max_color
                                # i's new neighbors must not have color c (they don't, since c not in forbidden)
                                # k's new neighbors must not have color max_color
                                k_forbidden = set()
                                for j in conflict_lists[k]:
                                    if j != i:
                                        k_forbidden.add(colors[j])
                                if max_color not in k_forbidden:
                                    # Swap is valid
                                    colors[i], colors[k] = colors[k], colors[i]
                                    new_cost = max(colors) + 1
                                    if new_cost < current_cost:
                                        current_cost = new_cost
                                    improved = True
                                    steps += 1
                                    break
                            else:
                                continue
                            break
        
        return colors, max(colors) + 1 if m > 0 else 0

    def color_graph(edges, adj, seed_offset=0, time_budget=25.0):
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
        
        # Ordering 3: Random shuffle - try more random orderings
        rng = random.Random(seed_offset)
        for _ in range(10):
            order3 = list(range(m))
            rng.shuffle(order3)
            colors3, score3 = greedy_color(edges, conflicts, order3)
            if score3 < best_score:
                best_score = score3
                best_colors = colors3[:]
        
        # Local search from best greedy solution
        if best_colors is not None:
            ls_colors, ls_score = local_search(edges, conflicts, best_colors, max_steps=2000, time_budget=time_budget)
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
        
    best_results = [[] for _ in range(len(graphs))]
    best_scores = [float('inf')] * len(graphs)
    
    # Allocate time per graph: roughly equal, but give more to harder graphs
    # Graph sizes: 40 nodes/d=4, 60 nodes/d=5, 80 nodes/d=6
    # Allocate ~35s each with 5s buffer
    per_graph_budget = [35.0, 35.0, 35.0]
    
    for idx, (edges, adj) in enumerate(graphs):
        graph_start = time.time()
        # Run with different seed offsets
        for offset in range(30):
            elapsed = time.time() - graph_start
            if elapsed > per_graph_budget[idx] or time.time() > deadline - 10.0:
                break
            # Use remaining time as budget for this run
            remaining = min(per_graph_budget[idx] - elapsed, 25.0)
            if remaining < 2.0:
                break
            colors, score = color_graph(edges, adj, seed_offset=offset, time_budget=remaining)
            if score < best_scores[idx]:
                best_scores[idx] = score
                best_results[idx] = colors
        
        # If we ran out of time, break outer loop
        if time.time() > deadline - 10.0:
            break
    
    # If any graph didn't get a result, fill with a quick greedy
    for i in range(len(graphs)):
        if not best_results[i]:
            colors, _ = color_graph(graphs[i][0], graphs[i][1], seed_offset=0, time_budget=5.0)
            best_results[i] = colors

    return best_results
