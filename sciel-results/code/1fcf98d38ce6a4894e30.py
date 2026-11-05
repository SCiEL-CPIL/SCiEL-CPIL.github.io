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
        """Try to improve coloring by recoloring individual edges."""
        m = len(edges)
        colors = initial_colors[:]
        
        # Precompute conflict lists as sorted lists for faster iteration
        conflict_lists = [list(c) for c in conflicts]
        
        rng = random.Random(42)
        improved = True
        steps = 0
        
        while improved and steps < max_steps:
            improved = False
            # Shuffle order to break symmetry
            order = list(range(m))
            rng.shuffle(order)
            
            for i in order:
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
                    improved = True
                    steps += 1
                    if steps > max_steps:
                        break
            
        return colors, max(colors) + 1

    def swap_search(edges, conflicts, initial_colors, max_swaps=500):
        """Try to improve coloring by swapping colors of two edges."""
        m = len(edges)
        colors = initial_colors[:]
        conflict_lists = [list(c) for c in conflicts]
        
        rng = random.Random(123)
        improved = True
        swaps = 0
        
        while improved and swaps < max_swaps:
            improved = False
            # Try random pairs of edges
            for _ in range(100):
                i, j = rng.sample(range(m), 2)
                ci, cj = colors[i], colors[j]
                if ci == cj:
                    continue
                
                # Check if swapping colors[i] and colors[j] is valid
                # i can take cj? Check conflicts of i don't have cj
                ok_i = True
                for k in conflict_lists[i]:
                    if colors[k] == cj and k != j:
                        ok_i = False
                        break
                if not ok_i:
                    continue
                
                # j can take ci? Check conflicts of j don't have ci
                ok_j = True
                for k in conflict_lists[j]:
                    if colors[k] == ci and k != i:
                        ok_j = False
                        break
                if not ok_j:
                    continue
                
                # Swap
                colors[i] = cj
                colors[j] = ci
                swaps += 1
                improved = True
                if swaps > max_swaps:
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
            ls_colors, ls_score = local_search(edges, conflicts, best_colors, max_steps=2000)
            if ls_score < best_score:
                best_score = ls_score
                best_colors = ls_colors
            
            # Swap search
            sw_colors, sw_score = swap_search(edges, conflicts, best_colors, max_swaps=500)
            if sw_score < best_score:
                best_score = sw_score
                best_colors = sw_colors
        
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
