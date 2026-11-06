"""Strong edge colouring via local search with simulated annealing and restarts."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def build_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
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
    
    def strong_edge_coloring(edges, adj, max_time):
        m = len(edges)
        if m == 0:
            return []
        
        # Precompute conflict: edge i conflicts with edge j if they share a vertex
        # or are joined by an edge (i.e., there's an edge between endpoints)
        # For induced matching: no two edges share a vertex AND no edge connects
        # an endpoint of one to an endpoint of the other.
        
        # Build edge index map
        edge_idx = {e: i for i, e in enumerate(edges)}
        
        # For each edge, compute which other edges conflict with it
        # Two edges (u,v) and (a,b) conflict if:
        # 1. They share a vertex: {u,v} & {a,b} != empty
        # 2. There's an edge connecting an endpoint of one to an endpoint of the other
        #    i.e., (u,a) in adj, (u,b) in adj, (v,a) in adj, or (v,b) in adj
        #    (but not if they share a vertex, already covered)
        
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i+1, m):
                a, b = edges[j]
                # Check if they share a vertex
                if u == a or u == b or v == a or v == b:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if there's an edge joining them
                # Need to check: is (u,a) an edge? (u,b)? (v,a)? (v,b)?
                # But we need to be careful: the edge (u,a) must be in the graph
                # and not be the same as edge i or edge j
                if (min(u,a), max(u,a)) in edge_idx or (min(u,b), max(u,b)) in edge_idx or \
                   (min(v,a), max(v,a)) in edge_idx or (min(v,b), max(v,b)) in edge_idx:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        # Convert to bitmasks for speed
        # Actually for m up to ~240 edges, sets might be fine but let's use lists
        conflict_list = [list(s) for s in conflict]
        
        # Greedy initialization using DSATUR-like ordering
        def greedy_color():
            # Order edges by some heuristic: start with edges that have most conflicts
            # Actually, let's use a simple approach: color edges one by one
            # choosing the color that minimizes conflicts with already-colored edges
            
            # Sort edges by degree in conflict graph (most constrained first)
            edge_order = sorted(range(m), key=lambda i: len(conflict_list[i]), reverse=True)
            
            color = [None] * m
            color_count = 0
            
            for i in edge_order:
                # Find the smallest color that doesn't conflict with any already-colored edge
                used_colors = set()
                for j in conflict_list[i]:
                    if color[j] is not None:
                        used_colors.add(color[j])
                
                # Try colors 0, 1, 2, ...
                c = 0
                while c in used_colors:
                    c += 1
                color[i] = c
                color_count = max(color_count, c + 1)
            
            return color, color_count
        
        best_color, best_count = greedy_color()
        
        if m <= 50:
            return [best_color[i] for i in range(m)]
        
        # Local search with simulated annealing
        def count_colors(color):
            return max(color) + 1 if m > 0 else 0
        
        def try_improve(color, color_count, time_limit):
            """Local search: try to recolor edges to reduce total colors."""
            start_time = time.time()
            current_color = color[:]
            current_count = color_count
            best_local = current_color[:]
            best_local_count = current_count
            
            # Build color class structures
            def rebuild_classes(clr):
                classes = defaultdict(list)
                for i in range(m):
                    classes[clr[i]].append(i)
                return classes
            
            classes = rebuild_classes(current_color)
            
            # For each color, track which edges are in it
            # Move: pick an edge, try to recolor it to a different color
            # We want to eliminate empty color classes
            
            rng = random.Random(42)
            
            while time.time() - start_time < time_limit:
                # Find the highest color that has edges
                max_c = max(current_color)
                
                # Try to eliminate the highest color
                # Pick an edge from the highest color class
                if not classes[max_c]:
                    # Shouldn't happen, but handle it
                    break
                
                # Try to move each edge from the highest color to a lower color
                moved = False
                for edge_i in classes[max_c][:]:
                    # Try to recolor edge_i to a lower color
                    for new_c in range(max_c):
                        # Check if edge_i can be moved to color new_c
                        # It must not conflict with any edge in color new_c
                        can_move = True
                        for j in conflict_list[edge_i]:
                            if current_color[j] == new_c:
                                can_move = False
                                break
                        if can_move:
                            # Move it
                            classes[max_c].remove(edge_i)
                            if new_c not in classes:
                                classes[new_c] = []
                            classes[new_c].append(edge_i)
                            current_color[edge_i] = new_c
                            moved = True
                            break
                    if moved:
                        break
                
                if not moved:
                    # Try a random perturbation: pick a random edge and recolor it
                    # to a random valid color
                    edge_i = rng.randint(0, m-1)
                    old_c = current_color[edge_i]
                    # Find all valid colors for edge_i
                    valid_colors = set(range(max_c + 1))
                    for j in conflict_list[edge_i]:
                        valid_colors.discard(current_color[j])
                    if valid_colors:
                        new_c = rng.choice(list(valid_colors))
                        if new_c != old_c:
                            classes[old_c].remove(edge_i)
                            if new_c not in classes:
                                classes[new_c] = []
                            classes[new_c].append(edge_i)
                            current_color[edge_i] = new_c
                
                # Update best
                cur_count = max(current_color) + 1
                if cur_count < best_local_count:
                    best_local_count = cur_count
                    best_local = current_color[:]
                    # Try to eliminate more colors
                    # Keep going to see if we can do better
                
                # Check if we've eliminated the top color
                if best_local_count <= max_c - 1:
                    # We made progress, continue
                    pass
                else:
                    # No progress in eliminating the top color, try restart
                    break
            
            return best_local, best_local_count
        
        # Multiple restarts with different initial orderings
        total_time = time.time()
        max_total = 120
        
        for restart in range(20):
            if time.time() - total_time > max_total - 5:
                break
            
            # Different greedy orderings for each restart
            rng_r = random.Random(restart * 12345 + 678)
            
            # Shuffle the edge order with some bias
            edge_order = list(range(m))
            # Weight by conflict degree
            weights = [len(conflict_list[i]) + 1 for i in range(m)]
            for i in range(len(edge_order) - 1, 0, -1):
                j = rng_r.randint(0, i)
                edge_order[i], edge_order[j] = edge_order[j], edge_order[i]
            
            # Greedy with this order
            color = [None] * m
            color_count = 0
            for i in edge_order:
                used_colors = set()
                for j in conflict_list[i]:
                    if color[j] is not None:
                        used_colors.add(color[j])
                c = 0
                while c in used_colors:
                    c += 1
                color[i] = c
                color_count = max(color_count, c + 1)
            
            # Local search
            remaining = max_total - (time.time() - total_time) - 5
            if remaining <= 0:
                break
            ls_time = min(remaining / 2, 3.0)
            
            improved_color, improved_count = try_improve(color, color_count, ls_time)
            
            if improved_count < best_count:
                best_count = improved_count
                best_color = improved_color
            
            # Early exit if we hit the theoretical lower bound
            # Lower bound: max degree + 1 (for proper edge coloring)
            # For strong coloring, lower bound is at least max degree + 1
            # Actually, the chromatic index of the square of the line graph
            # Simple lower bound: maximum degree in the conflict graph + 1
            max_conflict_deg = max(len(conflict_list[i]) for i in range(m))
            lb = max_conflict_deg + 1
            if best_count <= lb:
                break
        
        return [best_color[i] for i in range(m)]
    
    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    # Allocate time: more time for larger instances
    # (40,4): ~40 edges, (60,5): ~60 edges, (80,6): ~80 edges
    # Allocate proportional to edge count
    n_edges_est = [n * d // 2 for n, d, _ in specs]  # rough estimate
    total_est = sum(n_edges_est)
    time_alloc = [120 * ne / total_est for ne in n_edges_est]
    # Give minimum 10s to each
    time_alloc = [max(10, t) for t in time_alloc]
    # Normalize
    s = sum(time_alloc)
    time_alloc = [t * 120 / s for t in time_alloc]
    
    for idx, spec in enumerate(specs):
        n, d, seed = spec
        edges = build_instance(n, d, seed)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colors = strong_edge_coloring(edges, adj, time_alloc[idx])
        results.append(colors)
        
        # Check deadline
        if time.time() > deadline:
            # Return what we have
            break
    
    # If we ran out of time, fill remaining with greedy
    while len(results) < 3:
        idx = len(results)
        n, d, seed = specs[idx]
        edges = build_instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Quick greedy
        m = len(edges)
        edge_idx = {e: i for i, e in enumerate(edges)}
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i+1, m):
                a, b = edges[j]
                if u == a or u == b or v == a or v == b:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                if (min(u,a), max(u,a)) in edge_idx or (min(u,b), max(u,b)) in edge_idx or \
                   (min(v,a), max(v,a)) in edge_idx or (min(v,b), max(v,b)) in edge_idx:
                    conflict[i].add(j)
                    conflict[j].add(i)
        color = [0] * m
        for i in range(m):
            used = set()
            for j in conflict[i]:
                if j < i:
                    used.add(color[j])
            c = 0
            while c in used:
                c += 1
            color[i] = c
        results.append(color)
    
    return results
# EVOLVE-BLOCK-END
