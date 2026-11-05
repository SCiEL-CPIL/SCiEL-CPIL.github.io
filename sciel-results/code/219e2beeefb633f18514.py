"""Strong edge colourings via multi-restart greedy + class-elimination local search."""
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

    def build_conflicts(edges, n):
        """
        Precompute conflict sets for all edges.
        Two edges conflict if they share a vertex or are joined by an edge (distance 1 in line graph).
        """
        edge_index = {e: i for i, e in enumerate(edges)}
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        
        for i, (u, v) in enumerate(edges):
            # Collect all vertices within distance 1 of u or v
            # Neighbors of u and v, plus u and v themselves
            near_vertices = {u, v}
            near_vertices.update(adj[u])
            near_vertices.update(adj[v])
            
            for x in near_vertices:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
                            
        return conflicts

    def greedy_color(order, conflicts, n_edges):
        """Color edges in given order using first-fit with precomputed conflicts."""
        color = [-1] * n_edges
        # color_to_edges[c] = set of edge indices with color c
        color_to_edges = []
        
        for idx in order:
            used = set()
            # Check which colors are used by conflicting edges
            for c_idx, edges_in_color in enumerate(color_to_edges):
                if edges_in_color & conflicts[idx]:
                    used.add(c_idx)
            
            # Find smallest available color
            c = 0
            while c in used:
                c += 1
            
            color[idx] = c
            if c >= len(color_to_edges):
                color_to_edges.append(set())
            color_to_edges[c].add(idx)
            
        return color, len(color_to_edges)

    def try_eliminate_colors(color, conflicts, n_edges):
        """
        Try to reduce the number of colors by eliminating color classes.
        Returns (new_color, new_n_colors).
        """
        if not color:
            return color, 0
            
        n_colors = max(color) + 1
        if n_colors == 0:
            return color, n_colors
        
        # Build color classes
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)
        
        improved = True
        # We try to eliminate colors from the highest index down, as they are often the most "lonely"
        while improved:
            improved = False
            for c in range(n_colors - 1, -1, -1):
                if not classes[c]:
                    continue
                    
                # Try to move all edges in class c to other classes
                temp_assign = {}
                success = True
                for e in classes[c]:
                    placed = False
                    # Try to place e in any other class
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        # Check if e conflicts with any edge in class c2
                        if not (conflicts[e] & classes[c2]):
                            temp_assign[e] = c2
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                
                if success:
                    # Apply the moves
                    for e, c2 in temp_assign.items():
                        classes[c].discard(e)
                        classes[c2].add(e)
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    # Restart the loop to check for further eliminations
                    break
                    
        return color, n_colors

    def local_polish(color, conflicts, n_edges, max_iters=100):
        """
        Fine-grained local search: try to move individual edges to lower colors.
        """
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors
            
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)
            
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Try to reduce the highest color first
            for c in range(n_colors - 1, -1, -1):
                if not classes[c]:
                    continue
                # Try to move one edge from class c to a lower class
                for e in list(classes[c]):
                    for c2 in range(c):
                        if not (conflicts[e] & classes[c2]):
                            # Move e to c2
                            classes[c].discard(e)
                            classes[c2].add(e)
                            color[e] = c2
                            improved = True
                            break
                    if improved:
                        # Check if we can reduce n_colors
                        if not classes[n_colors - 1]:
                            n_colors -= 1
                        break
                if improved:
                    break
            if not improved:
                break
                
        return color, n_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        n_edges = len(edges)
        conflicts = build_conflicts(edges, n)
        
        best_color = None
        best_count = float('inf')
        
        # 1. Greedy with multiple random restarts
        rng = random.Random(seed + 999)
        num_restarts = 30 # Increased slightly as conflict check is fast
        
        for r in range(num_restarts):
            if time.time() > deadline - 15.0:
                break
            order = list(range(n_edges))
            rng.shuffle(order)
            color, n_colors = greedy_color(order, conflicts, n_edges)
            
            # 2. Class Elimination (Stronger reduction)
            color, n_colors = try_eliminate_colors(color, conflicts, n_edges)
            
            # 3. Local Polish (Fine-tuning)
            color, n_colors = local_polish(color, conflicts, n_edges)
            
            if n_colors < best_count:
                best_count = n_colors
                best_color = color[:]
                
        # Also try sorted order as a baseline
        if time.time() < deadline - 15.0:
            order = list(range(n_edges))
            color, n_colors = greedy_color(order, conflicts, n_edges)
            color, n_colors = try_eliminate_colors(color, conflicts, n_edges)
            color, n_colors = local_polish(color, conflicts, n_edges)
            if n_colors < best_count:
                best_count = n_colors
                best_color = color[:]

        # Final safety check: ensure we have a valid coloring
        if best_color is None:
            # Fallback to a simple greedy if something went wrong (shouldn't happen)
            order = list(range(n_edges))
            best_color, _ = greedy_color(order, conflicts, n_edges)
            
        # Convert to list of colours in sorted edge order
        # best_color is already indexed by edge index which corresponds to sorted order
        results.append(best_color)
        
    return results
