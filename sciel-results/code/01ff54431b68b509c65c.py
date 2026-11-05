"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    def build_data(edges, n):
        """Build adjacency and conflict structure."""
        m = len(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Edge index mapping
        edge_set = set(edges)
        edge_idx = {e: i for i, e in enumerate(edges)}
        
        # For each edge, precompute the set of "near" vertices (endpoints + neighbors)
        near = []
        for u, v in edges:
            ns = set([u, v])
            ns.update(adj[u])
            ns.update(adj[v])
            near.append(ns)
        
        # Conflict: two edges conflict if they share a vertex or are joined by an edge
        # i.e., if any vertex of edge i is in near of edge j (excluding the edge's own vertices handled by sharing)
        # Actually: edges e1=(a,b) and e2=(c,d) conflict if:
        #   - they share a vertex: {a,b} & {c,d} != empty
        #   - OR there's an edge between them: (a,c) in edge_set or (a,d) in edge_set or (b,c) in edge_set or (b,d) in edge_set
        # This is equivalent to: {a,b} intersects near(e2) where near includes endpoints and neighbors
        # But we need to be careful: if they share a vertex, that's already a conflict.
        # The "joined by an edge" means there exists x in {a,b}, y in {c,d} such that (x,y) is an edge in the graph
        # and x != y (which is always true since they're different vertices for different edges that don't share)
        
        # Precompute conflict bitsets for each edge
        conflicts = []
        for i, (u, v) in enumerate(edges):
            conflict_set = set()
            for j, (x, y) in enumerate(edges):
                if i == j:
                    continue
                # Check if edges share a vertex
                if (u == x or u == y or v == x or v == y):
                    conflict_set.add(j)
                    continue
                # Check if joined by an edge
                if (u, x) in edge_set or (u, y) in edge_set or (v, x) in edge_set or (v, y) in edge_set:
                    conflict_set.add(j)
            conflicts.append(conflict_set)
        
        return m, adj, near, conflicts, edge_set, edge_idx

    def greedy_max_induced_matching(edges, conflicts, used_edges):
        """Find a maximum induced matching greedily from unused edges."""
        available = [i for i in range(len(edges)) if i not in used_edges]
        if not available:
            return []
        
        # Greedy: pick edge that conflicts with fewest other available edges
        matching = []
        removed = set()
        
        while available:
            # Find edge with minimum conflicts among available
            best = None
            best_conf_count = float('inf')
            for i in available:
                if i in removed:
                    continue
                conf_count = len(conflicts[i] & set(available) - removed)
                if conf_count < best_conf_count:
                    best_conf_count = conf_count
                    best = i
            if best is None:
                break
            matching.append(best)
            removed.add(best)
            # Remove all edges that conflict with best
            for j in conflicts[best]:
                removed.add(j)
            available = [i for i in available if i not in removed]
        
        return matching

    def count_colors(coloring):
        if not coloring:
            return 0
        return max(coloring.values()) + 1

    def local_search(coloring, edges, conflicts, n, max_iter=5000):
        """Try to reduce colors via edge swaps and reassignments."""
        m = len(edges)
        current = coloring[:]
        current_num_colors = count_colors(current)
        
        rng = random.Random(42)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Try to reassign each edge to a lower color if possible
            for i in range(m):
                if time.time() > deadline - 5:
                    return current, current_num_colors
                c = current[i]
                if c == 0:
                    continue
                # Check if edge i can be moved to any color < c
                for new_c in range(c):
                    # Check if edge i conflicts with any edge of color new_c
                    can_move = True
                    for j in conflicts[i]:
                        if current[j] == new_c:
                            can_move = False
                            break
                    if can_move:
                        current[i] = new_c
                        improved = True
                        break
            
            # Try to merge color classes: if color c has no edges, shift higher colors down
            # Actually, let's try a different approach: try to reduce the max color
            # Find the highest color used
            if current_num_colors > 1:
                max_c = max(current)
                # Try to move edges from max_c to lower colors
                for i in range(m):
                    if time.time() > deadline - 5:
                        break
                    if current[i] != max_c:
                        continue
                    for new_c in range(max_c):
                        can_move = True
                        for j in conflicts[i]:
                            if current[j] == new_c:
                                can_move = False
                                break
                        if can_move:
                            current[i] = new_c
                            improved = True
                            break
            
            # Recompute num colors
            new_num = count_colors(current)
            if new_num < current_num_colors:
                current_num_colors = new_num
                improved = True
        
        return current, current_num_colors

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        m, adj, near, conflicts, edge_set, edge_idx = build_data(edges, n)
        
        # Initial greedy coloring: assign each edge a new color class (maximum induced matching per color)
        coloring = [0] * m
        used_edges = set()
        color_count = 0
        
        for _ in range(m):
            if time.time() > deadline - 5:
                break
            matching = greedy_max_induced_matching(edges, conflicts, used_edges)
            if not matching:
                break
            for i in matching:
                coloring[i] = color_count
                used_edges.add(i)
            color_count += 1
        
        # Ensure all edges are colored (should be, but safety)
        for i in range(m):
            if coloring[i] == 0 and i not in used_edges:
                # Assign a new color
                coloring[i] = color_count
                color_count += 1
                used_edges.add(i)
        
        best_coloring = coloring[:]
        best_num = count_colors(best_coloring)
        
        # Local search
        coloring, num = local_search(coloring, edges, conflicts, n, max_iter=10000)
        if num < best_num:
            best_coloring = coloring
            best_num = num
        
        return best_coloring

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 10:
            # Not enough time, use a quick greedy for remaining
            edges = instance(*spec)
            m = len(edges)
            coloring = [0] * m
            used = set()
            cc = 0
            for i in range(m):
                coloring[i] = cc
                cc += 1
            out.append(coloring)
            continue
        
        coloring = solve_instance(*spec)
        out.append(coloring)
    
    return out
# EVOLVE-BLOCK-END
