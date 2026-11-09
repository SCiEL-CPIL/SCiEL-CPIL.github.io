"""Strong edge colouring of bounded-degree graphs via greedy + backtracking + local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 5  # seconds to finish before 150s wall-clock

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Generate a random bounded-degree graph
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges), degree

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def precompute_conflicts(edges, adj):
        """
        Precompute the set of edges that conflict with each edge.
        Two edges conflict if they share a vertex or are joined by an edge (distance 2).
        """
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        
        # Build edge lists per vertex for fast lookup
        edges_at_vertex = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            edges_at_vertex[u].append(i)
            edges_at_vertex[v].append(i)
        
        # For each edge, find conflicting edges
        for i, (u1, v1) in enumerate(edges):
            # Edges sharing vertex u1 or v1
            for e2_idx in edges_at_vertex[u1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
                    conflicts[e2_idx].add(i)
            for e2_idx in edges_at_vertex[v1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
                    conflicts[e2_idx].add(i)
            
            # Distance 2 edges (joined by an edge)
            # Check neighbors of u1 and v1
            for neighbor in adj[u1]:
                if neighbor == v1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx != i and (u1, v1) != edges[e2_idx] and (v1, u1) != edges[e2_idx]:
                        # Check if e2_idx doesn't share a vertex with e1
                        u2, v2 = edges[e2_idx]
                        if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                            conflicts[i].add(e2_idx)
                            conflicts[e2_idx].add(i)
            for neighbor in adj[v1]:
                if neighbor == u1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx != i and (u1, v1) != edges[e2_idx] and (v1, u1) != edges[e2_idx]:
                        u2, v2 = edges[e2_idx]
                        if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                            conflicts[i].add(e2_idx)
                            conflicts[e2_idx].add(i)
        
        return conflicts

    def greedy_coloring(edges, conflicts, order):
        """
        Greedy coloring using precomputed conflicts.
        order: list of edge indices in the order to process.
        """
        n_edges = len(edges)
        color = [-1] * n_edges
        color_classes = [set() for _ in range(n_edges)] # Max possible colors
        
        for e_idx in order:
            # Find taken colors
            taken = set()
            for c_idx in conflicts[e_idx]:
                if color[c_idx] != -1:
                    taken.add(color[c_idx])
            
            c = 0
            while c in taken:
                c += 1
            color[e_idx] = c
            color_classes[c].add(e_idx)
            
        return color, color_classes

    def local_search(edges, conflicts, color, color_classes, max_iter=10000):
        """
        Local search: try to move edges to lower colors.
        Returns improved color and color_classes.
        """
        n_edges = len(edges)
        current_max_color = max(color) + 1 if color else 0
        rng = random.Random(42)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Process edges in random order
            order = list(range(n_edges))
            rng.shuffle(order)
            
            for e_idx in order:
                c_current = color[e_idx]
                if c_current == 0:
                    continue
                
                # Try to move to a lower color
                for new_c in range(c_current):
                    # Check if e_idx conflicts with any edge in color_classes[new_c]
                    can_move = True
                    if color_classes[new_c]:
                        # Quick check: if any conflict of e_idx is in new_c's class
                        for c_idx in conflicts[e_idx]:
                            if c_idx in color_classes[new_c]:
                                can_move = False
                                break
                    
                    if can_move:
                        # Move e_idx to new_c
                        color_classes[c_current].remove(e_idx)
                        color_classes[new_c].add(e_idx)
                        color[e_idx] = new_c
                        improved = True
                        # Update max color if we removed the last edge of the highest color
                        if c_current == current_max_color - 1 and not color_classes[c_current]:
                            current_max_color -= 1
                        break
            
            if not improved:
                break
        
        return color, color_classes, current_max_color

    def backtrack_reduce_color(edges, conflicts, color, max_color, time_limit=10):
        """
        Try to reduce the number of colors by backtracking.
        Attempts to recolor all edges using at most max_color colors.
        Returns True if successful, False otherwise.
        """
        n_edges = len(edges)
        start_time = time.time()
        
        # Order edges by number of conflicts (most constrained first)
        edge_order = sorted(range(n_edges), key=lambda i: len(conflicts[i]), reverse=True)
        
        # Reverse the order for backtracking (process least constrained last)
        # Actually, we want to process most constrained first in forward pass
        # But for backtracking, we process in reverse to undo efficiently
        backtrack_order = list(reversed(edge_order))
        
        # Try to find a valid coloring with max_color colors
        new_color = [-1] * n_edges
        color_classes = [set() for _ in range(max_color)]
        
        def can_assign(e_idx, c):
            """Check if edge e_idx can be assigned color c."""
            for conflict_idx in conflicts[e_idx]:
                if new_color[conflict_idx] == c:
                    return False
            return True
        
        def backtrack(pos):
            """Recursively try to assign colors to edges in backtrack_order."""
            if time.time() - start_time > time_limit:
                return False
            
            if pos == n_edges:
                return True
            
            e_idx = backtrack_order[pos]
            
            for c in range(max_color):
                if can_assign(e_idx, c):
                    new_color[e_idx] = c
                    color_classes[c].add(e_idx)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    new_color[e_idx] = -1
                    color_classes[c].discard(e_idx)
            
            return False
        
        if backtrack(0):
            # Update the original color array
            for i in range(n_edges):
                color[i] = new_color[i]
            return True
        return False

    def optimize_instance(n, d, seed, time_budget):
        """
        Optimize coloring for a single instance within time budget.
        """
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = precompute_conflicts(edges, adj)
        n_edges = len(edges)
        
        best_color_list = None
        best_max_color = float('inf')
        
        elapsed_start = time.time()
        
        # Define multiple orderings
        orderings = []
        
        # 1. Natural order
        orderings.append(list(range(n_edges)))
        
        # 2. Reverse natural
        orderings.append(list(reversed(range(n_edges))))
        
        # 3. Degree-based (high degree vertices first)
        edge_degrees = [degree[u] + degree[v] for u, v in edges]
        orderings.append(sorted(range(n_edges), key=lambda i: edge_degrees[i], reverse=True))
        
        # 4. Random shuffles
        for s in range(15):
            rng = random.Random(seed + 1000 + s)
            shuffled = list(range(n_edges))
            rng.shuffle(shuffled)
            orderings.append(shuffled)
        
        # 5. Most conflicts first
        conflict_counts = [len(conflicts[i]) for i in range(n_edges)]
        orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i], reverse=True))
        
        # 6. Least conflicts first
        orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i]))
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            if time.time() - elapsed_start > time_budget:
                break
            
            # Greedy
            color, color_classes = greedy_coloring(edges, conflicts, ordering)
            current_max = max(color) + 1 if color else 0
            
            if current_max < best_max_color:
                best_max_color = current_max
                best_color_list = list(color)
            
            # Local Search
            remaining = time_budget - (time.time() - elapsed_start)
            max_iter = max(50, int(remaining * 300))
            
            color, color_classes, current_max = local_search(edges, conflicts, color, color_classes, max_iter=max_iter)
            
            if current_max < best_max_color:
                best_max_color = current_max
                best_color_list = list(color)
        
        # Try backtracking to reduce colors further
        if best_color_list is not None and time.time() < deadline:
            remaining = min(time_budget - (time.time() - elapsed_start), 8.0)
            if remaining > 2.0:
                # Try reducing by 1, 2, 3 colors
                for reduction in range(1, 4):
                    target_colors = best_max_color - reduction
                    if target_colors < 1:
                        break
                    
                    # Make a copy to try backtracking
                    test_color = list(best_color_list)
                    if backtrack_reduce_color(edges, conflicts, test_color, target_colors, time_limit=remaining - 1.0):
                        best_max_color = target_colors
                        best_color_list = test_color
                        remaining = min(remaining - 1.0, 5.0)
                    else:
                        break
        
        # If no coloring found (should not happen), fallback to simple greedy
        if best_color_list is None:
            color, _ = greedy_coloring(edges, conflicts, list(range(n_edges)))
            best_color_list = list(color)
            best_max_color = max(best_color_list) + 1 if best_color_list else 0
            
        return best_color_list, best_max_color

    results = []
    
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    remaining_time = deadline - time.time()
    
    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            # Emergency fallback
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            conflicts = precompute_conflicts(edges, adj)
            color, _ = greedy_coloring(edges, conflicts, list(range(len(edges))))
            results.append(color)
            continue
        
        # Allocate more time to harder instances
        if idx == 0:
            time_budget = min(remaining_time * 0.3, 15.0)
        elif idx == 1:
            time_budget = min(remaining_time * 0.35, 20.0)
        else:
            time_budget = min(remaining_time * 0.35, 25.0)
        
        coloring, _ = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        remaining_time = deadline - time.time()
    
    return results

# EVOLVE-BLOCK-END
