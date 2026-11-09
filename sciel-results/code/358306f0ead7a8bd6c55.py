"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random

def construct():
    # Deadline guard: finish well before the 150s wall-clock kill.
    _t0 = time.time()
    _DEADLINE = 135.0  # leave ~15s margin

    def _instance(n, d, seed):
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

    def _conflict(u1, v1, u2, v2, adj):
        """Check if two edges (u1,v1) and (u2,v2) conflict (share vertex or are joined)."""
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return True
        # Check if any endpoint of first edge is adjacent to any endpoint of second edge
        if u2 in adj[u1] or v2 in adj[u1]:
            return True
        if u2 in adj[v1] or v2 in adj[v1]:
            return True
        return False

    def _greedy_color(edges, adj, order=None):
        """
        Greedy coloring with a given edge order.
        Returns list of colours (one per edge in original sorted order).
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        if order is None:
            order = list(range(n_edges))
        
        colour = [-1] * n_edges
        colour_to_edges = defaultdict(set)
        max_c = 0
        
        for i in order:
            u, v = edges[i]
            # Try colors 0, 1, 2, ...
            for c in range(max_c + 1):
                ok = True
                for j in colour_to_edges[c]:
                    if _conflict(u, v, edges[j][0], edges[j][1], adj):
                        ok = False
                        break
                if ok:
                    colour[i] = c
                    colour_to_edges[c].add(i)
                    if c == max_c:
                        max_c += 1
                    break
        
        return colour

    def _local_search(colours, edges, adj, time_limit=8.0):
        """
        Improve coloring by moving edges to lower colors and compacting.
        Uses multiple passes with random restarts for stuck edges.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        # Work on a copy
        col = colours[:]
        
        # Build colour_to_edges
        colour_to_edges = defaultdict(set)
        for i, c in enumerate(col):
            colour_to_edges[c].add(i)
        
        def _can_assign(i, c):
            u, v = edges[i]
            for j in colour_to_edges[c]:
                if j == i:
                    continue
                if _conflict(u, v, edges[j][0], edges[j][1], adj):
                    return False
            return True
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            # Pass 1: try to move each edge to the lowest possible color
            for i in range(n_edges):
                cur_c = col[i]
                if cur_c == 0:
                    continue
                for c in range(cur_c):
                    if _can_assign(i, c):
                        colour_to_edges[cur_c].discard(i)
                        colour_to_edges[c].add(i)
                        col[i] = c
                        improved = True
                        break
            
            # Compact colors
            if improved:
                present = sorted(set(col))
                if present != list(range(len(present))):
                    mapping = {old: new for new, old in enumerate(present)}
                    for i in range(n_edges):
                        col[i] = mapping[col[i]]
                    new_cte = defaultdict(set)
                    for i, c in enumerate(col):
                        new_cte[c].add(i)
                    colour_to_edges = new_cte
        
        return col

    def _improve_with_swap(colours, edges, adj, time_limit=5.0):
        """
        Try swapping colors of pairs of edges to reduce total colors.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        col = colours[:]
        colour_to_edges = defaultdict(set)
        for i, c in enumerate(col):
            colour_to_edges[c].add(i)
        
        max_c = max(col) if col else 0
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            for i in range(n_edges):
                cur_c = col[i]
                if cur_c == 0:
                    continue
                u, v = edges[i]
                
                # Try moving edge i to a lower color
                for c in range(cur_c):
                    if _can_assign_swap(i, c, col, colour_to_edges, edges, adj):
                        colour_to_edges[cur_c].discard(i)
                        colour_to_edges[c].add(i)
                        col[i] = c
                        improved = True
                        break
                
                # Try swapping with an edge in a lower color
                if not improved:
                    for c in range(cur_c):
                        for j in list(colour_to_edges[c]):
                            a, b = edges[j]
                            # Check if swapping works
                            # Remove i from cur_c and j from c temporarily
                            colour_to_edges[cur_c].discard(i)
                            colour_to_edges[c].discard(j)
                            
                            # Check if i can go to c and j can go to cur_c
                            ok_i = True
                            for k in colour_to_edges[c]:
                                if _conflict(u, v, edges[k][0], edges[k][1], adj):
                                    ok_i = False
                                    break
                            if ok_i:
                                ok_j = True
                                for k in colour_to_edges[cur_c]:
                                    if _conflict(a, b, edges[k][0], edges[k][1], adj):
                                        ok_j = False
                                        break
                            
                            if ok_i and ok_j:
                                # Do the swap
                                colour_to_edges[c].add(i)
                                colour_to_edges[cur_c].add(j)
                                col[i] = c
                                col[j] = cur_c
                                improved = True
                            else:
                                # Undo
                                colour_to_edges[cur_c].add(i)
                                colour_to_edges[c].add(j)
                            break
                        if improved:
                            break
            
            # Compact
            if improved:
                present = sorted(set(col))
                if present != list(range(len(present))):
                    mapping = {old: new for new, old in enumerate(present)}
                    for i in range(n_edges):
                        col[i] = mapping[col[i]]
                    new_cte = defaultdict(set)
                    for i, c in enumerate(col):
                        new_cte[c].add(i)
                    colour_to_edges = new_cte
        
        return col

    def _can_assign_swap(i, c, col, colour_to_edges, edges, adj):
        u, v = edges[i]
        for j in colour_to_edges[c]:
            if j == i:
                continue
            if _conflict(u, v, edges[j][0], edges[j][1], adj):
                return False
        return True

    def _smart_order(edges, adj):
        """
        Order edges by degree (sum of endpoint degrees) descending.
        High-degree edges first tend to use fewer colors.
        """
        n_edges = len(edges)
        order = list(range(n_edges))
        # Sort by sum of endpoint degrees descending
        order.sort(key=lambda i: len(adj[edges[i][0]]) + len(adj[edges[i][1]]), reverse=True)
        return order

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = _instance(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Try multiple orderings and pick the best
        best_colours = None
        best_num_colors = float('inf')
        
        # Order 1: natural order
        colours1 = _greedy_color(edges, adj, None)
        num1 = max(colours1) + 1 if colours1 else 0
        if num1 < best_num_colors:
            best_colours = colours1
            best_num_colors = num1
        
        # Order 2: degree-based
        order2 = _smart_order(edges, adj)
        colours2 = _greedy_color(edges, adj, order2)
        num2 = max(colours2) + 1 if colours2 else 0
        if num2 < best_num_colors:
            best_colours = colours2
            best_num_colors = num2
        
        # Order 3: reverse natural
        order3 = list(range(len(edges) - 1, -1, -1))
        colours3 = _greedy_color(edges, adj, order3)
        num3 = max(colours3) + 1 if colours3 else 0
        if num3 < best_num_colors:
            best_colours = colours3
            best_num_colors = num3
        
        # Local search improvement
        if time.time() < _DEADLINE - 10:
            best_colours = _local_search(best_colours, edges, adj, time_limit=6.0)
        
        # Swap-based improvement
        if time.time() < _DEADLINE - 5:
            best_colours = _improve_with_swap(best_colours, edges, adj, time_limit=4.0)
        
        out.append(best_colours)

    return out
# EVOLVE-BLOCK-END
