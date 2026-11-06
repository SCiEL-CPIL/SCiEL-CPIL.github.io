"""Strong edge colourings of bounded-degree graphs: branch-and-bound MIS + greedy hybrid."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod
from collections import defaultdict
from functools import lru_cache

_DEADLINE_FRAC = 0.82  # use 82% of the 150s budget for safety

def _bounded_degree_graph(n, d, seed):
    """Reproduce the verifier's graph generation exactly."""
    rng = _random_mod.Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u = rng.randrange(n)
        v = rng.randrange(n)
        e = (min(u, v), max(u, v))
        if u != v and degree[u] < d and degree[v] < d and e not in edges:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_conflict(edges, adj, n):
    """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
    m = len(edges)
    conflict = [set() for _ in range(m)]
    
    # For each pair of edges, check if they conflict
    # O(m^2) is fine for m ~ 150
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            # Share a vertex?
            if a == c or a == d or b == c or b == d:
                conflict[i].add(j)
                conflict[j].add(i)
                continue
            # Joined by an edge?
            # Check if any endpoint of edge j is adjacent to any endpoint of edge i
            if c in adj[a] or c in adj[b] or d in adj[a] or d in adj[b]:
                conflict[i].add(j)
                conflict[j].add(i)
    
    return conflict

def _greedy_coloring(edges, conflict, order=None):
    """Greedy coloring in the given order. Returns list of colours."""
    m = len(edges)
    if order is None:
        order = list(range(m))
    colour = [-1] * m
    
    # For each edge in order, find the smallest available colour
    for idx in order:
        used = set()
        for j in conflict[idx]:
            if colour[j] != -1:
                used.add(colour[j])
        c = 0
        while c in used:
            c += 1
        colour[idx] = c
    
    return colour

def _count_colors(colour):
    """Return the number of distinct colours used."""
    if not colour:
        return 0
    return max(colour) + 1

def _try_recolor(colour, idx, new_color, conflict):
    """Check if we can assign new_color to edge idx."""
    for j in conflict[idx]:
        if colour[j] == new_color:
            return False
    return True

def _local_improve(colour, conflict, max_iters=200):
    """Try to reduce the number of colours by recoloring edges."""
    m = len(colour)
    if m == 0:
        return colour
    
    current_max = max(colour) + 1
    
    for _ in range(max_iters):
        improved = False
        # Try to recolor each edge to a smaller colour
        for idx in range(m):
            current_color = colour[idx]
            for new_c in range(current_color):
                if _try_recolor(colour, idx, new_c, conflict):
                    colour[idx] = new_c
                    improved = True
                    break
        
        # Try to reduce the overall max colour
        if current_max > 0:
            max_c = current_max - 1
            max_edges = [i for i in range(m) if colour[i] == max_c]
            for idx in max_edges:
                for new_c in range(max_c):
                    if _try_recolor(colour, idx, new_c, conflict):
                        colour[idx] = new_c
                        current_max -= 1
                        improved = True
                        break
        
        if not improved:
            break
    
    return colour

def _color_merge_improve(colour, conflict, max_iters=50):
    """Try to merge color classes by recoloring all edges of one color to another."""
    m = len(colour)
    if m == 0:
        return colour
    
    current_max = max(colour) + 1
    
    for _ in range(max_iters):
        improved = False
        
        # Try to merge the highest color into a lower color
        for target_c in range(current_max - 1, -1, -1):
            if target_c >= current_max:
                continue
            edges_in_color = [i for i in range(m) if colour[i] == target_c]
            if not edges_in_color:
                continue
            
            # Try to move all edges of this color to another color
            for new_c in range(target_c):
                # Check if all edges can be moved
                can_move = True
                for idx in edges_in_color:
                    for j in conflict[idx]:
                        if colour[j] == new_c:
                            can_move = False
                            break
                    if not can_move:
                        break
                if can_move:
                    for idx in edges_in_color:
                        colour[idx] = new_c
                    improved = True
                    break
            
            if improved:
                current_max = max(colour) + 1
                break
        
        if not improved:
            break
    
    return colour

def _edge_ordering_largest_first(edges, adj, n):
    """Order edges by sum of endpoint degrees, largest first."""
    m = len(edges)
    edge_deg = [len(adj[u]) + len(adj[v]) for u, v in edges]
    return sorted(range(m), key=lambda i: -edge_deg[i])

def _dsatur_ordering(edges, conflict, adj, n):
    """
    DSATUR heuristic:
    1. Start with the edge with highest degree in conflict graph (or sum of endpoint degrees).
    2. At each step, pick the uncolored edge with the highest number of distinct colors
       among its colored neighbors. Tie-break by degree in conflict graph.
    Returns a list of edge indices in the order they should be colored.
    """
    m = len(edges)
    if m == 0:
        return []
    
    # Precompute conflict degrees
    conf_deg = [len(conflict[i]) for i in range(m)]
    
    uncolored = set(range(m))
    order = []
    colors = [-1] * m
    
    # Initial choice: highest conflict degree
    if uncolored:
        # Tie break by sum of endpoint degrees if needed, but conf_deg is usually sufficient
        first = max(uncolored, key=lambda i: (conf_deg[i], len(adj[edges[i][0]]) + len(adj[edges[i][1]])))
        order.append(first)
        uncolored.remove(first)
        colors[first] = 0
        
    while uncolored:
        best_edge = None
        best_sat = -1
        best_deg = -1
        
        for i in uncolored:
            # Count distinct colors in neighbors
            sat_colors = set()
            for j in conflict[i]:
                if colors[j] != -1:
                    sat_colors.add(colors[j])
            sat = len(sat_colors)
            
            # Tie break: higher conflict degree is better (more constrained)
            if sat > best_sat or (sat == best_sat and conf_deg[i] > best_deg):
                best_sat = sat
                best_deg = conf_deg[i]
                best_edge = i
        
        order.append(best_edge)
        uncolored.remove(best_edge)
        # Assign a temporary color for the heuristic (smallest available)
        used = set()
        for j in conflict[best_edge]:
            if colors[j] != -1:
                used.add(colors[j])
        c = 0
        while c in used:
            c += 1
        colors[best_edge] = c
            
    return order

def _solve_instance(edges, adj, n, time_budget):
    """Solve a single instance, returning a list of colours for edges in sorted order."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    
    # Precompute conflict graph
    conflict = _build_conflict(edges, adj, n)
    
    # Convert conflict to bitsets for fast MIS operations
    conflict_bits = [0] * m
    for i in range(m):
        bits = 0
        for j in conflict[i]:
            bits |= (1 << j)
        conflict_bits[i] = bits
    
    best_colour = None
    best_count = float('inf')
    
    def _mis_greedy(remaining_bits, deadline):
        """
        Greedily extract maximum independent set from remaining edges.
        Returns (independent_set_indices, remaining_bits_after)
        Uses bitset operations for speed.
        """
        if remaining_bits == 0:
            return [], 0
        
        # Find edge with minimum degree in the induced subgraph
        # to maximize the chance of a large independent set
        best_edge = -1
        best_deg = m + 1
        temp = remaining_bits
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            temp ^= lsb
            
            # Count neighbors in remaining set
            neighbors = conflict_bits[idx] & remaining_bits
            deg = bin(neighbors).count('1')
            
            if deg < best_deg:
                best_deg = deg
                best_edge = idx
                if deg == 0:  # Isolated vertex, take it immediately
                    break
        
        if best_edge == -1:
            return [], remaining_bits
        
        # Take this edge and all its non-neighbors
        independent = [best_edge]
        # Remove best_edge and its neighbors from remaining
        new_remaining = remaining_bits & ~(conflict_bits[best_edge] | (1 << best_edge))
        
        # Continue greedily
        sub_independent, final_remaining = _mis_greedy(new_remaining, deadline)
        independent.extend(sub_independent)
        
        return independent, final_remaining
    
    def _branch_and_bound_mis(remaining_bits, current_set, best_size, deadline):
        """
        Branch and bound for maximum independent set.
        Returns the largest independent set found.
        """
        if time.time() > deadline:
            return current_set[:]
        
        if remaining_bits == 0:
            if len(current_set) > best_size[0]:
                best_size[0] = len(current_set)
            return current_set[:]
        
        # Upper bound: greedy coloring of the remaining graph
        # gives an upper bound on the size of the independent set
        def _upper_bound(bits):
            if bits == 0:
                return 0
            # Simple upper bound: number of vertices / max degree + 1
            # Better: use greedy coloring
            count = 0
            temp = bits
            while temp:
                # Greedily pick a vertex
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                temp ^= lsb
                # Remove this vertex and its neighbors
                temp &= ~(conflict_bits[idx] | (1 << idx))
                count += 1
            return count
        
        ub = _upper_bound(remaining_bits)
        if len(current_set) + ub <= best_size[0]:
            return current_set[:]
        
        # Choose a vertex to branch on
        # Use the vertex with highest degree in the remaining subgraph
        best_vertex = -1
        best_deg = -1
        temp = remaining_bits
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            temp ^= lsb
            
            neighbors = conflict_bits[idx] & remaining_bits
            deg = bin(neighbors).count('1')
            
            if deg > best_deg:
                best_deg = deg
                best_vertex = idx
        
        if best_vertex == -1:
            return current_set[:]
        
        # Branch 1: Include best_vertex
        current_set.append(best_vertex)
        new_remaining = remaining_bits & ~(conflict_bits[best_vertex] | (1 << best_vertex))
        _branch_and_bound_mis(new_remaining, current_set, best_size, deadline)
        current_set.pop()
        
        # Branch 2: Exclude best_vertex
        new_remaining = remaining_bits & ~(1 << best_vertex)
        _branch_and_bound_mis(new_remaining, current_set, best_size, deadline)
        
        return current_set[:]
    
    # Strategy 1: Use branch-and-bound to find maximum independent sets iteratively
    # This directly minimizes the number of colors
    def _solve_with_mis(deadline):
        """
        Solve by repeatedly extracting maximum independent sets.
        Each independent set becomes one color class.
        """
        colour = [-1] * m
        remaining_bits = (1 << m) - 1  # All edges available
        color_idx = 0
        
        while remaining_bits != 0:
            if time.time() > deadline:
                break
            
            # Find a maximum independent set in the remaining graph
            best_size = [0]
            best_set = _branch_and_bound_mis(remaining_bits, [], best_size, deadline)
            
            if not best_set:
                # Fallback: just take one edge
                lsb = remaining_bits & (-remaining_bits)
                idx = lsb.bit_length() - 1
                best_set = [idx]
            
            # Assign this color
            for idx in best_set:
                colour[idx] = color_idx
            
            # Remove these edges from remaining
            for idx in best_set:
                remaining_bits &= ~(1 << idx)
                # Also remove all neighbors? No, they can still be in other color classes
                # We only remove the selected edges themselves
            
            color_idx += 1
        
        return colour
    
    # Try MIS-based approach
    mis_deadline = t0 + time_budget * 0.6
    if time.time() < mis_deadline:
        try:
            colour = _solve_with_mis(mis_deadline)
            if all(c >= 0 for c in colour):
                count = _count_colors(colour)
                if count < best_count:
                    best_count = count
                    best_colour = colour[:]
        except Exception:
            pass
    
    # Strategy 2: DSATUR ordering (keep as fallback and for comparison)
    if time.time() - t0 < time_budget * 0.8:
        order = _dsatur_ordering(edges, conflict, adj, n)
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=200)
        colour = _color_merge_improve(colour, conflict, max_iters=50)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
            
    # Strategy 3: Largest First
    if time.time() - t0 < time_budget * 0.8:
        order = _edge_ordering_largest_first(edges, adj, n)
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=200)
        colour = _color_merge_improve(colour, conflict, max_iters=50)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]

    # Strategy 4: Randomized Restarts
    rng = _random_mod.Random(12345)
    max_random = 80
    for attempt in range(max_random):
        if time.time() - t0 > time_budget:
            break
        order = list(range(m))
        rng.shuffle(order)
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=100)
        colour = _color_merge_improve(colour, conflict, max_iters=30)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
            
    if best_colour is None:
        best_colour = _greedy_coloring(edges, conflict)
    
    return best_colour

def construct():
    t_start = time.time()
    total_budget = 150.0 * _DEADLINE_FRAC
    
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    results = []
    
    for spec in specs:
        n, d, seed = spec
        edges = _bounded_degree_graph(n, d, seed)
        adj = _build_adj(edges, n)
        
        remaining = total_budget - (time.time() - t_start)
        if remaining < 2.0:
            # Not enough time; use a very fast fallback
            conflict = _build_conflict(edges, adj, n)
            colour = _greedy_coloring(edges, conflict)
            results.append(colour)
            continue
        
        # Allocate budget proportional to remaining, with a minimum
        instance_budget = max(20.0, remaining * 0.4)
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
    
    return results
# EVOLVE-BLOCK-END
