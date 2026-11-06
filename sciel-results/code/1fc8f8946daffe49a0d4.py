"""Strong edge colourings of bounded-degree graphs: optimized greedy + local improvement."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod
from collections import defaultdict

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

def _local_improve(colour, conflict, max_iters=100):
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
    
    best_colour = None
    best_count = float('inf')
    
    # 1. Try DSATUR ordering
    if time.time() - t0 < time_budget:
        order = _dsatur_ordering(edges, conflict, adj, n)
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=100)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
            
    # 2. Try Largest First
    if time.time() - t0 < time_budget * 0.8:
        order = _edge_ordering_largest_first(edges, adj, n)
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=100)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]

    # 3. Try Randomized Restarts with DSATUR-like tie-breaking or Random Shuffle
    # We can just shuffle the input to _dsatur_ordering's tie-breakers or just shuffle edges?
    # Better: Randomize the initial choice in DSATUR or just use random orderings.
    # Let's use random orderings but with better local search.
    
    rng = _random_mod.Random(12345)
    max_random = 50
    for attempt in range(max_random):
        if time.time() - t0 > time_budget:
            break
        order = list(range(m))
        rng.shuffle(order)
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=50)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
            
        # If we hit a very low count, we can stop or continue to try to beat it
        # The conjecture is 5d^2/4. For d=4, 20. d=5, 31. d=6, 45.
        # If we are close, keep trying.
        
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
        instance_budget = max(15.0, remaining * 0.35)
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
    
    return results
# EVOLVE-BLOCK-END
