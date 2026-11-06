"""Strong edge colourings of bounded-degree graphs: Bitset Branch-and-Bound MIS approach."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod

_DEADLINE_FRAC = 0.88  # Use 88% of the 150s budget for safety
_TOTAL_TIME_LIMIT = 150.0

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

def _build_conflict_bitsets(edges, adj):
    """
    Build a list of integers where bit j in integer i is 1 if edge i and edge j conflict.
    Conflict means they share a vertex or are joined by an edge in the original graph.
    """
    m = len(edges)
    conflict = [0] * m
    
    for i in range(m):
        u, v = edges[i]
        for j in range(i + 1, m):
            a, b = edges[j]
            # Check if they share a vertex
            if u == a or u == b or v == a or v == b:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                continue
            
            # Check if they are joined by an edge
            if a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
        
    return conflict

def _max_independent_set(candidates, conflict, time_limit, rng):
    """
    Finds a large independent set within the set of 'candidates' (an integer bitmask).
    Uses Branch and Bound with bitset operations.
    Returns a list of indices in the independent set.
    """
    if candidates == 0:
        return []
    
    best_set = []
    best_size = 0
    t0 = time.time()
    deadline = t0 + time_limit
    
    # Precompute: for each edge index, its conflict mask
    # Use a greedy initial solution to set a good lower bound
    # Greedy: repeatedly pick the edge with fewest conflicts in remaining set
    greedy_set = []
    rem = candidates
    while rem != 0:
        # Pick edge with minimum degree in rem
        best_v = -1
        best_deg = float('inf')
        tmp = rem
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            deg = (conflict[v] & rem).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            tmp ^= lsb
        if best_v < 0:
            break
        greedy_set.append(best_v)
        rem &= ~(conflict[best_v] | (1 << best_v))
    
    if greedy_set:
        best_size = len(greedy_set)
        best_set = greedy_set[:]
    
    def bb(node_set, current_set, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if node_set == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Upper bound: if current_size + popcount(node_set) <= best_size, prune
        if current_size + node_set.bit_count() <= best_size:
            return
            
        # Select a vertex to branch on: pick one with highest degree in node_set
        # This maximizes pruning in the "exclude" branch
        best_v = -1
        best_deg = -1
        tmp = node_set
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            deg = (conflict[v] & node_set).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            tmp ^= lsb
        
        v_bit = 1 << best_v
        
        # Branch 1: Include v (explored first since it builds the solution)
        new_node_set_1 = node_set & ~(conflict[best_v] | v_bit)
        current_set.append(best_v)
        bb(new_node_set_1, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: Exclude v
        new_node_set_2 = node_set & ~v_bit
        bb(new_node_set_2, current_set, current_size)
        
    bb(candidates, [], 0)
    
    return best_set

def _solve_instance_bitset(edges, adj, time_budget):
    """
    Solve a single instance using Bitset B&B MIS extraction.
    """
    m = len(edges)
    if m == 0:
        return []
    
    t_start = time.time()
    
    conflict = _build_conflict_bitsets(edges, adj)
    
    remaining = (1 << m) - 1
    colour = [-1] * m
    color_id = 0
    
    while remaining != 0:
        if time.time() - t_start > time_budget:
            # Fallback: Greedy coloring for remaining edges
            for i in range(m):
                if remaining & (1 << i):
                    used = set()
                    mask = conflict[i]
                    for j in range(m):
                        if (mask & (1 << j)) and colour[j] != -1:
                            used.add(colour[j])
                    c = 0
                    while c in used:
                        c += 1
                    colour[i] = c
            break
        
        elapsed = time.time() - t_start
        rem_time = time_budget - elapsed
        # Give more time to early (larger) MIS extractions
        frac_remaining = remaining.bit_count() / m
        step_time = max(0.05, min(8.0, rem_time * 0.15 * frac_remaining + 0.5))
        
        mis_nodes = _max_independent_set(remaining, conflict, step_time, _random_mod.Random(123))
        
        if not mis_nodes:
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            mis_nodes = [v]
            
        for v in mis_nodes:
            colour[v] = color_id
            remaining &= ~(1 << v)
            
        color_id += 1
        
    # Post-processing: try to reduce colors by recoloring
    # For each edge, try to move it to a lower color if possible
    if time.time() - t_start < time_budget * 0.8:
        _try_reduce_colors(colour, conflict, time_budget - (time.time() - t_start))
    
    return colour

def _try_reduce_colors(colour, conflict, time_budget):
    """Try to reduce the number of colors by recoloring edges to lower colors."""
    m = len(colour)
    if m == 0:
        return
    t_start = time.time()
    
    # Get list of color classes
    from collections import defaultdict
    color_classes = defaultdict(list)
    for i, c in enumerate(colour):
        color_classes[c].append(i)
    
    max_color = max(colour) if colour else 0
    
    # Try to recolor each edge to a lower color
    for i in range(m):
        if time.time() - t_start > time_budget:
            break
        orig_color = colour[i]
        if orig_color == 0:
            continue
        # Try colors 0 to orig_color-1
        for c in range(orig_color):
            # Check if edge i can take color c
            ok = True
            mask = conflict[i]
            for j in color_classes[c]:
                if mask & (1 << j):
                    ok = False
                    break
            if ok:
                colour[i] = c
                color_classes[orig_color].remove(i)
                color_classes[c].append(i)
                break

def _greedy_fallback(edges, conflict, adj):
    """Fast greedy fallback if time runs out."""
    m = len(edges)
    colour = [-1] * m
    # Order by degree (descending)
    degs = [conflict[i].bit_count() for i in range(m)]
    order = sorted(range(m), key=lambda i: -degs[i])
    for i in order:
        used = set()
        mask = conflict[i]
        for j in range(m):
            if (mask & (1 << j)) and colour[j] != -1:
                used.add(colour[j])
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    return colour

def construct():
    t_start = time.time()
    total_budget = _TOTAL_TIME_LIMIT * _DEADLINE_FRAC
    
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    results = []
    
    # Allocate time: give more to larger/harder instances
    # d=4: ~80 edges, d=5: ~150 edges, d=6: ~240 edges
    weights = [1.0, 2.0, 3.5]
    total_weight = sum(weights)
    
    for idx, spec in enumerate(specs):
        n, d, seed = spec
        edges = _bounded_degree_graph(n, d, seed)
        
        # Build adjacency for conflict construction
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        elapsed = time.time() - t_start
        remaining_budget = total_budget - elapsed
        
        if remaining_budget < 1.0:
            # Emergency fallback
            conflict = _build_conflict_bitsets(edges, adj)
            colour = _greedy_fallback(edges, conflict, adj)
            results.append(colour)
            continue
        
        allocated = (weights[idx] / total_weight) * remaining_budget
        # Ensure at least 2 seconds for any instance
        allocated = max(2.0, allocated)
        # But don't exceed remaining
        allocated = min(allocated, remaining_budget)
        
        colour_list = _solve_instance_bitset(edges, adj, allocated)
        results.append(colour_list)
        
    return results
# EVOLVE-BLOCK-END
