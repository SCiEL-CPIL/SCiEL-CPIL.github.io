"""Strong edge colourings: Bitset Branch-and-Bound MIS + local refinement."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod

_DEADLINE_FRAC = 0.88
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

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_conflict_bitsets(edges, adj, n):
    """
    Build a list of integers where bit j in integer i is 1 if edge i and edge j conflict.
    Conflict means they share a vertex or are joined by an edge in the original graph.
    """
    m = len(edges)
    conflict = [0] * m
    
    for i in range(m):
        u, v = edges[i]
        mask = 0
        for j in range(i + 1, m):
            a, b = edges[j]
            # Check if they share a vertex
            if u == a or u == b or v == a or v == b:
                mask |= (1 << j)
                conflict[j] |= (1 << i)
                continue
            
            # Check if they are joined by an edge
            if a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                mask |= (1 << j)
                conflict[j] |= (1 << i)
        conflict[i] = mask
        
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
    
    # Heuristic ordering for B&B: 
    # We want to explore promising branches first.
    # A common heuristic is to pick vertices with high degree in the induced subgraph.
    
    def bb(node_set, current_set, current_size):
        nonlocal best_set, best_size, t0
        
        if time.time() - t0 > time_limit:
            return
        
        if node_set == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Upper bound: if current_size + popcount(node_set) <= best_size, prune
        if current_size + node_set.bit_count() <= best_size:
            return
            
        # Select a vertex to branch on
        # Heuristic: pick the vertex with the highest degree in the conflict graph 
        # restricted to node_set.
        # For speed, we approximate by picking the lowest set bit, 
        # relying on the bound to do the work.
        
        lsb = node_set & -node_set
        v_bit = lsb
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates are node_set without v and without any neighbors of v
        new_node_set_1 = node_set & ~(conflict[v] | v_bit)
        current_set.append(v)
        bb(new_node_set_1, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: Exclude v
        new_node_set_2 = node_set & ~v_bit
        bb(new_node_set_2, current_set, current_size)
        
    bb(candidates, [], 0)
    
    return best_set

def _local_improve_bitset(colour, conflict, m, max_iters=300):
    """Aggressive local search to reduce colors using bitset conflict representation."""
    if m == 0:
        return colour
    
    current_max = max(colour) + 1
    
    for _ in range(max_iters):
        improved = False
        
        # Phase 1: Try to recolor each edge to a smaller color
        for idx in range(m):
            current_color = colour[idx]
            for new_c in range(current_color):
                # Check if any neighbor has new_c
                # We can iterate through neighbors of idx
                mask = conflict[idx]
                # We need to check if any bit j in mask has colour[j] == new_c
                # This is O(m) per check, which is slow.
                # Alternative: maintain color masks?
                # For now, simple loop.
                can_recolor = True
                j = 0
                while mask:
                    if mask & 1:
                        if colour[j] == new_c:
                            can_recolor = False
                            break
                    mask >>= 1
                    j += 1
                if can_recolor:
                    colour[idx] = new_c
                    improved = True
                    break
        
        # Phase 2: Try to eliminate the highest color
        if current_max > 0:
            max_c = current_max - 1
            max_edges = [i for i in range(m) if colour[i] == max_c]
            for idx in max_edges:
                for new_c in range(max_c):
                    can_recolor = True
                    mask = conflict[idx]
                    j = 0
                    while mask:
                        if mask & 1:
                            if colour[j] == new_c:
                                can_recolor = False
                                break
                        mask >>= 1
                        j += 1
                    if can_recolor:
                        colour[idx] = new_c
                        current_max -= 1
                        improved = True
                        break
        
        if not improved:
            break
    
    return colour

def _solve_by_mis_bitset(edges, conflict, m, time_budget, seed=0):
    """Extract induced matchings via B&B MIS peeling."""
    if m == 0:
        return []
    
    t0 = time.time()
    colour = [-1] * m
    remaining = (1 << m) - 1
    color_idx = 0
    
    rng = _random_mod.Random(seed)
    
    while remaining:
        if time.time() - t0 > time_budget:
            # Fallback: Greedy coloring for remaining edges
            for e in range(m):
                if remaining & (1 << e):
                    used = set()
                    mask = conflict[e]
                    j = 0
                    while mask:
                        if mask & 1:
                            if colour[j] != -1:
                                used.add(colour[j])
                        mask >>= 1
                        j += 1
                    c = 0
                    while c in used:
                        c += 1
                    colour[e] = c
            break
        
        # Allocate time for this MIS step
        elapsed = time.time() - t0
        rem_time = time_budget - elapsed
        # Cap step time to 2s or 10% of remaining, whichever is smaller but at least 0.05s
        step_time = max(0.05, min(2.0, rem_time * 0.1))
        
        mis_nodes = _max_independent_set(remaining, conflict, step_time, rng)
        
        if not mis_nodes:
            # Fallback if B&B fails or times out with no result
            # Just pick the first available edge
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            mis_nodes = [v]
            
        for v in mis_nodes:
            colour[v] = color_idx
            remaining &= ~(1 << v)
            
        color_idx += 1
        
    return colour

def _solve_instance(edges, adj, n, time_budget):
    """Solve a single instance, returning a list of colours for edges in sorted order."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    conflict = _build_conflict_bitsets(edges, adj, n)
    
    best_colour = None
    best_count = float('inf')
    
    def _update_best(colour):
        nonlocal best_colour, best_count
        if colour is not None and all(c >= 0 for c in colour):
            count = max(colour) + 1 if colour else 0
            if count < best_count:
                best_count = count
                best_colour = colour[:]
    
    # Strategy 1: MIS peeling with multiple seeds
    # Use most of the budget for this structural approach
    mis_budget = time_budget * 0.7
    for seed in range(5):
        if time.time() - t0 > t0 + mis_budget:
            break
        colour = _solve_by_mis_bitset(edges, conflict, m, mis_budget / 5, seed=seed)
        # Local improvement
        colour = _local_improve_bitset(colour, conflict, m, max_iters=100)
        _update_best(colour)
    
    # Strategy 2: Greedy fallback if needed
    if best_colour is None:
        colour = [0] * m
        # Simple greedy
        for i in range(m):
            used = set()
            mask = conflict[i]
            j = 0
            while mask:
                if mask & 1:
                    if colour[j] != -1:
                        used.add(colour[j])
                mask >>= 1
                j += 1
            c = 0
            while c in used:
                c += 1
            colour[i] = c
        _update_best(colour)
    
    return best_colour

def construct():
    t_start = time.time()
    total_budget = _TOTAL_TIME_LIMIT * _DEADLINE_FRAC
    
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    results = []
    
    # Allocate budget: smaller graphs get less, larger gets more
    # (40,4,101): ~10s, (60,5,102): ~15s, (80,6,103): ~25s
    budget_fracs = [0.2, 0.3, 0.5]
    
    for i, spec in enumerate(specs):
        n, d, seed = spec
        edges = _bounded_degree_graph(n, d, seed)
        adj = _build_adj(edges, n)
        
        elapsed = time.time() - t_start
        remaining_total = total_budget - elapsed
        if remaining_total < 2.0:
            # Emergency fallback
            conflict = _build_conflict_bitsets(edges, adj, n)
            m = len(edges)
            colour = [0] * m
            for i_edge in range(m):
                used = set()
                mask = conflict[i_edge]
                j = 0
                while mask:
                    if mask & 1:
                        if colour[j] != -1:
                            used.add(colour[j])
                    mask >>= 1
                    j += 1
                c = 0
                while c in used:
                    c += 1
                colour[i_edge] = c
            results.append(colour)
            continue
        
        instance_budget = max(5.0, remaining_total * budget_fracs[i])
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
    
    return results
# EVOLVE-BLOCK-END
