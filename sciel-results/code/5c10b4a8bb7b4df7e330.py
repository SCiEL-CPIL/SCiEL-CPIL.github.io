"""Strong edge colourings of bounded-degree graphs: Bitset Branch-and-Bound MIS approach."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod

_DEADLINE_FRAC = 0.85  # Use 85% of the 150s budget for safety
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
        mask = 0
        for j in range(i + 1, m):
            a, b = edges[j]
            # Check if they share a vertex
            if u == a or u == b or v == a or v == b:
                mask |= (1 << j)
                conflict[j] |= (1 << i)
                continue
            
            # Check if they are joined by an edge
            # i=(u,v), j=(a,b). Check if (u,a), (u,b), (v,a), (v,b) are in adj
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
    # A common heuristic is to pick vertices with high degree in the induced subgraph, 
    # or low degree. For MIS, picking high degree vertices often leads to faster pruning 
    # if we branch on "include" vs "exclude".
    
    # However, a simpler and very effective heuristic for small graphs is to just 
    # pick the vertex with the highest index or lowest index, but randomized restarts 
    # can help. Let's use a standard recursive B&B.
    
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
        # popcount is fast in Python 3.8+ via int.bit_count()
        if current_size + node_set.bit_count() <= best_size:
            return
            
        # Select a vertex to branch on
        # Heuristic: pick a vertex from node_set. 
        # To improve pruning, we can pick a vertex with high degree in the induced subgraph.
        # But computing degree for every node in the subgraph is expensive.
        # Let's just pick the lowest set bit for simplicity and speed, 
        # relying on the bound to do the work.
        # Or better: pick a random vertex from the candidates to avoid systematic bias?
        # Actually, for B&B, picking a vertex with many neighbors in the remaining set 
        # is good for the "exclude" branch (removes many nodes) but bad for "include".
        
        # Let's try picking the vertex with the highest degree in the conflict graph 
        # restricted to node_set.
        # We can approximate this or just pick the first one. 
        # Given the small size, let's pick the lowest index for determinism in single run, 
        # but we will call this function multiple times with different random shuffles?
        # No, the B&B structure is fixed. Let's use a simple heuristic: 
        # Pick the vertex v in node_set that has the most neighbors in node_set.
        
        # Optimization: Instead of finding the best v every time (O(n^2)), 
        # we can just pick the lowest set bit. The bound is strong enough for n<300.
        
        lsb = node_set & -node_set
        v_bit = lsb
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates are node_set without v and without any neighbors of v
        # neighbors of v in node_set: conflict[v] & node_set
        new_node_set_1 = node_set & ~(conflict[v] | v_bit)
        current_set.append(v)
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
            # Simple greedy on remaining
            for i in range(m):
                if remaining & (1 << i):
                    # Find smallest available color
                    used = set()
                    for j in range(m):
                        if (remaining & (1 << j)) and (conflict[i] & (1 << j)):
                            if colour[j] != -1:
                                used.add(colour[j])
                    c = 0
                    while c in used:
                        c += 1
                    colour[i] = c
            break
        
        # Allocate time for this MIS step
        # As remaining set shrinks, MIS is faster.
        # Give a fraction of remaining time, capped to avoid wasting time on one step.
        elapsed = time.time() - t_start
        rem_time = time_budget - elapsed
        # Cap step time to 5s or 10% of remaining, whichever is smaller but at least 0.1s
        step_time = max(0.1, min(5.0, rem_time * 0.1))
        
        # Find Max Independent Set in remaining
        # To improve quality, we can run multiple random restarts of B&B if time permits?
        # B&B is deterministic based on vertex selection order.
        # If we change the order of vertices in the conflict graph, we get different results.
        # But we can't easily change the bit positions without remapping.
        # Instead, we can just run the B&B once. It's quite effective.
        
        mis_nodes = _max_independent_set(remaining, conflict, step_time, _random_mod.Random(123))
        
        if not mis_nodes:
            # Fallback if B&B fails or times out with no result
            # Just pick the first available edge
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            mis_nodes = [v]
            
        for v in mis_nodes:
            colour[v] = color_id
            remaining &= ~(1 << v)
            
        color_id += 1
        
    return colour

def _greedy_fallback(edges, conflict, adj):
    """Fast greedy fallback if time runs out."""
    m = len(edges)
    colour = [-1] * m
    # Order by degree
    order = sorted(range(m), key=lambda i: -len([j for j in range(m) if conflict[i] & (1<<j)]))
    for i in order:
        used = set()
        mask = conflict[i]
        for j in range(m):
            if mask & (1 << j) and colour[j] != -1:
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
    
    # Allocate time: 
    # d=4 (n=40) is smallest. d=6 (n=80) is largest.
    # Give more time to larger instances.
    # Weights: 1, 1.5, 2.5 -> Sum 5.0
    weights = [1.0, 1.5, 2.5]
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
        
        # Allocate time proportional to weight, but ensure minimum time
        allocated = (weights[idx] / total_weight) * remaining_budget
        # Cap allocated to not starve later instances too much? 
        # Actually, just take the allocated amount.
        
        colour_list = _solve_instance_bitset(edges, adj, allocated)
        results.append(colour_list)
        
    return results
# EVOLVE-BLOCK-END
