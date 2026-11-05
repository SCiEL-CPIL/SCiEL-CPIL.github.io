import time
from random import Random
from collections import deque

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def solve_max_even_subgraph(edges, n, time_limit=10):
    """
    Finds a subgraph of edges where every vertex has even degree, maximizing the number of edges.
    Uses Simulated Annealing / Local Search.
    """
    m = len(edges)
    if m == 0:
        return set()
    
    # Initialize with empty set or random?
    # Start with all edges, then remove to fix parity?
    # Or start with empty, add edges?
    
    # Let's start with a random feasible solution or empty.
    # Empty is feasible (0 is even).
    x = [0] * m
    deg = [0] * n
    
    # Precompute incident edges for each vertex
    inc = {u: [] for u in range(n)}
    for i, (u, v) in enumerate(edges):
        inc[u].append(i)
        inc[v].append(i)
        
    def flip(i):
        nonlocal x, deg
        u, v = edges[i]
        if x[i]:
            x[i] = 0
            deg[u] -= 1
            deg[v] -= 1
        else:
            x[i] = 1
            deg[u] += 1
            deg[v] += 1
            
    def cost():
        # We want to maximize sum(x) subject to deg[u] % 2 == 0.
        # Penalty for odd degrees.
        odd_count = sum(1 for d in deg if d % 2 != 0)
        return -sum(x) + 100 * odd_count # Minimize this
        
    # Simulated Annealing
    import random
    rng = random.Random(42)
    
    # Initial state: Try to build a large subgraph.
    # Greedy initialization: Add edges randomly, if it creates odd parity, keep it but remember to fix later?
    # Better: Start with empty.
    
    current_cost = cost()
    best_x = x[:]
    best_cost = current_cost
    
    T = 10.0
    T_min = 0.01
    alpha = 0.99
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < time_limit:
        # Pick a random edge to flip
        i = rng.randint(0, m - 1)
        flip(i)
        new_cost = cost()
        
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < pow(2.718, -delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x[:]
        else:
            flip(i) # Revert
            
        T *= alpha
        
    # Restore best solution
    x = best_x[:]
    # Recalculate deg for best_x
    deg = [0] * n
    for i in range(m):
        if x[i]:
            u, v = edges[i]
            deg[u] += 1
            deg[v] += 1
            
    # The selected edges form a union of cycles.
    # We need to extract the cycles.
    selected_edges = [edges[i] for i in range(m) if x[i]]
    return set(selected_edges)

def extract_cycles_from_subgraph(selected_edges, n):
    """
    Given a set of edges where every vertex has even degree,
    decompose it into disjoint cycles.
    """
    if not selected_edges:
        return []
        
    adj = {u: set() for u in range(n)}
    for u, v in selected_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    cycles = []
    remaining_edges = set(selected_edges)
    
    while remaining_edges:
        # Pick any edge
        u, v = next(iter(remaining_edges))
        # Walk to form a cycle
        path = [u, v]
        remaining_edges.discard((u, v) if u < v else (v, u))
        current = v
        prev = u
        
        while current != u:
            # Find next neighbor in remaining edges
            next_node = None
            for nb in adj[current]:
                e = (current, nb) if current < nb else (nb, current)
                if e in remaining_edges:
                    next_node = nb
                    remaining_edges.discard(e)
                    break
            if next_node is None:
                # Should not happen if degrees are even and we started with an edge
                # But if the subgraph is not connected properly?
                # In a 2-regular graph, it's a cycle.
                # If degrees > 2, we might have intersecting cycles.
                # The walk might get stuck if we don't manage the "remaining" set correctly.
                # Actually, if we just remove edges as we go, and all degrees are even,
                # we will eventually return to the start?
                # Yes, because we are traversing a path in a graph where all vertices have even degree.
                # Wait, if a vertex has degree 4, we might enter and leave, but we removed one edge.
                # The "remaining" graph might not have even degrees anymore.
                # 
                # Correct approach for extracting cycles from an even subgraph:
                # 1. While edges remain:
                #    a. Find any vertex with degree > 0 in the CURRENT remaining subgraph.
                #    b. Walk along edges, removing them as we go, until we return to start.
                #    This works because the subgraph is a union of edge-disjoint cycles.
                #    Removing a cycle leaves a graph that still has even degrees.
                
                # Let's restart the walk logic more carefully.
                break
        
        if next_node is not None or current == u:
             # This logic is getting messy. Let's use a standard cycle extraction.
             pass
             
    # Let's rewrite extraction properly
    cycles = []
    remaining_edges = set(selected_edges)
    adj = {u: set() for u in range(n)}
    for u, v in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    while remaining_edges:
        # Find a start vertex
        start = None
        for u in range(n):
            if adj[u]:
                start = u
                break
        if start is None:
            break
            
        # Walk
        path = [start]
        prev = None
        curr = start
        while True:
            # Find an edge (curr, next) in remaining
            next_node = None
            for nb in list(adj[curr]):
                e = (curr, nb) if curr < nb else (nb, curr)
                if e in remaining_edges:
                    next_node = nb
                    break
                    
            if next_node is None:
                break # Should not happen if connected component is a cycle union
                
            # Remove edge
            e = (curr, next_node) if curr < next_node else (next_node, curr)
            remaining_edges.discard(e)
            adj[curr].discard(next_node)
            adj[next_node].discard(curr)
            
            path.append(next_node)
            prev = curr
            curr = next_node
            
            if curr == start:
                break
                
        if len(path) >= 3:
            cycles.append(path)
            
    return cycles

def decompose_graph(edges, n, time_limit=15):
    if not edges:
        return []
        
    # 1. Find max even subgraph
    selected_edges = solve_max_even_subgraph(edges, n, time_limit)
    
    # 2. Extract cycles from selected edges
    cycles = extract_cycles_from_subgraph(selected_edges, n)
    
    # 3. Remaining edges are single edges
    remaining_edges = set(tuple(sorted(e)) for e in edges) - set(tuple(sorted(e)) for e in selected_edges)
    
    pieces = cycles
    for e in remaining_edges:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start_time = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        
        # Time budget per graph: 40s total / 3 graphs = ~13s each.
        # Leave safety margin.
        time_remaining = 140 - (time.time() - start_time)
        if time_remaining < 5:
            # Fallback to all single edges
            results.append([list(ed) for ed in edges])
            continue
            
        t_limit = max(1, min(10, time_remaining / 3))
        
        pieces = decompose_graph(edges, n, t_limit)
        results.append(pieces)
        
    return results[:3]

if __name__ == "__main__":
    construct()
