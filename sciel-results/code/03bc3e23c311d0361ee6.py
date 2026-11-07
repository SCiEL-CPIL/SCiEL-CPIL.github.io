"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _cycle_edges(cycle):
    """Get the set of edges in a cycle."""
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edges.add((min(u, v), max(u, v)))
    return edges

def _verify_cycle(cycle, n, edge_set):
    """Verify cycle is valid: all distinct vertices, all edges in host graph."""
    if len(cycle) != n:
        return False
    if len(set(cycle)) != n:
        return False
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        e = (min(u, v), max(u, v))
        if e not in edge_set:
            return False
    return True

def _find_hamiltonian_cycle(adj, n, deadline, rng, max_steps=500000):
    """Find a Hamiltonian cycle using Warnsdorff's rule with randomization."""
    start = rng.randint(0, n - 1)
    path = [start]
    visited = {start}
    
    # Precompute degrees for tie-breaking
    deg = {v: len(adj[v]) for v in range(n)}
    
    steps = 0
    while len(path) < n:
        steps += 1
        if steps > max_steps:
            return None
        if time.time() > deadline:
            return None
        
        current = path[-1]
        candidates = []
        for nb in adj[current]:
            if nb not in visited:
                # Warnsdorff: prefer neighbors with fewer unvisited neighbors
                unvisited_count = sum(1 for x in adj[nb] if x not in visited)
                candidates.append((unvisited_count, nb))
        
        if not candidates:
            return None
        
        # Sort by unvisited count (Warnsdorff's rule)
        candidates.sort()
        
        # Add some randomization: pick from top candidates
        # If the minimum unvisited count is small, we can pick from those
        min_count = candidates[0][0]
        top_candidates = [nb for cnt, nb in candidates if cnt <= min_count + 1]
        
        if not top_candidates:
            top_candidates = [candidates[0][1]]
        
        # Randomly pick from top candidates
        next_node = rng.choice(top_candidates)
        path.append(next_node)
        visited.add(next_node)
    
    # Check if we can close the cycle
    if path[0] not in adj[path[-1]]:
        return None
    
    return path

def construct():
    n = P['n']
    host_edges = set()
    for u, v in P['edges']:
        host_edges.add((min(u, v), max(u, v)))
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # The graph is C_21(1,2,5): vertex i connects to i±1, i±2, i±5 (mod 21)
    # Construct 3 edge-disjoint Hamiltonian cycles using steps 1, 2, and 5.
    # Each step-s cycle uses only edges of type (i, i+s) which are distinct across s in {1,2,5}.
    # Max possible = floor(75/21) = 3, so this is optimal.
    
    cycles = []
    used_edges = set()
    
    for step in [1, 2, 5]:
        # Build cycle: 0, step, 2*step, ..., 20*step (mod 21)
        cycle = []
        for i in range(n):
            cycle.append((i * step) % n)
        
        # Verify this is a valid Hamiltonian cycle
        if not _verify_cycle(cycle, n, host_edges):
            continue
        
        # Check edge-disjointness
        ce = _cycle_edges(cycle)
        if ce & used_edges:
            continue
        
        used_edges |= ce
        cycles.append(cycle)
    
    # If structural approach gives us 3 cycles, we're done (optimal)
    if len(cycles) >= 3:
        result = []
        for c in cycles:
            result.append([int(v) for v in c])
        return result
    
    # Fallback: try to find more cycles using Warnsdorff
    if len(cycles) < 3:
        start_time = time.time()
        deadline = start_time + 85
        rng = random.Random(42)
        
        # Remove used edges from adjacency
        adj_search = {v: set(adj[v]) for v in range(n)}
        for e in used_edges:
            u, v = e
            adj_search[u].discard(v)
            adj_search[v].discard(u)
        
        # Try to find additional Hamiltonian cycles
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            cycle = _find_hamiltonian_cycle(adj_search, n, deadline, rng)
            if cycle is not None and _verify_cycle(cycle, n, host_edges):
                ce = _cycle_edges(cycle)
                if not (ce & used_edges):
                    used_edges |= ce
                    cycles.append(cycle)
                    
                    # Update adjacency
                    for e in ce:
                        u, v = e
                        adj_search[u].discard(v)
                        adj_search[v].discard(u)
                    
                    if len(cycles) >= 3:
                        break
        
        # Ensure proper types
        result = []
        for c in cycles:
            result.append([int(v) for v in c])
        
        if not result:
            result = [list(range(n))]
        
        return result
    
    # Ensure proper types
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    if not result:
        result = [list(range(n))]
    
    return result
# EVOLVE-BLOCK-END
