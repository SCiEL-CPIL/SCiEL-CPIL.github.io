"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_hamiltonian_cycle(adj, used_edges, n, deadline):
    """Find a Hamiltonian cycle using backtracking with Warnsdorff heuristic."""
    start = 0
    visited = {start}
    path = [start]
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if we can close the cycle
            last = path[-1]
            if _edge_key(last, start) not in used_edges:
                return list(path)
            return None
        
        cur = path[-1]
        # Get neighbors not yet visited, ordered by Warnsdorff's rule
        neighbors = []
        for nb in adj[cur]:
            if nb not in visited:
                if _edge_key(cur, nb) not in used_edges:
                    # Count how many unvisited neighbors nb has (excluding cur)
                    future_count = 0
                    for nb2 in adj[nb]:
                        if nb2 not in visited and nb2 != cur:
                            if _edge_key(nb, nb2) not in used_edges:
                                future_count += 1
                    neighbors.append((future_count, nb))
        
        if not neighbors:
            return None
        
        # Sort by Warnsdorff's rule: fewest future options first
        neighbors.sort()
        
        for _, nb in neighbors:
            ek = _edge_key(cur, nb)
            if ek in used_edges:
                continue
            visited.add(nb)
            path.append(nb)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
            visited.remove(nb)
        
        return None
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = P['n']
    adj = _build_adjacency(P['edges'])
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Reset for each attempt with different random seeds
        random.seed(attempt * 7919 + 13)
        
        # Shuffle adjacency lists for diversity
        adj_copy = {v: list(neighbors) for v, neighbors in adj.items()}
        for v in adj_copy:
            random.shuffle(adj_copy[v])
        
        used_edges_this_attempt = set()
        cycles_this_attempt = []
        
        for cycle_idx in range(3):
            if time.time() > deadline:
                break
            
            # Find a Hamiltonian cycle avoiding used edges
            cycle = _find_hamiltonian_cycle(adj_copy, used_edges_this_attempt, n, deadline)
            
            if cycle is None:
                break
            
            # Mark edges as used
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges_this_attempt.add(_edge_key(u, v))
            
            cycles_this_attempt.append(cycle)
        
        if len(cycles_this_attempt) > len(cycles):
            cycles = cycles_this_attempt
            if len(cycles) == 3:
                break
    
    # Validate and convert to proper types
    valid_cycles = []
    for cycle in cycles:
        if len(cycle) < 3:
            continue
        # Check all vertices are distinct
        if len(set(cycle)) != len(cycle):
            continue
        # Check all consecutive pairs are edges
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in used_edges:
                valid = False
                break
        if valid:
            valid_cycles.append([int(v) for v in cycle])
    
    # Verify edge-disjointness
    seen_edges = set()
    final_cycles = []
    for cycle in valid_cycles:
        edges_in_cycle = set()
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek in seen_edges or ek in edges_in_cycle:
                valid = False
                break
            edges_in_cycle.add(ek)
        if valid:
            seen_edges.update(edges_in_cycle)
            final_cycles.append(cycle)
    
    if not final_cycles:
        # Fallback: return at least one valid cycle if possible
        # Try a simple approach
        fallback_used = set()
        for attempt in range(100):
            random.seed(attempt)
            adj_copy = {v: list(neighbors) for v, neighbors in adj.items()}
            for v in adj_copy:
                random.shuffle(adj_copy[v])
            cycle = _find_hamiltonian_cycle(adj_copy, fallback_used, n, deadline)
            if cycle is not None:
                final_cycles = [[int(v) for v in cycle]]
                break
    
    return final_cycles
# EVOLVE-BLOCK-END
