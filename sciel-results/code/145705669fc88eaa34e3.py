"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle_greedy(adj, n, seed, deadline=None):
    """
    Find a long cycle using greedy longest-path extension.
    Returns a list of vertices forming a cycle, or empty list if none found.
    """
    if not adj:
        return []
    
    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []
    
    rng = random.Random(seed)
    rng.shuffle(vertices)
    
    # Try multiple starting vertices
    for start in vertices[:8]:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue
        
        # Greedy longest path from start
        path = [start]
        visited = {start}
        
        # Extend path greedily to longest simple path
        while True:
            u = path[-1]
            # Find unvisited neighbor with highest degree (to keep options open)
            candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
            if not candidates:
                break
            candidates.sort(reverse=True)
            next_v = candidates[0][1]
            path.append(next_v)
            visited.add(next_v)
        
        # Check if we can close the cycle
        if len(path) >= 3 and start in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)
        
        # Try to find a cycle by looking for back edges
        # For each pair (i, j) in path where i < j-1 and path[i] adjacent to path[j], 
        # we have a cycle path[i..j]
        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 2, len(path)):
                if path[i] in adj[path[j]]:
                    cycle = path[i:j+1]
                    if len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break  # Only take the first (longest) cycle from this i
        
        if len(best_cycle) >= n - 1:  # Found a near-Hamiltonian cycle
            break
    
    return best_cycle

def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    deadline = start_time + time_limit
    
    def greedy_decompose(remaining_edges, n, seed_offset=0):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_counter = 0
        while remaining:
            if time.time() > deadline:
                # Emergency exit: add all remaining as single edges
                for e in remaining:
                    pieces.append([e[0], e[1]])
                return pieces
            
            # Find a long cycle
            cycle = _find_cycle_greedy(adj, n, seed + seed_offset + cycle_counter, deadline=deadline)
            if cycle:
                # Extract cycle edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)
                
                # Remove from remaining and adj
                for e in cycle_edges:
                    if e in remaining:
                        remaining.discard(e)
                        u, v = e
                        adj[u].discard(v)
                        adj[v].discard(u)
                
                pieces.append(cycle)
                cycle_counter += 1
            else:
                # No cycle found, take a single edge
                # Pick an edge that is likely a bridge or in a small component
                # Strategy: pick edge where both endpoints have low degree
                e = min(remaining, key=lambda e: (len(adj[e[0]]) + len(adj[e[1]]), e))
                remaining.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])
                
        return pieces

    # Multi-start search with different random seeds
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:  # Safety margin
            break
        
        pieces = greedy_decompose(edges, n, restart)
        current_count = len(pieces)
        
        if current_count < best_count:
            best_count = current_count
            best_pieces = pieces
            
    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]
    
    return best_pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    for idx, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        
        # Allocate time proportionally, but ensure each gets at least 10s
        remaining_time = total_deadline - time.time()
        graphs_left = len(specs) - idx
        time_for_this = max(10.0, remaining_time / graphs_left - 1.0)
        
        pieces = _decompose(edges, n, seed, time_limit=time_for_this)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
