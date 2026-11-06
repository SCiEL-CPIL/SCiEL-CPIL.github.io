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

def _find_longest_cycle(adj, n, seed, max_visits=20000):
    """
    Finds a long cycle in the graph defined by adj.
    Uses randomized DFS with a visit limit to balance quality and speed.
    """
    if not adj:
        return []
    
    best_cycle = []
    vertices = list(adj.keys())
    if not vertices:
        return []
    
    rng = random.Random(seed)
    rng.shuffle(vertices)
    
    # Try a few random start vertices
    for start in vertices[:10]:
        if not adj[start]:
            continue
            
        visited = {start}
        path = [start]
        visits = 0
        
        def dfs(u):
            nonlocal visits, best_cycle
            if visits > max_visits:
                return
            
            # Check if we can close the cycle
            if start in adj[u] and len(path) >= 3:
                if len(path) > len(best_cycle):
                    best_cycle = list(path)
                # Do not return immediately, try to find longer cycle
            
            # Sort neighbors to encourage specific exploration patterns?
            # Randomize for diversity in restarts
            neighbors = list(adj[u])
            rng.shuffle(neighbors)
            
            for v in neighbors:
                if v == start:
                    continue
                if v not in visited:
                    visits += 1
                    if visits > max_visits:
                        return
                    visited.add(v)
                    path.append(v)
                    dfs(v)
                    path.pop()
                    visited.remove(v)
                    if visits > max_visits:
                        return
        
        dfs(start)
        if visits > max_visits:
            break
            
    return best_cycle

def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    
    # Initial greedy decomposition to have a baseline
    def greedy_decompose(remaining_edges, n, seed_offset=0):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(list)
        for u, v in remaining:
            adj[u].append(v)
            adj[v].append(u)
        
        cycle_counter = 0
        while remaining:
            # Find a long cycle
            cycle = _find_longest_cycle(adj, n, seed + seed_offset + cycle_counter, max_visits=10000)
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
                        adj[u].remove(v)
                        adj[v].remove(u)
                
                pieces.append(cycle)
                cycle_counter += 1
            else:
                # No cycle found, take a single edge
                # Pick an edge incident to a high-degree vertex to keep graph connected?
                # Or just any edge.
                # To maximize future cycles, maybe remove edges that are "bridges" or in small components?
                # Simple: pick random edge
                e = next(iter(remaining))
                remaining.discard(e)
                u, v = e
                adj[u].remove(v)
                adj[v].remove(u)
                pieces.append([u, v])
                
        return pieces

    # Multi-start search
    for restart in range(200):
        if time.time() - start_time > time_limit:
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
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=40.0)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
