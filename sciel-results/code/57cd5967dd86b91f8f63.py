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

def _local_search(pieces, edges, n, seed, time_limit=10.0):
    """Local search to improve piece count by merging or reorganizing pieces."""
    if not pieces:
        return pieces
    
    # Convert pieces to a mutable structure
    current_pieces = [list(p) for p in pieces]
    current_count = len(current_pieces)
    best_pieces = [list(p) for p in current_pieces]
    best_count = current_count
    
    start_time = time.time()
    rng = random.Random(seed)
    
    # Build adjacency from edges for quick lookup
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    # Try to merge two single edges into a cycle if possible
    # Or try to extend a cycle by absorbing adjacent edges
    
    for _ in range(50):  # Limited number of local search steps
        if time.time() - start_time > time_limit:
            break
        
        # Find a random single edge piece
        single_edge_indices = [i for i, p in enumerate(current_pieces) if len(p) == 2]
        if not single_edge_indices:
            break
        
        idx = rng.choice(single_edge_indices)
        u, v = current_pieces[idx]
        
        # Try to find a cycle that could absorb this edge
        # This is complex, so for now just try random swaps
        
        # Try to find another single edge and see if they form a path that closes a cycle
        # This is too complex for now, so we'll just do a simple improvement:
        # Try to re-run greedy on the remaining edges after removing some pieces
        
        # Actually, let's try a different approach: 
        # For each cycle, try to see if we can extend it by one edge
        # For each single edge, try to see if it can be part of a new cycle with existing cycle edges
        
        # This is getting complex, so let's just do a simple local search:
        # Randomly perturb the piece set and see if we can improve
        
        # For now, this is a placeholder for a more sophisticated local search
        pass
    
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
