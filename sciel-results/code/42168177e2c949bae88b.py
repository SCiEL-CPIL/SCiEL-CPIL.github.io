import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s safety margin
    
    best_result = {'n': 39, 'edges': []}
    
    # We know n=46 is possible. Try 47 first, then 46, etc.
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        required_min_deg = n - 39
        
        # Try to construct for this n using bitmask-based greedy
        result = _bitmask_greedy_construct(n, required_min_deg, deadline)
        
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
            
        # If bitmask greedy fails, try local search for smaller n
        if n >= 44 and time.time() < deadline:
            result = _local_search_construct(n, required_min_deg, deadline)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}

    # Fallback: return a valid small graph
    # n=39, min_deg=0 is always valid (empty graph)
    return best_result

def _check_c4_add(adj, u, v):
    """
    Check if adding edge (u,v) to graph defined by adjacency matrix adj creates a C4.
    Vectorized: C4 exists iff there's x in N(u)\{v} and y in N(v)\{u} with x!=y and (x,y) in E.
    """
    # N(u) excluding v
    mask_u = adj[u].copy()
    mask_u[v] = False
    # N(v) excluding u  
    mask_v = adj[v].copy()
    mask_v[u] = False
    
    # For each x in N(u)\{v}, check if N(x) intersects N(v)\{u}\{x}
    # This is equivalent to: sum over x in N(u)\{v} of |N(x) & N(v)\{u}\{x}| > 0
    # But we can vectorize: for all x, check adj[x] & mask_v, then sum over x where mask_u[x]
    
    # adj[x] & mask_v gives for each x, the intersection of N(x) with N(v)\{u}
    # We need to exclude x itself from mask_v when checking (since x!=y required)
    # Actually, if x is in N(v)\{u}, then adj[x][x]=0 so it won't be in adj[x]
    # Wait, adj[x][x] is always 0 in a simple graph, so adj[x] & mask_v won't include x
    # unless x is in mask_v AND adj[x][x]=1, which is impossible.
    
    # So: for each x where mask_u[x]=True, check if np.any(adj[x] & mask_v)
    # Vectorized: (adj & mask_v[:, None]) gives for each row x, the intersection
    # Then sum along axis 1, and check if any x with mask_u[x] has sum > 0
    
    # adj[x] & mask_v for all x: this is adj & mask_v[None, :]
    # Sum along axis 1: for each x, count of y in N(x) ∩ N(v)\{u}
    intersections = np.sum(adj & mask_v[None, :], axis=1)
    
    # Check if any x in N(u)\{v} has intersection > 0
    return np.any(intersections[mask_u] > 0)

def _bitmask_greedy_construct(n, required_min_deg, deadline):
    """
    Attempts to build a C4-free graph with min degree >= required_min_deg.
    Uses bitmask-based C4 checking for speed, enabling many more restarts.
    """
    rng = np.random.default_rng()
    
    # More restarts since check is now faster
    max_restarts = 500
    
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
            
        adj = np.zeros((n, n), dtype=bool)
        # Precompute bitmask representations for faster C4 checking
        # S2_mask[u] = bitmask of vertices v such that (u,v) would create a C4 if added
        # We'll maintain this incrementally
        
        success = False
        while time.time() < deadline:
            degrees = adj.sum(axis=1)
            min_deg = np.min(degrees)
            
            if min_deg >= required_min_deg:
                success = True
                break
                
            # Find vertices with minimum degree
            min_vertices = np.where(degrees == min_deg)[0]
            if len(min_vertices) == 0:
                break
            
            # Try each min-degree vertex, pick one that can be improved
            # Shuffle to break symmetry
            shuffled = min_vertices.copy()
            rng.shuffle(shuffled)
            
            u = shuffled[0]
            
            # Get non-neighbors of u
            non_neighbors = np.where(~adj[u])[0]
            non_neighbors = non_neighbors[non_neighbors != u]
            
            if len(non_neighbors) == 0:
                break
            
            # Shuffle to try random order
            rng.shuffle(non_neighbors)
            
            added = False
            for v in non_neighbors:
                if _check_c4_add(adj, u, v):
                    continue
                # Safe to add
                adj[u, v] = True
                adj[v, u] = True
                added = True
                break  # Restart loop to re-evaluate degrees
            
            if not added:
                # Could not add any edge to u without creating C4
                # Try a different vertex or restart
                break
                
        if success:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return (n, edges)
            
    return None

def _local_search_construct(n, required_min_deg, deadline):
    """
    Fallback: Local search with edge swaps to fix stuck states.
    """
    # First try greedy to get a starting point
    result = _bitmask_greedy_construct(n, required_min_deg, deadline)
    if result is not None:
        return result
        
    # If greedy fails, try edge-swap local search
    # Start from a random C4-free graph and try to increase min degree
    for seed in range(30):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(seed)
        adj = np.zeros((n, n), dtype=bool)
        
        # Build a random C4-free graph
        pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u, v]:
                continue
            if not _check_c4_add(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                
        degrees = adj.sum(axis=1)
        if np.min(degrees) >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return (n, edges)
        
        # Local search: try edge swaps to increase min degree
        # For each vertex with low degree, try to swap an edge to raise its degree
        for iteration in range(800):
            if time.time() > deadline:
                break
            
            degrees = adj.sum(axis=1)
            if np.min(degrees) >= required_min_deg:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v]:
                            edges.append([u, v])
                return (n, edges)
            
            # Find vertex with minimum degree
            min_deg = np.min(degrees)
            min_vertices = np.where(degrees == min_deg)[0]
            u = rng.choice(min_vertices)
            
            # Try to remove an edge incident to a high-degree vertex and add an edge incident to u
            # This is complex; for now, just try adding edges to u
            non_neighbors = np.where(~adj[u])[0]
            non_neighbors = non_neighbors[non_neighbors != u]
            
            if len(non_neighbors) == 0:
                continue
            
            rng.shuffle(non_neighbors)
            
            for v in non_neighbors:
                if not _check_c4_add(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    break
            
    return None

# EVOLVE-BLOCK-END
