"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _creates_c4_numpy(adj_matrix, u, v):
    """
    Check if adding edge (u, v) creates a C4 using NumPy.
    C4 exists if there exist x, y such that adj[x][y] and x in adj[u] and y in adj[v].
    Equivalently: sum over x in adj[u] of (adj[x] & adj[v]) > 0
    """
    # Get neighbors of u and v (excluding each other since we're checking before adding)
    nu = adj_matrix[u]
    nv = adj_matrix[v]
    # For each neighbor x of u, check if x has any neighbor in nv
    # This is: (adj_matrix[nu] & nv).sum() > 0
    if len(nu) == 0 or len(nv) == 0:
        return False
    # adj_matrix[nu] gives a (len(nu), n) boolean array
    # Check if any row has overlap with nv
    overlap = (adj_matrix[nu] & nv).sum(axis=1)
    return np.any(overlap > 0)

def _construct_c4_free(n, min_deg_required, time_limit, seed=None):
    """
    Construct a C4-free graph on n vertices with min degree >= min_deg_required.
    Uses local search with NumPy-accelerated C4 checking.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    start_time = time.time()
    
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize adjacency matrix (numpy boolean array)
        adj = np.zeros((n, n), dtype=bool)
        
        # Strategy: Start with a random sparse graph, then add edges via local search
        # Or start empty and build up
        # Let's start with a random graph with moderate density, then repair
        
        # Initialize with random edges (low density to avoid C4s initially)
        # Actually, let's start empty and greedily add edges
        degrees = np.zeros(n, dtype=int)
        edge_count = 0
        
        # Precompute all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Phase 1: Greedy construction with randomization
        # Sort edges by some heuristic, or just random order
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() - start_time > time_limit:
                break
            # Check if adding (u,v) creates C4
            if not _creates_c4_numpy(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                degrees[u] += 1
                degrees[v] += 1
                edge_count += 1
        
        # Phase 2: Local search to improve minimum degree
        # Repeatedly try to add edges to low-degree vertices
        max_local_steps = 1000
        for step in range(max_local_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Check if we've met the requirement
            min_deg = int(np.min(degrees))
            if min_deg >= min_deg_required:
                # Success - extract edges
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v]:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            # Find vertex with minimum degree
            min_deg_vertex = int(np.argmin(degrees))
            
            # Get non-neighbors of min_deg_vertex
            non_neighbors = np.where(~adj[min_deg_vertex])[0]
            non_neighbors = non_neighbors[non_neighbors != min_deg_vertex]
            
            if len(non_neighbors) == 0:
                # Vertex is connected to everyone, can't improve
                break
            
            # Try adding edges to non-neighbors that don't create C4
            # Randomize order
            rng.shuffle(non_neighbors.tolist())
            
            found = False
            for v in non_neighbors:
                if not _creates_c4_numpy(adj, min_deg_vertex, v):
                    # Add edge
                    adj[min_deg_vertex, v] = True
                    adj[v, min_deg_vertex] = True
                    degrees[min_deg_vertex] += 1
                    degrees[v] += 1
                    edge_count += 1
                    found = True
                    break
            
            if not found:
                # Stuck - try removing a random edge to open up possibilities
                # Find all edges
                edge_indices = np.argwhere(adj)
                edge_indices = edge_indices[edge_indices[:, 0] < edge_indices[:, 1]]
                
                if len(edge_indices) == 0:
                    break
                
                # Remove a random edge
                idx = rng.randint(0, len(edge_indices) - 1)
                u, v = edge_indices[idx]
                adj[u, v] = False
                adj[v, u] = False
                degrees[u] -= 1
                degrees[v] -= 1
                edge_count -= 1
                
                # Limit repair steps to avoid infinite loops
                if step > max_local_steps // 2:
                    break
        
        # Check again after local search
        min_deg = int(np.min(degrees))
        if min_deg >= min_deg_required:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    return None

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s limit (using 115 for safety)
    
    best_result = None
    
    # Try from largest n down
    # n=47 requires min degree 8.
    # n=46 requires min degree 7.
    # n=45 requires min degree 6.
    
    time_budgets = {
        47: 25.0,
        46: 35.0,
        45: 20.0,
        44: 10.0,
        43: 5.0,
        42: 5.0,
        41: 5.0,
        40: 5.0,
        39: 2.0
    }
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
            
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2:
            break
            
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Cap the time for this n
        t_limit = min(time_budgets.get(n, 5.0), remaining_time - 1)
        
        if t_limit <= 0:
            continue
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result is not None:
            return result
    
    # Fallback: return a valid small graph
    # For n=39, min degree 0 is fine (empty graph)
    # But we can try a simple construction
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
