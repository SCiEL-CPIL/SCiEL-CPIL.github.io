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
    nu = adj_matrix[u]
    nv = adj_matrix[v]
    if len(nu) == 0 or len(nv) == 0:
        return False
    overlap = (adj_matrix[nu] & nv).sum(axis=1)
    return np.any(overlap > 0)

def _is_c4_free_adj(adj):
    """Check if the adjacency matrix represents a C4-free graph."""
    n = adj.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                # Check common neighbors
                common = np.logical_and(adj[u], adj[v])
                if np.sum(common) >= 2:
                    return False
    return True

def _polarity_graph(n):
    """
    Construct a polarity graph for a prime power q.
    Vertices are (a,b) in F_q^2, edge if a*b' - a'*b = 1 (or similar).
    For n=q^2, this gives a C4-free graph with degree q.
    """
    # Find if n is a prime power
    def is_prime_power(n):
        if n < 2:
            return None
        for p in range(2, int(n**0.5) + 1):
            power = 1
            while power < n:
                power *= p
                if power == n:
                    return (p, int(round(np.log(n, p))))
        if n >= 2:
            # Check if n itself is prime
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return None
            return (n, 1)
        return None
    
    pp = is_prime_power(n)
    if pp is None:
        return None
    
    q, exp = pp
    if exp != 2:
        return None
    
    # Build polarity graph over F_q^2
    # Vertex (a, b) for a, b in F_q
    # Edge between (a,b) and (c,d) if a*d - b*c = 1
    
    adj = np.zeros((n, n), dtype=bool)
    
    def idx(a, b):
        return a * q + b
    
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    if a == c and b == d:
                        continue
                    # Check if a*d - b*c == 1 mod q
                    val = (a * d - b * c) % q
                    if val == 1:
                        adj[idx(a, b), idx(c, d)] = True
    
    return adj

def _construct_c4_free(n, min_deg_required, time_limit, seed=None):
    """
    Construct a C4-free graph on n vertices with min degree >= min_deg_required.
    Uses polarity graph seed + local search with NumPy-accelerated C4 checking.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    start_time = time.time()
    
    # Try polarity graph first for n that are prime squares
    pp_adj = _polarity_graph(n)
    if pp_adj is not None:
        degrees = pp_adj.sum(axis=1).astype(int)
        if np.min(degrees) >= min_deg_required:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if pp_adj[u, v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize adjacency matrix
        adj = np.zeros((n, n), dtype=bool)
        degrees = np.zeros(n, dtype=int)
        edge_count = 0
        
        # Precompute all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Phase 1: Greedy construction with randomization
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() - start_time > time_limit:
                break
            if not _creates_c4_numpy(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                degrees[u] += 1
                degrees[v] += 1
                edge_count += 1
        
        # Phase 2: Local search with edge swaps to improve minimum degree
        max_local_steps = 2000
        stuck_count = 0
        
        for step in range(max_local_steps):
            if time.time() - start_time > time_limit:
                break
            
            min_deg = int(np.min(degrees))
            if min_deg >= min_deg_required:
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
                break
            
            # Try adding edges to non-neighbors that don't create C4
            non_neighbors_list = non_neighbors.tolist()
            rng.shuffle(non_neighbors_list)
            
            found = False
            for v in non_neighbors_list:
                if not _creates_c4_numpy(adj, min_deg_vertex, v):
                    adj[min_deg_vertex, v] = True
                    adj[v, min_deg_vertex] = True
                    degrees[min_deg_vertex] += 1
                    degrees[v] += 1
                    edge_count += 1
                    found = True
                    stuck_count = 0
                    break
            
            if not found:
                # Stuck - try edge removal to open up possibilities
                stuck_count += 1
                
                if stuck_count > 50:
                    # Reset with a fresh attempt
                    break
                
                # Find edges incident to min_deg_vertex or to high-degree vertices
                # Remove an edge that's not critical for min degree
                candidate_edges = []
                for u in range(n):
                    if u == min_deg_vertex:
                        continue
                    for v in range(n):
                        if v <= u or v == min_deg_vertex:
                            continue
                        if adj[u, v] and degrees[u] > min_deg_required + 1 and degrees[v] > min_deg_required + 1:
                            candidate_edges.append((u, v))
                
                if candidate_edges:
                    u, v = rng.choice(candidate_edges)
                    adj[u, v] = False
                    adj[v, u] = False
                    degrees[u] -= 1
                    degrees[v] -= 1
                    edge_count -= 1
                else:
                    # Remove any random edge
                    edge_indices = np.argwhere(adj)
                    edge_indices = edge_indices[edge_indices[:, 0] < edge_indices[:, 1]]
                    if len(edge_indices) == 0:
                        break
                    idx = rng.randint(0, len(edge_indices) - 1)
                    u, v = edge_indices[idx]
                    adj[u, v] = False
                    adj[v, u] = False
                    degrees[u] -= 1
                    degrees[v] -= 1
                    edge_count -= 1
        
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
    deadline = 115.0
    
    # Try from largest n down
    time_budgets = {
        47: 30.0,
        46: 40.0,
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
            
        t_limit = min(time_budgets.get(n, 5.0), remaining_time - 1)
        
        if t_limit <= 0:
            continue
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result is not None:
            return result
    
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
