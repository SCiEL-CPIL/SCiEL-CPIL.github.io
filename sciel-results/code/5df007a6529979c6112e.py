"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_signed_adj(graph, signs):
    """Build the signed adjacency matrix S (n x n) of the base graph."""
    n = graph['n']
    edges = graph['edges']
    S = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        S[u, v] = s
        S[v, u] = s
    return S

def _spectral_radius(S):
    eigvals = np.linalg.eigvalsh(S)
    return np.max(np.abs(eigvals))

def _lift_connected(graph, signs):
    """Check if the 2-lift is connected using cycle product criterion.
    The lift is connected iff there exists at least one cycle in the base
    graph whose product of edge signs is -1 (i.e., the signing is not a coboundary).
    """
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    # Build adjacency for BFS
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # BFS to find a spanning tree and compute potentials
    # potential[v] = product of signs along the tree path from 0 to v
    # The lift is disconnected iff for every cycle, the product of signs equals +1
    # which is equivalent to: there exists a vertex labeling phi: V -> {+1,-1}
    # such that for every edge (u,v) with sign s, phi[u]*phi[v]*s = +1
    # i.e., s = phi[u]*phi[v]
    
    # Use BFS to assign potentials
    phi = [None] * n
    phi[0] = 1
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v, ei in adj[u]:
            if phi[v] is None:
                phi[v] = phi[u] * signs[ei]
                queue.append(v)
    
    # Check if the signing is a coboundary: for every edge (u,v),
    # signs[ei] should equal phi[u] * phi[v]
    for i, (u, v) in enumerate(edges):
        if phi[u] * phi[v] != signs[i]:
            return True  # Not a coboundary, so lift is connected
    
    return False  # All cycles have product +1, lift is disconnected

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Classify edges
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    top_cycle_edges.sort()
    bottom_cycle_edges.sort()
    matching_edges.sort()
    
    # Gauge fixing via spanning tree:
    # Spanning tree = all matching edges (m) + first (m-1) top cycle edges
    # This connects all 2m vertices.
    # Free edges: 1 top cycle edge + all m bottom cycle edges = m+1 free variables
    # For m=9: 2^10 = 1024 combinations
    
    # Fixed edges (spanning tree)
    fixed_edges = set()
    for i in matching_edges:
        fixed_edges.add(i)
    for i in top_cycle_edges[:-1]:  # First m-1 top cycle edges
        fixed_edges.add(i)
    
    # Free edges
    free_edges = []
    for i in top_cycle_edges:
        if i not in fixed_edges:
            free_edges.append(i)
    for i in bottom_cycle_edges:
        free_edges.append(i)
    
    num_free = len(free_edges)
    
    # Pre-compute fixed signs (all +1 for spanning tree)
    base_signs = np.ones(num_edges, dtype=np.int8)
    
    best_signs = None
    best_radius = float('inf')
    
    # Exhaustive enumeration over 2^num_free
    for mask in range(1 << num_free):
        if time.time() > deadline:
            break
        
        signs = base_signs.copy()
        for j, edge_idx in enumerate(free_edges):
            if (mask >> j) & 1:
                signs[edge_idx] = -1
        
        # Check connectivity
        if not _lift_connected(graph, signs):
            continue
        
        # Compute spectral radius
        S = _build_signed_adj(graph, signs)
        radius = _spectral_radius(S)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # If exhaustive didn't find anything valid, do random search
    if best_signs is None:
        rng = np.random.default_rng(3)
        for _ in range(min(budget, 5000)):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            if _lift_connected(graph, signs):
                S = _build_signed_adj(graph, signs)
                radius = _spectral_radius(S)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    
    if best_signs is None:
        # Last resort
        best_signs = np.ones(num_edges, dtype=np.int8)
    
    return best_signs
# EVOLVE-BLOCK-END
