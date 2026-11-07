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
    """Check if the 2-lift is connected using fundamental cycle products.
    With gauge fixing (tree edges = +1), the lift is connected iff at least
    one fundamental cycle (defined by a free/chord edge) has sign product -1.
    
    Fundamental cycles for the prism graph with our spanning tree:
    - Top cycle chord (last top edge): product = prod of all top cycle signs
    - Each bottom cycle edge j: product = prod of bottom cycle signs
    (because matching edges and top tree edges are +1 in the tree)
    """
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    top_prod = 1
    bottom_prod = 1
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if u < m and v < m:
            top_prod *= s
        elif u >= m and v >= m:
            bottom_prod *= s
        # matching edges are in the tree (sign +1), don't affect cycle products
    
    return (top_prod == -1) or (bottom_prod == -1)

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 70.0  # Safety margin (evaluator kills at 90s)
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Classify edges and build index arrays
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
    
    # Gauge fixing: spanning tree = all matching edges + first (m-1) top cycle edges
    # Free edges: last top cycle edge + all m bottom cycle edges = m+1 variables
    # For m=4: 2^5=32, m=5: 2^6=64, m=7: 2^8=256, m=9: 2^10=1024
    
    # Fixed edges (spanning tree)
    fixed_edges = set(matching_edges)
    fixed_edges.update(top_cycle_edges[:-1])
    
    # Free edges
    free_edges = [top_cycle_edges[-1]] + bottom_cycle_edges
    num_free = len(free_edges)
    
    # Pre-compute base signs (all +1)
    base_signs = np.ones(num_edges, dtype=np.int8)
    
    # Pre-compute edge endpoint arrays for fast S construction
    u_arr = np.array([e[0] for e in edges], dtype=np.int32)
    v_arr = np.array([e[1] for e in edges], dtype=np.int32)
    
    best_signs = None
    best_radius = float('inf')
    
    # Exhaustive enumeration over 2^num_free
    for mask in range(1 << num_free):
        if time.time() > deadline:
            break
        
        # Build signs vector
        signs = base_signs.copy()
        for j, edge_idx in enumerate(free_edges):
            if (mask >> j) & 1:
                signs[edge_idx] = -1
        
        # Fast connectivity check: top_prod * bottom_prod == -1
        # (since tree edges are +1, fundamental cycle products are just
        #  the product of top cycle signs and bottom cycle signs)
        top_prod = 1
        for i in top_cycle_edges:
            top_prod *= signs[i]
        bottom_prod = 1
        for i in bottom_cycle_edges:
            bottom_prod *= signs[i]
        
        if top_prod == 1 and bottom_prod == 1:
            continue  # Disconnected
        
        # Build signed adjacency matrix S (symmetric)
        S = np.zeros((n, n), dtype=np.float64)
        S[u_arr, v_arr] = signs.astype(np.float64)
        S[v_arr, u_arr] = signs.astype(np.float64)
        
        # Spectral radius
        eigvals = np.linalg.eigvalsh(S)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Last resort: random search
        rng = np.random.default_rng(3)
        for _ in range(min(budget, 5000)):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            if _lift_connected(graph, signs):
                S = np.zeros((n, n), dtype=np.float64)
                S[u_arr, v_arr] = signs.astype(np.float64)
                S[v_arr, u_arr] = signs.astype(np.float64)
                eigvals = np.linalg.eigvalsh(S)
                radius = np.max(np.abs(eigvals))
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
    
    return best_signs
# EVOLVE-BLOCK-END
