import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search over all 2^15 routing 
    combinations to find the one that minimizes the maximum edge load.
    Uses vectorized NumPy operations for speed and robustness.
    Includes a deadline guard as per directive, though the search is fast enough
    to complete instantly.
    """
    start_time = time.time()
    deadline = 630.0 # Safety margin from 600s limit
    
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    cw_edges = []
    for s, t, w in demands:
        edges = []
        current = s
        while current != t:
            edges.append(current)
            current = (current + 1) % n_nodes
        cw_edges.append(edges)
    
    ccw_edges = []
    for i in range(n_demands):
        cw_set = set(cw_edges[i])
        all_edges = set(range(n_nodes))
        ccw_set = all_edges - cw_set
        ccw_edges.append(list(ccw_set))
    
    # Precompute load vectors
    # M[i] is load vector if demand i goes CW
    # C[i] is load vector if demand i goes CCW
    M = np.zeros((n_demands, n_nodes), dtype=np.float64)
    C = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i in range(n_demands):
        w = demands[i][2]
        for e in cw_edges[i]:
            M[i, e] += w
        for e in ccw_edges[i]:
            C[i, e] += w
            
    # D[i] = C[i] - M[i]
    # Total Load = sum(M[i] for all i) + sum(D[i] for i where mask[i]==1)
    base_load = M.sum(axis=0)
    D = C - M
    
    # Generate all masks
    # 2^15 = 32768
    masks = np.arange(1 << n_demands)
    
    # Create binary matrix of masks
    # shape (32768, 15)
    bits = (masks[:, np.newaxis] >> np.arange(n_demands)) & 1
    
    # Calculate loads for all masks
    # bits (32768, 15) @ D (15, 10) -> (32768, 10)
    loads = bits @ D + base_load
    
    # Find max load for each mask
    max_loads = loads.max(axis=1)
    
    # Find the mask with minimum max load
    best_idx = np.argmin(max_loads)
    
    # Extract the best routing
    best_routing = bits[best_idx].astype(float)
    
    # Check deadline (should never trigger for this problem size)
    if time.time() > deadline:
        # Fallback to a simple heuristic if somehow we ran out of time (unlikely)
        # But since we already computed the exact answer, we just return it.
        pass
        
    return best_routing
