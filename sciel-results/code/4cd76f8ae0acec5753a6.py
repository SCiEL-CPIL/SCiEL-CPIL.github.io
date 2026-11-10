# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    """
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute for each demand and each direction (0=clockwise, 1=counter-clockwise)
    # which edges are used
    # edge_mask[d][dirc] is a boolean array of length 10
    edge_masks = np.zeros((D, 2, n_nodes), dtype=np.int8)
    
    for d, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod 10)
        # If s <= t: edges s, s+1, ..., t-1
        # If s > t: edges s, s+1, ..., 9, 0, ..., t-1
        cw_edges = []
        e = s
        while e != t:
            cw_edges.append(e)
            e = (e + 1) % n_nodes
        for e in cw_edges:
            edge_masks[d, 0, e] = 1
        
        # Counter-clockwise: all other edges
        for e in range(n_nodes):
            if edge_masks[d, 0, e] == 0:
                edge_masks[d, 1, e] = 1
    
    # Weights for each demand
    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)
    
    # For exhaustive search: 2^15 = 32768 assignments
    # We'll vectorize using numpy
    # Each assignment is a 15-bit integer from 0 to 32767
    # Bit d of the integer indicates direction for demand d (0=cw, 1=ccw)
    
    all_assignments = np.arange(2**D, dtype=np.int32)
    
    # For each assignment, compute the load on each edge
    # load[assignment, edge] = sum over demands d where direction[d] is used and edge is in that direction's mask
    
    best_max_load = np.inf
    best_assignment = 0
    
    # Process in chunks to avoid memory issues
    chunk_size = 4096
    for start in range(0, 2**D, chunk_size):
        end = min(start + chunk_size, 2**D)
        chunk = all_assignments[start:end]
        
        # For each demand d, determine which direction is used
        # direction[d] = (assignment >> d) & 1
        # We want load[edge] = sum_d weight[d] * (edge is used in direction d for assignment)
        
        # Build a matrix: for each assignment in chunk, for each demand, which direction
        # dirs shape: (chunk_size, D)
        dirs = np.zeros((len(chunk), D), dtype=np.int8)
        for d in range(D):
            dirs[:, d] = (chunk >> d) & 1
        
        # Now compute loads: (chunk_size, n_nodes)
        # For each demand d, contribution to edge e is weight[d] if (dirs[., d] == 0 and edge_masks[d,0,e]==1) or (dirs[., d] == 1 and edge_masks[d,1,e]==1)
        
        loads = np.zeros((len(chunk), n_nodes), dtype=np.float64)
        for d in range(D):
            # For each assignment, which edges are used by demand d
            # edges_used = np.where(dirs[:, d] == 0, edge_masks[d, 0], edge_masks[d, 1])
            # But we need to do this per-assignment
            for e in range(n_nodes):
                # For this demand and edge, it contributes weight[d] if the direction used includes this edge
                mask_cw = edge_masks[d, 0, e]  # 1 if cw uses edge e
                mask_ccw = edge_masks[d, 1, e]  # 1 if ccw uses edge e
                # contribution = weight[d] * (dirs[:,d]==0 ? mask_cw : mask_ccw)
                contrib = np.where(dirs[:, d] == 0, mask_cw, mask_ccw) * weights[d]
                loads[:, e] += contrib
        
        max_loads = np.max(loads, axis=1)
        min_idx = np.argmin(max_loads)
        if max_loads[min_idx] < best_max_load:
            best_max_load = max_loads[min_idx]
            best_assignment = chunk[min_idx]
    
    # Convert best_assignment to routing array
    routing = np.zeros(D)
    for d in range(D):
        routing[d] = (best_assignment >> d) & 1
    
    return routing


# EVOLVE-BLOCK-END
