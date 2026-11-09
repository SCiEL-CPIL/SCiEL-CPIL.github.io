# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem with 10 nodes, 10 edges, 15 demands.
    Returns an array of 15 values; >= 0.5 means counter-clockwise, else clockwise.
    """
    n = 10
    D = 15
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
        (5, 2, 2),
    ]
    
    # Precompute for each demand:
    # - clockwise edges (from s to t going s, s+1, ..., t-1 mod n)
    # - counter-clockwise edges (the complement)
    cw_edges = []  # list of sets of edge indices for clockwise route
    ccw_edges = []  # list of sets of edge indices for counter-clockwise route
    
    for s, t, w in demands:
        # Clockwise from s to t: edges s, (s+1)%n, ..., (t-1)%n
        # Edge e joins node e and (e+1)%n
        # Going clockwise from s to t means traversing edges s, s+1, ..., t-1 (mod n)
        # But we need to be careful: if t > s, edges are s, s+1, ..., t-1
        # If t <= s, edges are s, s+1, ..., n-1, 0, ..., t-1
        cw = set()
        if t > s:
            for e in range(s, t):
                cw.add(e)
        elif t < s:
            for e in range(s, n):
                cw.add(e)
            for e in range(0, t):
                cw.add(e)
        else:
            # s == t, no edges (but this case doesn't occur here)
            pass
        cw_edges.append(cw)
        
        # Counter-clockwise: all edges not in cw
        ccw = set(range(n)) - cw
        ccw_edges.append(ccw)
    
    # Precompute as boolean arrays for fast vectorized load computation
    # For each demand d, cw_mask[d] is a boolean array of length n
    cw_mask = np.zeros((D, n), dtype=bool)
    ccw_mask = np.zeros((D, n), dtype=bool)
    weights = np.array([w for s, t, w in demands], dtype=float)
    
    for d in range(D):
        for e in cw_edges[d]:
            cw_mask[d, e] = True
        for e in ccw_edges[d]:
            ccw_mask[d, e] = True
    
    def compute_max_load(routing_bits):
        """
        routing_bits: array of 0/1 of length D. 0 = clockwise, 1 = counter-clockwise.
        Returns the maximum edge load.
        """
        # For each edge, load = sum of weights of demands routed through that edge
        # If demand d is clockwise (0), it uses cw_mask[d]
        # If demand d is counter-clockwise (1), it uses ccw_mask[d]
        
        # Build a combined mask: for each demand, select the appropriate edge mask
        # combined[d, e] = True if demand d uses edge e
        # If routing_bits[d] == 0: use cw_mask[d]
        # If routing_bits[d] == 1: use ccw_mask[d]
        
        # We can compute loads as:
        # loads[e] = sum over d of weights[d] * (routing_bits[d] * ccw_mask[d,e] + (1-routing_bits[d]) * cw_mask[d,e])
        
        # Vectorized: 
        # use_ccw = routing_bits (0 or 1)
        # use_cw = 1 - routing_bits
        # loads = use_cw @ cw_mask.T * weights... no, let me think more carefully.
        
        # loads[e] = sum_d weights[d] * indicator(d uses edge e)
        # indicator(d uses edge e) = (1 - r[d]) * cw_mask[d,e] + r[d] * ccw_mask[d,e]
        
        r = routing_bits.astype(float)
        # contribution from clockwise: (1-r) * cw_mask
        # contribution from counter-clockwise: r * ccw_mask
        # total indicator matrix: (1-r)[:,None] * cw_mask + r[:,None] * ccw_mask
        # loads = weights @ indicator_matrix  -> shape (n,)
        
        indicator = (1 - r)[:, None] * cw_mask + r[:, None] * ccw_mask
        loads = weights @ indicator  # shape (n,)
        return np.max(loads)
    
    # Exhaustive search over all 2^15 = 32768 possibilities
    # This is fast enough with vectorized operations
    
    best_load = float('inf')
    best_routing = np.zeros(D, dtype=float)
    
    # We'll iterate through all 2^15 combinations
    # Use integer representation for speed
    total_combos = 1 << D
    
    # Pre-compute for efficiency: 
    # For each combination, we need to compute max load.
    # Let's vectorize over chunks of combinations.
    
    chunk_size = 4096
    for start in range(0, total_combos, chunk_size):
        end = min(start + chunk_size, total_combos)
        # Generate all combinations in this chunk
        combo_indices = np.arange(start, end, dtype=np.int64)
        
        # Convert to binary: shape (chunk, D)
        # bit d of combo_index indicates direction of demand d
        bits = (combo_indices[:, None] >> np.arange(D)[None, :]) & 1  # shape (chunk, D)
        r = bits.astype(float)
        
        # Compute loads for all combinations in this chunk
        # indicator: (chunk, D, n)
        # indicator[i, d, e] = (1-r[i,d]) * cw_mask[d,e] + r[i,d] * ccw_mask[d,e]
        
        # Use broadcasting
        # r_expanded: (chunk, D, 1)
        # cw_mask_expanded: (1, D, n)
        # ccw_mask_expanded: (1, D, n)
        
        indicator = (1 - r[:, :, None]) * cw_mask[None, :, :] + r[:, :, None] * ccw_mask[None, :, :]
        # shape: (chunk, D, n)
        
        # loads: (chunk, n)
        loads = np.einsum('d,den->en', weights, indicator)
        # Actually: loads[i, e] = sum_d weights[d] * indicator[i, d, e]
        # = np.tensordot(weights, indicator, axes=(0, 1))  -> shape (chunk, n)
        
        loads = np.tensordot(weights, indicator, axes=(0, 1))  # shape (chunk, n)
        
        max_loads = np.max(loads, axis=1)  # shape (chunk,)
        
        # Find best in this chunk
        local_best_idx = np.argmin(max_loads)
        local_best_load = max_loads[local_best_idx]
        
        if local_best_load < best_load:
            best_load = local_best_load
            best_combo = combo_indices[local_best_idx]
            best_routing = ((best_combo >> np.arange(D)) & 1).astype(float)
    
    # best_routing is now 0/1, which satisfies the return contract
    # (>= 0.5 means counter-clockwise, < 0.5 means clockwise)
    return best_routing


# EVOLVE-BLOCK-END
