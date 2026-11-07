import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix for ant colony optimization of CVRP.
    
    Uses a structural change: instead of a simple inverse-distance heuristic,
    we compute a "capacity-feasible neighborhood" heuristic that accounts for
    the fact that in CVRP, routes must respect capacity constraints.
    
    The key insight: a good heuristic should not just prefer nearby nodes,
    but should also prefer nodes that are likely to be visited together
    (i.e., nodes that can fit in the same route segment).
    
    We use a two-component heuristic:
    1. Inverse distance (standard)
    2. A "route compatibility" factor based on how well demands can be packed
    
    Since the heuristic is static (computed once), we approximate route
    compatibility by considering the average remaining capacity after visiting
    a node.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    d = distance_matrix.astype(float)
    n = d.shape[0]
    
    # Parameters
    alpha = 1.0  # Distance exponent
    beta = 0.3   # Demand-awareness strength (reduced from 0.5)
    gamma = 0.2  # Route compatibility strength
    
    # Base heuristic: inverse distance
    eps = 1e-9
    base = 1.0 / (d + eps)
    
    # Demand-awareness factor: prefer edges to low-demand nodes
    demand_factor = 1.0 + beta * (1.0 - demands / capacity)
    demand_factor_col = demand_factor.reshape(1, -1)
    
    # Route compatibility: approximate how well nodes can be grouped
    # For each pair (i, j), estimate if they can be in the same route segment
    # by checking if their combined demand is a reasonable fraction of capacity
    # This is a static approximation since we don't know the actual route structure
    
    # Compute a "packing efficiency" score for each node
    # Nodes with demand that divides capacity well are easier to pack
    # We use a simple proxy: how many such nodes fit in one route
    nodes_per_route = capacity / (np.mean(demands[1:]) + 1e-9)
    
    # For each node, compute how "packable" it is
    # A node is more packable if its demand is a small fraction of capacity
    packability = 1.0 / (demands / capacity + 0.1)
    packability[0] = 1.0  # Depot is always packable
    
    # Build compatibility matrix: higher when both nodes are packable
    # and their demands sum to a reasonable fraction of capacity
    compat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Estimate if i and j can be in the same route segment
            # Simple heuristic: if their combined demand is < capacity, they're compatible
            combined_demand = demands[i] + demands[j]
            if combined_demand <= capacity:
                # Compatibility is higher when combined demand is a moderate fraction of capacity
                # (not too small, not too large)
                ratio = combined_demand / capacity
                # Peak at ratio ~0.5-0.7
                compat[i, j] = np.exp(-0.5 * ((ratio - 0.6) / 0.2) ** 2)
            else:
                compat[i, j] = 0.0
    
    # Normalize compatibility to [0, 1]
    max_compat = np.max(compat)
    if max_compat > 0:
        compat = compat / max_compat
    
    # Combine all components
    # eta[i,j] = base[i,j] * demand_factor[j] * (1 + gamma * compat[i,j])
    compat_factor = 1.0 + gamma * compat
    heuristic = base * demand_factor_col * compat_factor
    
    # Set diagonal to a small value
    np.fill_diagonal(heuristic, 0.0)
    
    # Replace non-positive or non-finite with 1e-9 as per task contract
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have (it's already computed)
        pass
    
    return heuristic
