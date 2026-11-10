"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Return an edge-attractiveness matrix (larger = more attractive) of the same shape.
    
    Uses a node-potential field approach:
    - Base attraction: inverse distance (standard)
    - Load-balancing pressure: exponential decay based on distance to centroid of
      remaining high-demand customers, guiding ants toward dense demand regions
    - Capacity-aware scaling: edges to high-demand nodes get boosted when
      they can be served in a single trip segment
    """
    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Ensure finite and positive
    dm = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    dm = np.maximum(dm, 1e-9)
    
    # Base heuristic: inverse distance
    base = 1.0 / dm
    
    # Compute centroid of customers (excluding depot) weighted by demand
    # Only consider customers with demand > 0
    cust_mask = dem > 0
    if np.sum(cust_mask) > 0:
        weighted_coords = coords[cust_mask] * dem[cust_mask, None]
        total_demand = np.sum(dem[cust_mask])
        centroid = weighted_coords.sum(axis=0) / total_demand
    else:
        centroid = coords[0]
    
    # Distance from each node to the demand centroid
    dist_to_centroid = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
    
    # Load-balancing pressure: nodes closer to the centroid of remaining demand
    # get a boost. Use exponential decay with a characteristic length scale.
    # The scale is chosen relative to the typical inter-customer distance.
    typical_dist = np.mean(dm[cust_mask][:, cust_mask]) if np.sum(cust_mask) > 1 else 1.0
    scale = max(typical_dist * 0.5, 0.01)
    
    # Pressure term: higher for nodes closer to centroid (encourages visiting
    # central/dense regions first to reduce total travel)
    pressure = np.exp(-dist_to_centroid / scale)
    # Normalize pressure to [0.5, 1.5] range to modulate without dominating
    pressure = 0.5 + pressure  # now in [0.5, 1.5]
    
    # Demand-based boost: edges to higher-demand customers are slightly more attractive
    # because serving them early reduces the risk of capacity fragmentation
    demand_boost = 1.0 + 0.1 * (dem / max(dem.max(), 1.0))
    
    # Combine: base * pressure * demand_boost
    # Apply pressure per destination node j
    heuristic = base * pressure[None, :] * demand_boost[None, :]
    
    # Capacity-aware adjustment: if a customer's demand is close to capacity,
    # it's better to visit it early (when the truck is empty)
    # This is approximated by boosting edges to high-demand nodes from the depot
    # and from nodes where remaining capacity is likely high (early in route)
    # Since we don't know the route state, we approximate by boosting depot->high_demand
    high_demand_mask = dem > cap * 0.5
    if np.any(high_demand_mask):
        # Boost edges from depot to high-demand customers
        heuristic[0, high_demand_mask] *= 1.3
        # Also boost edges between high-demand customers (clustering them)
        hd_idx = np.where(high_demand_mask)[0]
        for i in hd_idx:
            for j in hd_idx:
                if i != j:
                    heuristic[i, j] *= 1.1
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
