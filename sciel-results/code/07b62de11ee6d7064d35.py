"""Model-and-optimize heuristic for Capacitated VRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic using Capacity-Adaptive Nearest Neighbor.
    
    Strategy:
    1. Filter feasible nodes based on remaining capacity.
    2. Determine a 'search radius' based on remaining capacity.
       - High capacity: Larger radius, allowing selection of more distant nodes 
         to establish the route perimeter early.
       - Low capacity: Smaller radius, forcing selection of nearby nodes 
         to close the route efficiently near the depot.
    3. Score candidates within the radius (or all if none in radius) using 
       a weighted sum of distance from current and distance to depot.
       - Distance from current minimizes immediate travel.
       - Distance to depot ensures we don't get stranded far away when capacity is low.
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: nearest feasible node
        unvisited_arr = np.asarray(unvisited_nodes)
        dm_arr = np.asarray(distance_matrix)
        dems_arr = np.asarray(demands)
        feasible = unvisited_arr[dems_arr[unvisited_arr] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(dm_arr[current_node, feasible])])

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = np.asarray(unvisited_nodes)
    dm = np.asarray(distance_matrix)
    dems = np.asarray(demands)

    # Filter to feasible customers
    feasible = unvisited[dems[unvisited] <= rest_capacity]
    if len(feasible) == 0:
        return 0

    n_feasible = len(feasible)

    if n_feasible == 1:
        return int(feasible[0])

    # Distances from current node to each feasible
    dist_cur = dm[current_node, feasible]
    
    # Distances from each feasible to depot
    dist_depot = dm[feasible, depot]

    # Estimate total capacity to normalize remaining capacity.
    # We don't have the total capacity parameter, but we can infer it from the instance size or just use a heuristic.
    # A common capacity for 50 customers with demands 1-9 is around 50-100.
    # Let's assume a reference capacity to scale the radius.
    # If rest_capacity is high, we are early in the route.
    # Let's use the maximum demand as a lower bound for scale, or just a fixed scale.
    # Better: Use the range of demands or a fixed heuristic constant.
    # Let's use a "scale" that represents a "full" route capacity.
    # Since we don't know it, we can estimate it as sum(demands) / num_routes? No, we don't know num_routes.
    # Let's just use a fixed reference, e.g., 50 (from task description: capacity 50).
    REF_CAPACITY = 50.0
    
    # Capacity fraction: 1.0 if full, 0.0 if empty.
    cap_frac = rest_capacity / REF_CAPACITY
    # Clamp between 0 and 1
    cap_frac = np.clip(cap_frac, 0.0, 1.0)

    # Define search radius.
    # When cap_frac is 1.0 (start of route), we want a large radius to pick "good" distant nodes.
    # When cap_frac is 0.0 (end of route), we want a small radius to pick nearby nodes.
    # Let's define radius as a fraction of the maximum distance in the matrix.
    max_dist = np.max(dm)
    
    # Radius interpolation: 
    # At cap_frac=1, radius = 0.8 * max_dist (wide search)
    # At cap_frac=0, radius = 0.1 * max_dist (tight search)
    radius = (0.1 + 0.7 * cap_frac) * max_dist

    # Filter candidates within the radius
    in_radius_mask = dist_cur <= radius
    
    if np.any(in_radius_mask):
        candidates = feasible[in_radius_mask]
        c_dist_cur = dist_cur[in_radius_mask]
        c_dist_depot = dist_depot[in_radius_mask]
    else:
        # If no candidates in radius, use all feasible
        candidates = feasible
        c_dist_cur = dist_cur
        c_dist_depot = dist_depot

    # Scoring:
    # We want to minimize immediate distance (c_dist_cur) and future return cost (c_dist_depot).
    # However, when capacity is high, we might accept a slightly longer immediate distance
    # to get a node that is "strategically" good (e.g., far from depot, on boundary).
    # When capacity is low, we heavily weight c_dist_depot to ensure we can return.
    
    # Weight for dist_depot increases as capacity decreases.
    # w_depot = 1.0 + (1.0 - cap_frac) * 2.0  # 1.0 to 3.0
    # w_cur = 1.0
    
    w_depot = 1.0 + (1.0 - cap_frac) * 2.0
    w_cur = 1.0

    scores = w_cur * c_dist_cur + w_depot * c_dist_depot
    
    best_idx_local = int(np.argmin(scores))
    return int(candidates[best_idx_local])
# EVOLVE-BLOCK-END
