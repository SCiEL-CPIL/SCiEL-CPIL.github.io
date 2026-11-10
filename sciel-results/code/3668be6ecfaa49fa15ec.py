"""Capacitated VRP heuristic: physics-inspired vector field with demand-weighted attraction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache per-instance data to avoid recomputation across calls within the same instance
_instance_cache = {}

def _get_instance_data(depot, distance_matrix, demands):
    """Cache instance-level precomputed data keyed by a simple hash."""
    key = (id(distance_matrix), id(demands), depot)
    if key in _instance_cache:
        return _instance_cache[key]
    
    dm = np.asarray(distance_matrix)
    dem = np.asarray(demands)
    n = dm.shape[0]
    
    # Precompute distances from depot to all nodes
    depot_dists = dm[depot, :].copy()
    
    # Precompute demand array
    data = {
        'dm': dm,
        'dem': dem,
        'n': n,
        'depot_dists': depot_dists,
        'start_time': time.time()
    }
    _instance_cache[key] = data
    return data

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a physics-inspired vector field heuristic.
    The depot emits a repulsive force (stronger when capacity is low),
    and unvisited customers exert attractive forces weighted by demand and inverse distance.
    We select the node that maximizes the net attractive pull while respecting capacity.
    
    Args:
        current_node: The node currently being visited.
        depot: The index of the depot (usually 0).
        unvisited_nodes: An array-like object of indices of nodes not yet visited.
        rest_capacity: The remaining capacity of the vehicle.
        demands: An array of demands for each node.
        distance_matrix: A 2D array of distances between nodes.
        
    Returns:
        The index of the next node to visit, or 0 (depot) if the route should be closed.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Deadline guard: if we've been running too long, just close the route
    data = _get_instance_data(depot, distance_matrix, demands)
    elapsed = time.time() - data['start_time']
    if elapsed > 110.0:
        return depot

    unvisited = np.asarray(unvisited_nodes)
    dm = data['dm']
    dem = data['dem']
    depot_dists = data['depot_dists']
    
    # Filter feasible nodes: demand must be <= rest_capacity
    feasible_mask = dem[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Physics-inspired scoring:
    # Attractive force from each node: proportional to demand / distance^alpha
    # This makes high-demand nearby nodes more attractive
    # Repulsive pressure from depot: when rest_capacity is low relative to total remaining demand,
    # we want to close the route sooner
    
    # Distance from current node to each feasible node
    dists_to_current = dm[current_node, feasible_nodes]
    
    # Distance from each feasible node to depot
    dists_to_depot = depot_dists[feasible_nodes]
    
    # Demand of each feasible node
    node_demands = dem[feasible_nodes]
    
    # Attraction score: demand-weighted inverse distance
    # Higher demand nodes are more attractive, closer nodes are more attractive
    # Use a soft minimum distance to avoid division by zero
    safe_dist = np.maximum(dists_to_current, 1e-6)
    attraction = node_demands / safe_dist**0.7
    
    # Penalty for going far from depot (encourages compact routes)
    # This is the "repulsive" component from the depot
    depot_penalty = dists_to_depot * 0.3
    
    # Capacity pressure: if we're running low on capacity, prefer nodes that are
    # close to depot so we can return soon
    total_remaining_demand = dem[unvisited].sum()
    if total_remaining_demand > 0:
        capacity_pressure = 1.0 - (rest_capacity / max(total_remaining_demand, 1.0))
        capacity_pressure = max(0.0, min(1.0, capacity_pressure))
    else:
        capacity_pressure = 0.0
    
    # When capacity pressure is high, increase the depot penalty weight
    depot_penalty_weight = 0.3 + 0.5 * capacity_pressure
    depot_penalty = dists_to_depot * depot_penalty_weight
    
    # Combined score: we want to MAXIMIZE attraction and MINIMIZE depot penalty
    # So the score to minimize is: depot_penalty - attraction (or equivalently, maximize attraction - depot_penalty)
    score = depot_penalty - attraction
    
    # Also consider the savings heuristic component: prefer nodes where
    # dist(current->node) + dist(node->depot) < dist(current->depot) + dist(node->node)
    # Simplified: prefer nodes where the detour cost is low
    detour_cost = dists_to_current + dists_to_depot - dm[current_node, depot]
    
    # Final combined score (minimize this)
    # Blend physics attraction with savings-based detour cost
    final_score = 0.4 * score + 0.6 * detour_cost
    
    best_idx = np.argmin(final_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
