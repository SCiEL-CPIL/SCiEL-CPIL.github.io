"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _savings_based_heuristic(dm, demands_arr, capacity, n):
    """
    Structural change: Use Clarke-Wright savings to derive edge attractiveness.
    
    Idea: An edge (i,j) is attractive if combining it into a route saves distance
    compared to serving i and j separately from the depot. We compute savings
    for all pairs and use them as the base heuristic, scaled by inverse distance.
    
    Additionally, we incorporate a demand-aware penalty: edges to high-demand
    nodes are slightly penalized because they consume more capacity, potentially
    forcing more trips.
    
    This is a fundamentally different approach from parametric distance scaling.
    """
    # Clarke-Wright savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    # Higher savings means it's more beneficial to connect i and j in the same route.
    d0 = dm[0, 1:]  # d(0, j) for j=1..n-1
    # savings[i,j] for i,j in 1..n-1
    savings = d0[:, np.newaxis] + d0[np.newaxis, :] - dm[1:, 1:]
    # Clip to non-negative (savings can be negative if triangle inequality is violated)
    savings = np.maximum(savings, 0.0)
    
    # Build full (n x n) savings matrix
    full_savings = np.zeros((n, n))
    full_savings[1:, 1:] = savings
    full_savings[:, 0] = 0  # Edges to depot don't have savings in the same sense
    full_savings[0, :] = 0
    
    # Inverse distance component (standard ACO heuristic)
    d_safe = dm + 1e-9
    inv_dist = 1.0 / d_safe
    
    # Demand-aware penalty: visiting a high-demand node early is good (packs heavy first),
    # but it also means less capacity remaining. We use a mild penalty proportional to demand.
    # demand_penalty[j] = 1 / (1 + 0.1 * demand[j]) -- mild penalty for high demand
    demand_penalty = 1.0 / (1.0 + 0.1 * demands_arr)
    demand_penalty_mat = np.tile(demand_penalty, (n, 1))
    
    # Combine: heuristic = inv_dist * (1 + savings) * demand_penalty
    # The (1 + savings) term rewards edges that are part of good savings-based routes.
    heuristic = inv_dist * (1.0 + full_savings) * demand_penalty_mat
    
    # Edges to/from depot: use pure inverse distance (no savings concept)
    heuristic[0, 1:] = inv_dist[0, 1:]
    heuristic[1:, 0] = inv_dist[1:, 0]
    heuristic[0, 0] = 0
    
    # Ensure positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic


def _greedy_constructive_route(dm, demands_arr, capacity, n, heuristic):
    """
    Run a greedy constructive heuristic using the given heuristic matrix.
    Returns the total route cost. Used to evaluate heuristic quality.
    """
    total_cost = 0.0
    remaining = set(range(1, n))
    
    while remaining:
        # Start a new route from depot
        current = 0
        load = 0.0
        route_cost = 0.0
        
        while remaining:
            # Find feasible customers
            feasible = [j for j in remaining if demands_arr[j] <= capacity - load]
            if not feasible:
                break
            
            # Pick the one with highest heuristic value (greedy)
            best_j = max(feasible, key=lambda j: heuristic[current, j])
            
            route_cost += dm[current, best_j]
            load += demands_arr[best_j]
            remaining.remove(best_j)
            current = best_j
        
        # Return to depot
        route_cost += dm[current, 0]
        total_cost += route_cost
    
    return total_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO using savings-based constructive heuristic.
    
    Structural change: Instead of tuning scalar exponents via grid search + proxy ACO,
    we derive edge attractiveness from Clarke-Wright savings, which directly encodes
    which pairs of customers benefit from being in the same route. This is a different
    modelling family: constructive route logic → edge weights, rather than parametric
    distance scaling.
    """
    start_time = time.time()
    deadline = start_time + 300.0
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    # Generate the savings-based heuristic
    heuristic = _savings_based_heuristic(dm, demands_arr, capacity, n)
    
    # Validate
    if heuristic.shape != (n, n):
        heuristic = np.ones((n, n)) * 1e-9
        np.fill_diagonal(heuristic, 1e-9)
        return heuristic
    
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
