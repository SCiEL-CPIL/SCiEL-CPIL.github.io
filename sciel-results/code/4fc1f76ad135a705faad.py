"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ACO on CVRP.
    
    Uses a combination of:
    1. Inverse distance with power tuning (greedy towards nearest neighbors)
    2. Demand-aware weighting to prefer moves that leave good remaining capacity
    3. Smooth surrogate for capacity constraint to avoid hard cutoffs
    """
    eps = 1e-9
    n = distance_matrix.shape[0]
    
    # Ensure distances are strictly positive
    d = distance_matrix.copy()
    d = np.where(d > 0, d, eps)
    
    # Base heuristic: inverse distance with power 1.5 (between linear and quadratic)
    # Power 1.5 is a good balance - not too greedy, not too uniform
    base_eta = 1.0 / (d ** 1.5)
    
    # Demand-aware adjustment:
    # For each edge i->j, consider the demand of j
    # We want to slightly discourage edges to high-demand nodes when
    # they would leave awkward remaining capacity
    demands_arr = np.asarray(demands, dtype=float)
    demands_arr[0] = 0  # depot has 0 demand
    
    # Compute a smooth "capacity fit" score for each edge
    # For edge i->j, the remaining capacity after visiting j from a full load
    # is (capacity - demands[j]). We want this to be "round" or at least
    # not too small.
    
    # A simple approach: penalize edges to nodes with demand that is a "bad"
    # fraction of capacity. Use a smooth function.
    # Let r = demands[j] / capacity
    # We want to slightly boost edges where r is "nice" (e.g., close to 1/c for small c)
    # But this is complex. Simpler: just use inverse distance with a mild
    # demand-based scaling.
    
    # Alternative: use the coordinate information to add a "directional" bonus
    # Edges that continue in a similar direction (low turn angle) are slightly
    # more attractive, as this reduces total distance in practice.
    
    coords = np.asarray(coordinates, dtype=float)
    
    # Compute direction vectors for each edge
    # For edge i->j, direction is coords[j] - coords[i]
    # For a "smooth" continuation, we want the angle between consecutive edges to be small.
    # But this requires knowing the previous node, which we don't have in a static heuristic.
    
    # Instead, let's use a different approach: 
    # The heuristic should reflect the "desirability" of moving from i to j.
    # A key insight: in CVRP, we want to group customers with compatible demands.
    # So for edge i->j, if demands[j] is small, it's easier to fit into many routes.
    # If demands[j] is large, it's more constrained.
    # We can slightly boost edges to low-demand nodes.
    
    demand_scale = demands_arr.copy()
    demand_scale[0] = 0
    
    # Normalize demands to [0, 1]
    max_demand = np.max(demand_scale) if np.max(demand_scale) > 0 else 1.0
    norm_demands = demand_scale / max_demand
    
    # Boost factor: 1 + alpha * (1 - norm_demand[j])
    # This slightly favors edges to low-demand nodes
    alpha = 0.1  # Mild boost
    boost = 1.0 + alpha * (1.0 - norm_demands[np.newaxis, :])
    
    # Apply boost to base heuristic
    eta = base_eta * boost
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Normalize to have mean of 1 for consistency
    mean_eta = np.mean(eta)
    if mean_eta > 0:
        eta = eta / mean_eta
    
    return eta
# EVOLVE-BLOCK-END
