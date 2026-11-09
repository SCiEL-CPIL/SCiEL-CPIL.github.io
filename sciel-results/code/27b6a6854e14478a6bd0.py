"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    h_base = 1.0 / (distance_matrix + 1e-9)
    
    # Demand-aware weighting:
    # For each destination j, we want to favor customers with moderate demand
    # (not too low - wastes capacity, not too high - may not fit with others)
    # Optimal single-visit demand is around capacity/2 = 25, but since max demand is 9,
    # we want to slightly favor higher demand customers to reduce number of trips
    
    # Strategy: weight by demand to encourage filling vehicles
    # H[i,j] = 1 / (dist[i,j] * (1 + alpha * (capacity - demand[j]) / capacity))
    # This penalizes going to low-demand customers slightly
    
    # Alternative: use a demand factor that rewards higher demand
    # demand_factor[j] = demand[j] / max_demand (normalized)
    max_demand = max(demands[1:], 1.0)  # exclude depot
    demand_norm = np.ones(n)
    demand_norm[1:] = demands[1:] / max_demand
    
    # Weight: slightly favor higher demand customers
    # H[i,j] = 1 / (dist[i,j]) * (0.5 + 0.5 * demand_norm[j])
    # This gives range [0.5, 1.0] multiplier based on demand
    
    demand_weight = 0.5 + 0.5 * demand_norm  # shape (n,)
    h_demand = h_base * demand_weight[None, :]
    
    # Additional consideration: depot-to-customer edges should be slightly boosted
    # since starting a new route should be attractive when the vehicle is empty
    # Customer-to-depot edges (returning) should also be considered
    
    # Combine: use the demand-weighted version as primary
    # Ensure all values are positive
    result = np.maximum(h_demand, 1e-9)
    
    # Verify shape
    assert result.shape == distance_matrix.shape
    
    return result
# EVOLVE-BLOCK-END
