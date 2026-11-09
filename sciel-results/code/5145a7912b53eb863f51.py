"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for CVRP using a Lagrangian Relaxation-inspired heuristic.
    
    The heuristic approximates the reduced cost of an edge (i, j) in a Lagrangian relaxation
    of the CVRP where the capacity constraint is penalized by a dual variable mu.
    The effective cost of traversing edge (i, j) is modeled as distance(i, j) + mu * demand(j).
    The attractiveness is the inverse of this cost.
    """
    # Ensure inputs are numpy arrays
    dm = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # Handle edge cases or invalid inputs
    if dm.shape[0] != dm.shape[1]:
        raise ValueError("Distance matrix must be square")
    if len(dem) != dm.shape[0]:
        raise ValueError("Demands length must match matrix dimension")
    
    # Dual variable mu approximating the penalty for capacity usage.
    # Chosen such that the penalty term is comparable to typical distances.
    # Max demand is 9, so max penalty is 9 * mu.
    # Typical distance is ~0.5. 
    # If mu = 0.1, max penalty is 0.9, which is significant.
    # If mu = 0.05, max penalty is 0.45.
    # Let's use 0.1 to strongly discourage filling the truck with high-demand nodes 
    # unless they are very close, encouraging more balanced routes or earlier returns 
    # if high-demand nodes are far.
    mu = 0.1
    
    # Calculate relaxed costs: cost[i, j] = dist[i, j] + mu * demand[j]
    # We add a small epsilon to avoid division by zero if dist is 0 and demand is 0 (depot).
    # Depot demand is 0, so cost[i, 0] = dist[i, 0]. cost[0, j] = dist[0, j].
    # For customer j, cost[i, j] = dist[i, j] + mu * demand[j].
    
    # Create a matrix of demands broadcasted to columns
    # demands shape (N,), we want (N, N) where each column j has demand[j]
    demand_col = dem.reshape(1, -1)
    
    # Relaxed cost matrix
    relaxed_costs = dm + mu * demand_col
    
    # Ensure no zeros (though dist[i,i]=0 and demand[i]>0 makes cost[i,i]>0 for i>0)
    # dist[i,i]=0, demand[i]>0 -> cost = mu*demand[i] > 0.
    # dist[i,0]=dist[0,i] > 0 (usually), demand[0]=0 -> cost = dist.
    # If any distance is 0 for i!=j, cost = mu*demand[j] > 0.
    # So relaxed_costs should be strictly positive for all valid inputs.
    # Add epsilon just in case of numerical issues or zero distances between distinct nodes.
    relaxed_costs += 1e-9
    
    # Heuristic is inverse of cost
    eta = 1.0 / relaxed_costs
    
    # Ensure finite values
    eta[~np.isfinite(eta)] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
