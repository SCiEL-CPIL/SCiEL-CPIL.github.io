import numpy as np
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Constructs a heuristic matrix by simulating ACO dynamics with a cooling schedule.
    The returned matrix encodes the expected pheromone reinforcement trajectory.
    """
    n = len(demands)
    demands = np.array(demands, dtype=float)
    
    # Base heuristic: inverse distance
    base_eta = 1.0 / (distance_matrix + 1e-9)
    
    # Simulation parameters
    n_ants = 20
    n_iters = 30
    evaporation = 0.9
    
    # Initialize pheromone
    tau = np.ones_like(distance_matrix)
    
    # Pre-compute feasible transitions for speed
    # For each node i, which nodes j can be visited next given capacity?
    # Actually, feasibility depends on remaining capacity, which changes.
    # So we compute on the fly, but optimize the inner loop.
    
    # We'll use a simple random seed for reproducibility within the heuristic generation
    # The task says "seeded per instance" for the real ACO, but our heuristic
    # generation is internal. We can use a fixed seed for our simulation to be deterministic.
    rng = np.random.RandomState(42)
    
    for it in range(n_iters):
        best_cost_iter = float('inf')
        best_tour_iter = None
        
        # Cooling schedule: reduce beta or alpha over time?
        # The directive mentions "time-decaying cooling schedule for the heuristic exponent".
        # Let's introduce a beta that decays, making the heuristic less dominant over time?
        # Or more dominant?
        # Usually, we want to exploit early and explore later, or vice versa.
        # In ACO, beta is the weight of the heuristic.
        # If we want to "anticipate convergence", maybe we start with high beta (greedy)
        # and lower it? Or start low and increase?
        # Let's try: beta starts at 2.0 and decays to 1.0.
        beta = 2.0 * (1.0 - it / n_iters) + 1.0
        
        for _ in range(n_ants):
            current_node = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            remaining_capacity = capacity
            tour = [0]
            current_cost = 0.0
            
            while not np.all(visited):
                # Find feasible neighbors
                # j such that not visited[j] and demands[j] <= remaining_capacity
                # Optimize: precompute a list of unvisited nodes?
                # For n=50, a simple loop is fine, but let's vectorize the check.
                
                # Create a mask of feasible nodes
                # 1. Not visited
                # 2. Demand fits
                # 3. Not the current node (though visited handles this)
                
                feasible_mask = (~visited) & (demands <= remaining_capacity)
                feasible_mask[current_node] = False # Can't go to self
                
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    # Return to depot
                    current_cost += distance_matrix[current_node, 0]
                    current_node = 0
                    remaining_capacity = capacity
                    continue
                
                # Calculate probabilities
                # prob proportional to (tau * eta^beta)
                # Use the current tau and base_eta
                tau_vals = tau[current_node, feasible_indices]
                eta_vals = base_eta[current_node, feasible_indices]
                
                # Apply beta exponent
                # eta_vals ** beta
                # Ensure eta_vals are positive (they are, from 1/d)
                # If beta is not integer, negative eta would be an issue, but eta > 0.
                
                weights = tau_vals * (eta_vals ** beta)
                
                # Handle non-finite or zero weights
                weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1e-9)
                
                total_weight = np.sum(weights)
                if total_weight == 0:
                    # Fallback to nearest
                    dists = distance_matrix[current_node, feasible_indices]
                    next_node = feasible_indices[np.argmin(dists)]
                else:
                    probs = weights / total_weight
                    r = rng.random()
                    cum_prob = np.cumsum(probs)
                    next_idx = np.searchsorted(cum_prob, r)
                    if next_idx >= len(feasible_indices):
                        next_idx = len(feasible_indices) - 1
                    next_node = feasible_indices[next_idx]
                
                # Move
                current_cost += distance_matrix[current_node, next_node]
                remaining_capacity -= demands[next_node]
                visited[next_node] = True
                tour.append(next_node)
                current_node = next_node
            
            # Return to depot
            current_cost += distance_matrix[current_node, 0]
            tour.append(0)
            
            if current_cost < best_cost_iter:
                best_cost_iter = current_cost
                best_tour_iter = tour[:]
        
        # Update pheromone
        if best_cost_iter < float('inf') and best_tour_iter is not None:
            tau *= evaporation
            
            deposit = 1.0 / best_cost_iter
            
            # Reinforce best tour edges
            # The tour is a list of nodes: [0, c1, c2, ..., 0]
            # Edges are (0,c1), (c1,c2), ..., (ck, 0)
            for i in range(len(best_tour_iter) - 1):
                u = best_tour_iter[i]
                v = best_tour_iter[i+1]
                tau[u, v] += deposit
                # Assuming symmetric distance matrix, reinforce both directions
                # If the matrix is not symmetric, this might be wrong, but VRP usually is.
                # The problem statement implies a standard metric space (unit square), so symmetric.
                tau[v, u] += deposit
            
            # Ensure positivity
            tau[tau <= 0] = 1e-9
    
    # Final cleanup
    heur_matrix = tau.copy()
    heur_matrix[~np.isfinite(heur_matrix)] = 1e-9
    heur_matrix[heur_matrix <= 0] = 1e-9
    
    return heur_matrix
