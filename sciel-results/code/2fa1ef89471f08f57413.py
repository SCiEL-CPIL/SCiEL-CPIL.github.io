"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_heuristic(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix that anticipates the ACO pheromone
    reinforcement trajectory. We model the expected pheromone accumulation over
    30 iterations with 20 ants, using a cooling schedule that decays the
    heuristic exponent to simulate convergence bias.
    
    The idea: edges that are part of short tours will receive more pheromone
    reinforcement in early iterations. We approximate this by computing, for
    each edge, a weighted sum of its attractiveness under different 'temperatures'
    that decrease over time, mimicking how the colony converges.
    
    We also incorporate a spatial clustering component: edges connecting nearby
    nodes are more likely to be part of efficient routes.
    """
    n = distance_matrix.shape[0]
    heuristic = np.zeros_like(distance_matrix)
    
    # Base heuristic: inverse distance
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Spatial proximity heuristic: favor edges between nearby nodes
    # This captures the idea that good routes tend to be spatially compact
    spatial_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Simulate the expected pheromone trajectory
    # Pheromone starts at 1, decays by 0.9 each iteration, gains 1/best_cost
    # We approximate the best cost trajectory as decreasing
    # For a 50-customer instance with capacity 50, we expect ~5-10 routes
    
    # Cooling schedule: temperature starts high (exploration) and decreases (exploitation)
    n_iters = 30
    n_ants = 20
    initial_pheromone = 1.0
    evaporation = 0.9
    
    # Approximate the best cost at each iteration
    # Early iterations: poor solutions, later: better
    # Use a simple model: cost decreases roughly linearly in log space
    base_cost_estimate = np.sum(distance_matrix) / n * 2  # rough total tour length
    best_costs = []
    for t in range(n_iters):
        # Cost improves over time; use exponential decay model
        improvement = np.exp(-t / 10.0)
        cost_t = base_cost_estimate * (0.5 + 0.5 * improvement)
        best_costs.append(max(cost_t, 1e-6))
    
    # For each edge, compute expected pheromone contribution
    # An edge's attractiveness is proportional to how often it's chosen
    # which depends on pheromone[i,j] * heuristic[i,j]
    
    # We model this as: the heuristic should be higher for edges that
    # are likely to be in good tours. A proxy: edges with short distance
    # AND that connect nodes with compatible demands for routing.
    
    # Demand compatibility: nodes with similar demand patterns are more likely
    # to be in the same route
    # Create a demand similarity matrix
    demand_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                demand_sim[i, j] = 1.0
            else:
                # Similar demands are more likely to be routed together
                diff = abs(demands[i] - demands[j])
                demand_sim[i, j] = 1.0 / (1.0 + diff)
    
    # Combine: base heuristic (inverse distance) weighted by demand similarity
    # and a cooling-based weight that anticipates convergence
    
    # Compute weighted sum over iterations with cooling
    heuristic = np.zeros_like(distance_matrix)
    for t in range(n_iters):
        # Temperature/cooling: early iterations explore more (lower exponent),
        # later iterations exploit more (higher exponent)
        alpha = 1.0 + t / n_iters  # exponent increases over time
        # Weight this iteration's contribution
        # Pheromone at iteration t is approximately:
        # tau_t = initial_pheromone * evaporation^t + sum of reinforcements
        # Approximate reinforcement as 1/best_cost * (1 - evaporation) / (1 - evaporation)
        pheromone_estimate = initial_pheromone * (evaporation ** t) + \
                             (1.0 - evaporation ** n_iters) / (1.0 - evaporation) * \
                             np.mean([1.0 / bc for bc in best_costs[:t+1]])
        
        # Edge attractiveness at this iteration
        # Probability proportional to pheromone * heuristic
        # We want the heuristic to guide toward edges that will get reinforced
        # Use: heuristic_t[i,j] = (base_heuristic[i,j] ** alpha) * demand_sim[i,j]
        
        h_t = (base_heuristic ** alpha) * demand_sim
        heuristic += pheromone_estimate * h_t
    
    # Add a spatial compactness bonus: favor edges that keep routes spatially tight
    # This helps avoid long detours
    spatial_bonus = 0.1 * spatial_heuristic ** 1.5
    
    heuristic += spatial_bonus
    
    # Normalize to avoid extreme values
    heuristic = heuristic / (np.max(heuristic) + 1e-9)
    
    # Ensure positive values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix (larger is more attractive) of the
    same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit
    
    try:
        result = _compute_heuristic(distance_matrix, coordinates, demands, capacity)
        
        # Validate
        if result.shape != distance_matrix.shape:
            raise ValueError("Wrong shape")
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite values")
        
        return result
    except Exception:
        # Fallback to simple inverse distance heuristic
        result = 1.0 / (distance_matrix + 1e-9)
        result = np.where(result > 0, result, 1e-9)
        return result
# EVOLVE-BLOCK-END
