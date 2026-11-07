import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Constructs a heuristic matrix by simulating ACO dynamics with a cooling schedule.
    The returned matrix encodes the expected pheromone reinforcement trajectory.
    """
    n = len(demands)
    
    # Base heuristic: inverse distance
    # Ensure no division by zero
    base_eta = 1.0 / (distance_matrix + 1e-9)
    
    # Simulation parameters
    n_ants = 20
    n_iters = 30
    alpha = 1.0
    evaporation = 0.9
    
    # Initialize pheromone
    tau = np.ones_like(distance_matrix)
    
    # Pre-compute demand array for speed
    demands_arr = np.array(demands)
    
    # Cooling schedule for beta (heuristic influence)
    # Start with high beta (greedy) and decay to lower beta (exploratory) 
    # OR start low and increase? 
    # Standard ACO usually keeps beta fixed. 
    # The "cooling" idea suggests we want to converge. 
    # Let's try decaying a "randomness" parameter or increasing beta to focus on best edges.
    # Actually, if we return the final tau, we want tau to be concentrated on good edges.
    # High beta in simulation makes ants pick best edges more aggressively, reinforcing them more.
    # Let's ramp up beta from 1.0 to 5.0 over iterations to focus the simulation.
    
    for it in range(n_iters):
        # Cooling: increase beta to focus on best edges as we converge
        beta = 1.0 + (it / n_iters) * 4.0 
        
        best_cost_iter = float('inf')
        best_tour_iter = None
        
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
                feasible_mask = ~visited
                feasible_mask[0] = False
                feasible_mask &= (demands_arr <= remaining_capacity)
                
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    # Return to depot
                    current_cost += distance_matrix[current_node, 0]
                    current_node = 0
                    remaining_capacity = capacity
                    continue
                
                # Calculate probabilities: proportional to tau[i, j]^alpha * base_eta[i, j]^beta
                # Vectorized calculation
                tau_vals = tau[current_node, feasible_indices]
                eta_vals = base_eta[current_node, feasible_indices]
                
                # Ensure positivity
                tau_vals = np.where(tau_vals <= 0, 1e-9, tau_vals)
                eta_vals = np.where(eta_vals <= 0, 1e-9, eta_vals)
                
                weights = np.power(tau_vals, alpha) * np.power(eta_vals, beta)
                
                total_weight = np.sum(weights)
                if total_weight == 0 or not np.isfinite(total_weight):
                    # Fallback to nearest
                    dists = distance_matrix[current_node, feasible_indices]
                    next_idx_local = np.argmin(dists)
                    next_node = feasible_indices[next_idx_local]
                else:
                    probs = weights / total_weight
                    # Multinomial choice
                    r = np.random.random()
                    cum_prob = np.cumsum(probs)
                    next_idx_local = np.searchsorted(cum_prob, r)
                    if next_idx_local >= len(feasible_indices):
                        next_idx_local = len(feasible_indices) - 1
                    next_node = feasible_indices[next_idx_local]
                
                # Move
                current_cost += distance_matrix[current_node, next_node]
                remaining_capacity -= demands_arr[next_node]
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
            # Evaporate
            tau *= evaporation
            
            # Deposit on best tour edges
            # "every entry gains 1/(best cost)" in the real evaluator implies global addition,
            # but for our simulation to produce a *useful* heuristic (concentrated on good edges),
            # we must reinforce only the best tour edges. If we added to all, tau would flatten.
            # The returned matrix is used as the *heuristic* in the real run.
            # The real run's pheromone starts at 1.
            # If the real run's pheromone flattens (due to "every entry gains"),
            # then the probability is proportional to 1 * heuristic.
            # So the heuristic I return MUST be the guide.
            # Therefore, my simulation should produce a matrix that is high on good edges.
            # Standard ACO update (reinforce best tour) achieves this.
            
            deposit = 1.0 / best_cost_iter
            
            for i in range(len(best_tour_iter) - 1):
                u = best_tour_iter[i]
                v = best_tour_iter[i+1]
                tau[u, v] += deposit
                # Distance matrix is symmetric in typical VRP, but let's be safe.
                # If the tour uses u->v, we reinforce that direction.
                # If the matrix is symmetric, reinforcing v->u is also good for undirected intuition,
                # but ACO is directed. Let's just reinforce the directed edge used.
                # However, if the graph is effectively undirected (dist[i,j]==dist[j,i]),
                # reinforcing both helps if the ant might traverse it in reverse in a different context?
                # No, in a single tour, direction is fixed.
                # But since the real ACO will also be directed, reinforcing the specific edges
                # in the best tour is the most direct signal.
                # Let's reinforce both directions to be robust against any symmetry assumptions in the evaluator's distance matrix usage.
                tau[v, u] += deposit

            # Ensure positivity
            tau[tau <= 0] = 1e-9

    # Final cleanup
    heur_matrix = tau.copy()
    heur_matrix[~np.isfinite(heur_matrix)] = 1e-9
    heur_matrix[heur_matrix <= 0] = 1e-9
    
    return heur_matrix
