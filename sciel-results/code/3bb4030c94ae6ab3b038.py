import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the heuristic matrix for the ACO algorithm.
    Uses a fast mini-ACO simulation to learn edge attractiveness based on 
    the specific instance structure, combined with a base inverse distance.
    """
    start_time = time.time()
    d = np.asarray(distance_matrix, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = d.shape[0]
    
    # Base heuristic: Inverse distance with softening
    eps = 1e-4
    base_heur = 1.0 / (d + eps)
    np.fill_diagonal(base_heur, 0.0)
    
    # Mini-ACO Simulation to learn structural preferences
    # Parameters for the fast simulation
    n_ants = 5
    n_iterations = 10
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    # Heuristic for the mini-ACO (simple inverse distance)
    eta_mini = base_heur.copy()
    # Ensure positive values for mini-ACO
    eta_mini[eta_mini <= 0] = 1e-9
    
    edge_counts = np.zeros((n, n))
    best_cost_found = float('inf')
    
    alpha = 1.0
    beta = 2.0
    
    for _ in range(n_iterations):
        # Check time budget
        if time.time() - start_time > 5.0:
            break
            
        current_best_cost = float('inf')
        current_best_edges = []
        
        for _ in range(n_ants):
            # Ant construction
            unvisited = set(range(1, n))
            current_node = 0
            current_load = 0.0
            route = [0]
            total_dist = 0.0
            
            while unvisited:
                # Find feasible next nodes
                feasible = []
                for j in unvisited:
                    if current_load + demands[j] <= capacity:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    total_dist += d[current_node, 0]
                    current_node = 0
                    current_load = 0.0
                    continue
                
                # Calculate probabilities
                probs = np.zeros(len(feasible))
                for k, j in enumerate(feasible):
                    probs[k] = (tau[current_node, j] ** alpha) * (eta_mini[current_node, j] ** beta)
                
                total_prob = np.sum(probs)
                if total_prob == 0:
                    # Fallback to nearest
                    dists = np.array([d[current_node, j] for j in feasible])
                    next_node = feasible[np.argmin(dists)]
                else:
                    probs = probs / total_prob
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    next_node = feasible[np.searchsorted(cumsum, r)]
                
                # Move to next node
                total_dist += d[current_node, next_node]
                current_load += demands[next_node]
                unvisited.remove(next_node)
                route.append(next_node)
                current_node = next_node
                
            # Return to depot
            total_dist += d[current_node, 0]
            route.append(0)
            
            if total_dist < current_best_cost:
                current_best_cost = total_dist
                current_best_edges = route
        
        # Update global best
        if current_best_cost < best_cost_found:
            best_cost_found = current_best_cost
            # Count edges in the best route found in this iteration
            for i in range(len(current_best_edges) - 1):
                u = current_best_edges[i]
                v = current_best_edges[i+1]
                edge_counts[u, v] += 1
                edge_counts[v, u] += 1
        
        # Simple pheromone update for mini-ACO (optional, but helps convergence)
        tau = 0.9 * tau
        if best_cost_found < float('inf'):
            # Reinforce edges from the best route found so far? 
            # For simplicity in mini-sim, we just let the counts accumulate.
            pass
        
    # Construct final heuristic
    # Combine base distance heuristic with learned edge counts
    # Normalize edge counts
    max_count = np.max(edge_counts)
    if max_count > 0:
        normalized_counts = edge_counts / max_count
    else:
        normalized_counts = np.zeros_like(edge_counts)
    
    # Weight for the learned component
    lambda_learn = 2.0
    
    final_heuristic = base_heur * (1.0 + lambda_learn * normalized_counts)
    
    # Ensure diagonal is 0 (or 1e-9 later)
    np.fill_diagonal(final_heuristic, 0.0)
    
    # Handle non-finite or non-positive values
    mask = ~np.isfinite(final_heuristic) | (final_heuristic <= 0)
    final_heuristic[mask] = 1e-9
    
    return final_heuristic
