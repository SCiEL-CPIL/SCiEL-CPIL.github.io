import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for ACO.
    Strategy: 
    1. Define a parametric heuristic: eta_ij = 1 / (dist_ij * (1 + max(demand_i, demand_j))^beta)
       This penalizes long distances and also biases the route construction to consider demand 
       (e.g., grouping high demand items or visiting them strategically).
    2. Simulate the exact ACO process (20 ants, 30 iterations) for a few candidate betas.
    3. Return the heuristic matrix corresponding to the beta that yields the lowest 
       best-route cost in the simulation.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for the evaluator's 300s limit

    n = len(demands)
    demands = np.array(demands)
    
    # Precompute distance matrix properties
    # Ensure no division by zero in heuristic calculation later
    # distance_matrix is given.
    
    # Candidate betas for the demand-aware component
    # beta=0 is pure distance. Higher beta makes demand more influential.
    betas = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    best_beta = 0.0
    best_cost = float('inf')
    
    # We need to simulate the ACO to evaluate the heuristic.
    # Since we don't know the seed, we assume a standard ACO implementation 
    # as described in the contract. We will run a few iterations to see trends.
    # To save time, we will simulate with fewer ants/iterations for the search, 
    # but the trend should hold. Or just use the full spec if it's fast enough.
    # 20 ants * 30 iters * 50 nodes is ~30k steps per instance. Very fast.
    
    def simulate_aco(eta_matrix, n_ants=20, n_iters=30, seed=42):
        """
        Simulates the ACO process described in the contract.
        Returns the lowest total route distance found by any ant.
        """
        rng = np.random.default_rng(seed)
        
        # Pheromone matrix
        tau = np.ones((n, n))
        
        # Precompute feasible moves? 
        # Feasibility depends on current load.
        
        best_route_cost = float('inf')
        
        for it in range(n_iters):
            iteration_best_cost = float('inf')
            
            for ant in range(n_ants):
                # Ant state
                current = 0
                load = 0.0
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                total_dist = 0.0
                route = [0]
                
                while not np.all(visited):
                    # Find feasible customers
                    # j such that not visited[j] and load + demands[j] <= capacity
                    feasible = []
                    for j in range(1, n):
                        if not visited[j] and load + demands[j] <= capacity:
                            feasible.append(j)
                    
                    if not feasible:
                        # Return to depot
                        total_dist += distance_matrix[current, 0]
                        current = 0
                        load = 0.0
                        # Reset visited? No, visited stays true. 
                        # The ant continues to visit unvisited customers.
                        # Wait, if no feasible customer from current, we return to depot.
                        # From depot, load is 0, so any unvisited customer with demand <= cap is feasible.
                        # So we loop back to top of while, current=0, load=0.
                        continue
                    
                    # Calculate probabilities
                    # prob[i,j] proportional to tau[i,j] * eta[i,j]
                    # Only for feasible j
                    probs = np.zeros(len(feasible))
                    for k, j in enumerate(feasible):
                        probs[k] = tau[current, j] * eta_matrix[current, j]
                    
                    # Normalize
                    sum_probs = np.sum(probs)
                    if sum_probs == 0:
                        # Fallback: pick random feasible
                        idx = rng.integers(0, len(feasible))
                    else:
                        probs = probs / sum_probs
                        idx = rng.choice(len(feasible), p=probs)
                    
                    next_node = feasible[idx]
                    total_dist += distance_matrix[current, next_node]
                    load += demands[next_node]
                    visited[next_node] = True
                    current = next_node
                    route.append(current)
                
                # Return to depot from last position
                if current != 0:
                    total_dist += distance_matrix[current, 0]
                    route.append(0)
                
                # Update best for iteration
                if total_dist < iteration_best_cost:
                    iteration_best_cost = total_dist
                
                # Update global best
                if total_dist < best_route_cost:
                    best_route_cost = total_dist
            
            # Pheromone update
            # tau = 0.9 * tau + 1 / best_cost_of_iteration
            tau = 0.9 * tau + 1.0 / iteration_best_cost
            
        return best_route_cost

    # Search for best beta
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
        
        # Construct heuristic matrix
        # eta_ij = 1 / (dist_ij * (1 + max(demand_i, demand_j))^beta)
        # Note: Node 0 is depot, demand 0.
        # We need to handle the matrix construction efficiently.
        
        # Create demand matrix for broadcasting
        # demand_max[i,j] = max(demands[i], demands[j])
        # For i=0, demand is 0.
        
        # Efficient construction:
        # dist is n x n
        # demand_col = demands[:, None]
        # demand_row = demands[None, :]
        # max_demand = np.maximum(demand_col, demand_row)
        
        # However, we only care about i != j and i != 0 for the "choice" part?
        # The heuristic is used for transitions i->j.
        # If beta is 0, it's just 1/dist.
        
        safe_dist = distance_matrix + 1e-9
        
        if beta == 0.0:
            eta_matrix = 1.0 / safe_dist
        else:
            # Compute max demand term
            # demands is 1D array of length n
            # We want a matrix where M[i,j] = max(demands[i], demands[j])
            # This is O(N^2) memory. N=51, so 2601 floats. Trivial.
            d_col = demands[:, None]
            d_row = demands[None, :]
            max_d = np.maximum(d_col, d_row)
            
            # term = (1 + max_d) ** beta
            # To avoid overflow or issues, clip or log?
            # Demands are 1-9. Max is 9. 1+9=10. 10^2 = 100. Small.
            term = (1.0 + max_d) ** beta
            eta_matrix = 1.0 / (safe_dist * term)
        
        # Enforce constraints: non-positive or non-finite -> 1e-9
        eta_matrix[~np.isfinite(eta_matrix)] = 1e-9
        eta_matrix[eta_matrix < 1e-9] = 1e-9
        
        # Simulate
        cost = simulate_aco(eta_matrix, n_ants=20, n_iters=30, seed=42)
        
        if cost < best_cost:
            best_cost = cost
            best_beta = beta

    # Reconstruct the best heuristic
    safe_dist = distance_matrix + 1e-9
    
    if best_beta == 0.0:
        result = 1.0 / safe_dist
    else:
        d_col = demands[:, None]
        d_row = demands[None, :]
        max_d = np.maximum(d_col, d_row)
        term = (1.0 + max_d) ** best_beta
        result = 1.0 / (safe_dist * term)
    
    # Final safety checks
    result[~np.isfinite(result)] = 1e-9
    result[result < 1e-9] = 1e-9
    
    return result
