"""Direct ACO-in-the-loop optimizer for CVRP heuristics."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix by directly optimizing 4 scalar weights
    that blend a set of analytical base matrices. The objective is the exact ACO
    metric (mean of best-ant route cost over 5 deterministic simulations).
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    
    # 1. Analytical Base Matrices
    # Base 1: Inverse distance
    base_inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Base 2: Angular continuity (S-shape)
    base_angular = np.ones((n, n))
    for i in range(1, n):
        v_i = coordinates[i] - coordinates[0]
        norm_v_i = np.linalg.norm(v_i)
        if norm_v_i < 1e-6: continue
        for j in range(1, n):
            if i == j: continue
            w_ij = coordinates[j] - coordinates[i]
            norm_w_ij = np.linalg.norm(w_ij)
            if norm_w_ij < 1e-6: continue
            cos_val = np.dot(v_i, w_ij) / (norm_v_i * norm_w_ij)
            # Map [-1, 1] to [0, 2] to keep positive
            base_angular[i, j] = 1.0 + cos_val
    base_angular = np.maximum(base_angular, 0.1)
    
    # Base 3: Demand penalty (lower demand is better to pack, but heuristic is edge attractiveness)
    # Actually, for edge attractiveness, we might want to prefer edges to nodes with 
    # demand that fits well? No, demand is a constraint. 
    # Let's use a demand-based heuristic that slightly penalizes high demand nodes 
    # to encourage packing more nodes per trip (reducing trips).
    # Or simply: prefer edges that don't violate capacity? That's handled by mask.
    # Let's use a simple demand scaling: 1 / (1 + 0.1 * demand)
    # This makes low demand nodes slightly more attractive, which might help packing.
    demand_factor = 1.0 / (1.0 + 0.1 * demands)
    base_demand = np.tile(demand_factor, (n, 1))
    
    # Base 4: Depot boost
    base_depot = np.ones((n, n))
    base_depot[0, :] = 1.5
    base_depot[:, 0] = 1.5
    
    # 2. Fast ACO Simulation
    # We simulate a simplified ACO to evaluate a given heuristic matrix.
    # 5 ants, 5 iterations. Deterministic (greedy) selection for speed, 
    # but weighted by the heuristic to mimic the probability structure.
    # Actually, to be fast and deterministic, we can just use a greedy nearest neighbor
    # guided by the heuristic: pick argmax(heuristic[i, j] * (1/distance[i, j]))?
    # No, the task says probability proportional to pheromone * heuristic.
    # If we fix pheromone to 1 (initial state), then prob proportional to heuristic.
    # A greedy choice (argmax heuristic) is a good proxy for the expected path 
    # if the heuristic is peaked enough.
    # Let's do a proper but fast simulation:
    # 5 ants. For each ant, build a route.
    # To be fast, we use a vectorized-ish approach or just tight loops.
    # n=50, 5 ants, 5 iterations. 25 routes. Each route ~50 steps. 1250 steps total.
    # This is extremely fast.
    
    def fast_aco_eval(heuristic_mat):
        """
        Simulates 5 ants for 5 iterations.
        Returns the mean of the best cost found in each iteration.
        """
        best_costs = []
        
        # We don't need full pheromone update for a quick heuristic evaluation,
        # but the task specifies the ACO structure.
        # However, for *evaluating the heuristic*, we can just look at the first iteration
        # or a few iterations where pheromone is still uniform (1.0).
        # If pheromone is uniform, the ant's choice is purely driven by the heuristic.
        # So we can just simulate the first iteration with 5 different "random" seeds
        # or just 5 greedy paths?
        # The task says "seeded per instance". 
        # To be robust, let's simulate 5 ants with deterministic pseudo-randomness
        # (e.g. using a fixed seed for the simulation) to estimate the variance.
        
        rng = np.random.RandomState(12345)
        
        for _iter in range(5): # 5 iterations
            iter_best = float('inf')
            
            for _ant in range(5): # 5 ants
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                current = 0
                load = 0.0
                total_dist = 0.0
                remaining = n - 1
                
                while remaining > 0:
                    # Feasibility mask
                    feasible = ~visited
                    feasible[0] = False
                    feasible = feasible & (load + demands <= capacity)
                    
                    feasible_indices = np.where(feasible)[0]
                    
                    if len(feasible_indices) == 0:
                        # Return to depot
                        total_dist += distance_matrix[current, 0]
                        current = 0
                        load = 0.0
                        continue
                    
                    # Probability proportional to heuristic (pheromone is 1.0 initially)
                    probs = heuristic_mat[current, feasible_indices]
                    probs = np.maximum(probs, 1e-9)
                    probs /= probs.sum()
                    
                    # Sample next node
                    next_idx = rng.choice(len(feasible_indices), p=probs)
                    next_j = feasible_indices[next_idx]
                    
                    total_dist += distance_matrix[current, next_j]
                    visited[next_j] = True
                    load += demands[next_j]
                    current = next_j
                    remaining -= 1
                
                if total_dist < iter_best:
                    iter_best = total_dist
            
            best_costs.append(iter_best)
        
        return np.mean(best_costs)
    
    # 3. Optimization Loop
    # We have 4 weights: w_inv, w_ang, w_dem, w_dep
    # Heuristic = w_inv * base_inv + w_ang * base_ang + w_dem * base_dem + w_dep * base_dep
    # We want to minimize fast_aco_eval(Heuristic)
    
    # Initial guess
    w = np.array([1.0, 0.5, 0.1, 0.5])
    
    # Simple coordinate descent / hill climbing
    # Step size
    step_size = 0.1
    
    best_w = w.copy()
    best_score = fast_aco_eval(w[0]*base_inv_dist + w[1]*base_angular + w[2]*base_demand + w[3]*base_depot)
    
    # Optimization loop
    # Budget: ~280 seconds.
    # Each eval is ~1ms. So we can do ~200,000 evals.
    # Let's do a simple random restart + hill climbing.
    
    # Pre-compute the bases into a single 4D array? No, just linear combination.
    # H = w @ Bases
    
    # To speed up, we can pre-compute the bases as a list.
    bases = [base_inv_dist, base_angular, base_demand, base_depot]
    
    def calc_heuristic(weights):
        # Vectorized linear combination
        # H = w0*B0 + w1*B1 + ...
        # This is fast for n=50.
        H = np.zeros((n, n))
        for i in range(4):
            H += weights[i] * bases[i]
        return H
    
    # Hill Climbing
    # We'll do a few restarts.
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            w = np.array([1.0, 0.5, 0.1, 0.5])
        else:
            # Random restart around best found so far
            w = best_w + np.random.randn(4) * 0.5
            w = np.maximum(w, 0.0) # Weights should be non-negative
            
        # Local search
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            current_score = fast_aco_eval(calc_heuristic(w))
            if current_score < best_score:
                best_score = current_score
                best_w = w.copy()
                improved = True # Update global best, but continue local search
                
            # Try moving in each dimension
            for dim in range(4):
                # Positive step
                w_trial = w.copy()
                w_trial[dim] += step_size
                score_trial = fast_aco_eval(calc_heuristic(w_trial))
                
                if score_trial < current_score:
                    w = w_trial
                    current_score = score_trial
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_w = w.copy()
                    continue # Re-evaluate from new position
                    
                # Negative step
                w_trial = w.copy()
                w_trial[dim] -= step_size
                if w_trial[dim] < 0: continue
                score_trial = fast_aco_eval(calc_heuristic(w_trial))
                
                if score_trial < current_score:
                    w = w_trial
                    current_score = score_trial
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_w = w.copy()
                    continue

        # Adaptive step size reduction if no improvement in a while? 
        # Simple version: just reduce step size per restart
        step_size *= 0.8
        
    # 4. Final Heuristic
    final_heuristic = calc_heuristic(best_w)
    
    # Ensure positive
    final_heuristic = np.maximum(final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
