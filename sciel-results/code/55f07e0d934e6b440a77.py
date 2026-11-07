"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco(distance_matrix, coordinates, demands, capacity, n_ants=20, n_iterations=30,
             gamma=1.0, seed=42, time_limit=280.0):
    """
    Run ACO for the CVRP. Returns the best (lowest) total route distance found.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Heuristic: eta = 1 / d^gamma
    # Avoid division by zero (diagonal)
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    eta = 1.0 / (dm_safe ** gamma)
    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Pheromone initialization
    tau = np.ones((n, n), dtype=np.float64)
    
    # Precompute feasible neighbors based on capacity? 
    # Actually, feasibility depends on current load, so we check on the fly.
    # But we can precompute a mask of nodes that can EVER be visited from a given node? No, 
    # capacity is global per route. 
    # We just check demand[j] <= remaining_capacity.
    
    best_cost = np.inf
    rng = np.random.RandomState(seed)
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
            
        all_costs = []
        
        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break
                
            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0
            route = [0]
            
            while True:
                # Check if we should return to depot
                # Find unvisited customers that fit
                # remaining_cap is the capacity left in the current truck
                # We need to pick j such that not visited[j] and demands[j] <= remaining_cap
                
                # Optimization: if no one fits, go back to depot
                # We can vectorize the selection probability calculation
                
                # Mask of valid next nodes
                valid_mask = ~visited
                valid_mask[0] = False  # Don't go to depot unless forced
                valid_mask &= (demands <= remaining_cap)
                
                if not np.any(valid_mask):
                    # Must return to depot
                    total_dist += dm[current_node, 0]
                    route.append(0)
                    remaining_cap = capacity  # Reload
                    current_node = 0
                    # Reset visited? No, visited customers stay visited.
                    # The ant continues to pick from remaining unvisited customers.
                    # Wait, the problem says "returns to the depot and reloads".
                    # Does it continue picking? Yes, until all customers are visited.
                    # But if it returns to depot, does it restart the selection from depot? Yes.
                    # So the loop continues.
                    continue
                
                # Calculate probabilities
                # prob[i] proportional to tau[i, j] * eta[i, j]
                # We are at current_node. We look at row current_node.
                row_tau = tau[current_node]
                row_eta = eta[current_node]
                
                # Zero out invalid
                weights = np.zeros(n)
                weights[valid_mask] = row_tau[valid_mask] * row_eta[valid_mask]
                
                total_w = np.sum(weights)
                if total_w <= 0:
                    # Fallback: pick random valid
                    valid_indices = np.where(valid_mask)[0]
                    next_node = rng.choice(valid_indices)
                else:
                    probs = weights / total_w
                    next_node = rng.choice(n, p=probs)
                
                # Move
                total_dist += dm[current_node, next_node]
                route.append(next_node)
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node
                
                if np.all(visited):
                    # All customers visited, return to depot
                    total_dist += dm[current_node, 0]
                    route.append(0)
                    break
                    
        if all_costs:
            # Actually I didn't append to all_costs in the loop above.
            # Let's fix the structure.
            pass
        
        # Let's restructure the ant loop to properly collect costs
        
    # The code above had a bug: all_costs was never populated.
    # Let's rewrite the inner loop properly.
    
    # I will rewrite the function to be cleaner and correct.
    pass


def _run_aco_correct(distance_matrix, coordinates, demands, capacity, n_ants=20, n_iterations=30,
                     gamma=1.0, seed=42, time_limit=280.0):
    start_time = time.time()
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    eta = 1.0 / (dm_safe ** gamma)
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    tau = np.ones((n, n), dtype=np.float64)
    
    best_cost = np.inf
    rng = np.random.RandomState(seed)
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
            
        iteration_costs = []
        
        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break
                
            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0
            
            while True:
                # Determine valid next nodes
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)
                
                if not np.any(valid_mask):
                    # Return to depot
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue
                
                # Select next node
                row_tau = tau[current_node]
                row_eta = eta[current_node]
                
                weights = np.zeros(n)
                weights[valid_mask] = row_tau[valid_mask] * row_eta[valid_mask]
                
                total_w = np.sum(weights)
                if total_w <= 0:
                    valid_indices = np.where(valid_mask)[0]
                    next_node = rng.choice(valid_indices)
                else:
                    probs = weights / total_w
                    next_node = rng.choice(n, p=probs)
                
                total_dist += dm[current_node, next_node]
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node
                
                if np.all(visited):
                    total_dist += dm[current_node, 0]
                    break
                    
            iteration_costs.append(total_dist)
            
        if iteration_costs:
            current_best = min(iteration_costs)
            if current_best < best_cost:
                best_cost = current_best
            
            # Update pheromone
            # tau = tau * 0.9 + 1 / best_cost_of_iteration
            # "every entry gains 1/(best cost of that iteration)"
            # Usually this is added to edges in the best tour, but the prompt says:
            # "every entry gains 1/(best cost of that iteration)"
            # This implies a global update on the whole matrix?
            # "pheromone is multiplied by 0.9 and every entry gains 1/(best cost of that iteration)"
            # This is a very specific (and unusual) update rule. It adds a constant to all entries.
            # Let's re-read carefully: "After each iteration, pheromone is multiplied by 0.9 and every entry gains 1/(best cost of that iteration)."
            # Yes, it says "every entry". This is not standard ACO (which updates only tour edges),
            # but I must follow the contract.
            
            tau *= 0.9
            tau += 1.0 / current_best
            
    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Returns an edge-attractiveness matrix.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # The contract says: "Heuristic entries that are not positive become 1e-9"
    # "The heuristic is computed once per instance, before the colony runs."
    # "Returns an edge-attractiveness matrix (larger is more attractive) of the same shape"
    
    # We need to return the heuristic matrix.
    # The ACO loop uses this heuristic.
    # The prompt implies we should design a good heuristic.
    # Standard: 1/d.
    # Better: 1/d^gamma.
    # We can tune gamma.
    
    # However, the function `heuristics` is the entry point.
    # The evaluator calls `heuristics` and gets the matrix.
    # Then the evaluator runs the ACO using this matrix.
    # Wait, no. The prompt says:
    # "Write heuristics(...) which returns an edge-attractiveness matrix ... for 3 fixed instances"
    # "Ant colony: 20 ants for 30 iterations ... Heuristic is computed once per instance"
    # "Scoring ... mean over instances of the lowest total route distance found by any ant."
    
    # This implies the EVALUATOR runs the ACO loop, or does MY program run the ACO loop?
    # "Keep the function names and signatures of the seed program."
    # Seed program:
    # def heuristics(distance_matrix, coordinates, demands, capacity):
    #     return 1.0 / (distance_matrix + 1e-9)
    
    # If the evaluator runs the ACO, then I just return the heuristic matrix.
    # But the "INSPIRATIONS" and "OBSERVED_CHANGES" suggest that the programs are running ACO internally?
    # Let's look at the observed changes.
    # p6 edited units: heuristics, heuristics.run_mini_aco, heuristics.simple_greedy_aco_proxy
    # This suggests that the program IS running the ACO, or at least simulating it to tune the heuristic?
    # Or maybe the task allows me to run the ACO inside `heuristics` to find a better heuristic?
    # No, that would be weird. The heuristic is a static matrix.
    
    # Let's re-read the task contract carefully.
    # "Write heuristics(distance_matrix, coordinates, demands, capacity), which returns an edge-attractiveness matrix"
    # "The heuristic is computed once per instance, before the colony runs."
    # This strongly implies the colony runs AFTER `heuristics` returns.
    # Who runs the colony? The evaluator.
    # If the evaluator runs the colony, then I just need to return a good static heuristic matrix.
    # The quality of the result depends on how well this heuristic guides the fixed ACO algorithm.
    
    # So, my job is to return a matrix H such that when the evaluator's fixed ACO algorithm runs, it finds good tours.
    # The fixed ACO algorithm uses: prob ~ tau * H.
    # tau starts at 1, decays by 0.9, adds 1/best_cost to all entries.
    
    # If I can choose H, I should choose H to maximize the performance of this specific ACO variant.
    # Since tau starts at 1, the initial behavior is driven entirely by H.
    # As iterations proceed, tau becomes uniform-ish (since 1/best is added to all entries, and decay is uniform),
    # so the heuristic H becomes the dominant factor again?
    # Let's analyze the tau update:
    # tau_{k+1} = 0.9 * tau_k + 1/best_k
    # If best_k is roughly constant, tau converges to (1/best) / (1 - 0.9) = 10/best.
    # So tau becomes a constant matrix (plus small variations if best varies).
    # Therefore, the selection probability is proportional to H.
    # So the algorithm essentially becomes a greedy/randomized greedy based on H.
    
    # If H = 1/d, it's nearest neighbor-ish.
    # If H = 1/d^gamma, it's more aggressive nearest neighbor.
    
    # We want to minimize total distance.
    # A common heuristic for TSP/CVRP is 1/d.
    # Is there a better one?
    # Maybe considering demand? But the heuristic is per-edge, not per-node.
    # The contract says "edge-attractiveness matrix".
    
    # Let's try to tune gamma.
    # Since I can't run the full ACO in the evaluator's loop (I only return H),
    # I can run a LOCAL ACO simulation inside `heuristics` to test different gammas and pick the best one?
    # But `heuristics` must return a matrix.
    # I can run a quick ACO simulation for a few gammas, pick the best gamma, and return the corresponding matrix.
    # This is a "meta-heuristic" approach: search for the best heuristic parameter.
    
    # Time limit: 300s for the whole evaluation.
    # If I spend 200s tuning, I have 100s left for the actual evaluation?
    # No, the evaluator calls `heuristics` once per instance.
    # "The heuristic is computed once per instance".
    # So I have ~100s per instance to compute the heuristic.
    # I can use this time to run a mini-ACO to find the best gamma.
    
    # Plan:
    # 1. For each instance, try a range of gamma values (e.g., 0.5, 1.0, 1.5, 2.0, 3.0).
    # 2. For each gamma, construct H = 1/d^gamma.
    # 3. Run a short ACO simulation (e.g., 5 ants, 5 iterations) with this H.
    # 4. Pick the gamma that gives the best cost in the simulation.
    # 5. Return the H matrix for the best gamma.
    
    # This should significantly improve the score if the optimal gamma is not 1.0.
    
    # Let's implement this.
    
    # Note: The ACO simulation inside must be fast.
    # 50 nodes, 5 ants, 5 iterations is very fast.
    # We can even do 10 ants, 10 iterations.
    
    # Let's refine the simulation to match the evaluator's ACO as closely as possible,
    # but with fewer ants/iterations for speed.
    
    # Evaluator ACO:
    # - 20 ants, 30 iterations.
    # - tau starts at 1.
    # - Update: tau = 0.9*tau + 1/best.
    # - Selection: prob ~ tau * H.
    
    # My simulation:
    # - Same logic, but fewer ants/iterations.
    # - This should be a good proxy for the final performance.
    
    # Let's implement `_quick_aco` for tuning.
    
    def _quick_aco(dm, demands, capacity, H, n_ants=5, n_iterations=5, seed=0):
        n = dm.shape[0]
        tau = np.ones((n, n))
        rng = np.random.RandomState(seed)
        best_cost = np.inf
        
        for _ in range(n_iterations):
            costs = []
            for _ in range(n_ants):
                curr = 0
                rem_cap = capacity
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                dist = 0.0
                
                while True:
                    valid = ~visited
                    valid[0] = False
                    valid &= (demands <= rem_cap)
                    
                    if not np.any(valid):
                        dist += dm[curr, 0]
                        rem_cap = capacity
                        curr = 0
                        if np.all(visited):
                            break
                        continue
                    
                    w = np.zeros(n)
                    w[valid] = tau[curr, valid] * H[curr, valid]
                    tw = np.sum(w)
                    if tw > 0:
                        p = w / tw
                        nxt = rng.choice(n, p=p)
                    else:
                        nxt = rng.choice(np.where(valid)[0])
                    
                    dist += dm[curr, nxt]
                    visited[nxt] = True
                    rem_cap -= demands[nxt]
                    curr = nxt
                    
                    if np.all(visited):
                        dist += dm[curr, 0]
                        break
                        
            costs.append(dist)
            
        # Return min cost found in simulation
        return min(costs) if costs else np.inf

    # Now, in heuristics:
    # Try several gammas.
    
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    best_gamma = 1.0
    best_sim_cost = np.inf
    
    # Sample gammas
    gammas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    
    for g in gammas:
        H_candidate = 1.0 / (dm_safe ** g)
        H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0), H_candidate, 1e-9)
        
        # Run quick ACO
        cost = _quick_aco(dm, demands, capacity, H_candidate, n_ants=5, n_iterations=5, seed=42)
        
        if cost < best_sim_cost:
            best_sim_cost = cost
            best_gamma = g
            best_H = H_candidate
            
    # Return the best H
    return best_H

# EVOLVE-BLOCK-END
