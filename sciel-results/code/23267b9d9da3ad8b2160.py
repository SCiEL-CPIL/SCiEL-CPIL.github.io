"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _aco_simulate(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """Fast simulation of the ACO to evaluate a heuristic's quality."""
    n = len(demands)
    rng = np.random.RandomState(seed)
    pheromone = np.ones_like(distance_matrix)
    best_cost = float('inf')
    demands_arr = demands.astype(float)
    
    for it in range(n_iters):
        iter_best = float('inf')
        for ant in range(n_ants):
            route_cost = 0.0
            current = 0
            remaining_cap = capacity
            visited = set()
            
            while True:
                # Find feasible neighbors
                feasible = []
                for j in range(1, n):
                    if j not in visited and demands_arr[j] <= remaining_cap:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    route_cost += distance_matrix[current, 0]
                    break
                
                # Compute probabilities
                probs = np.zeros(len(feasible))
                for idx, j in enumerate(feasible):
                    probs[idx] = pheromone[current, j] * heuristic[current, j]
                
                total = probs.sum()
                if total < 1e-30:
                    # Pick random feasible
                    j = feasible[rng.randint(len(feasible))]
                else:
                    probs /= total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(feasible):
                        idx = len(feasible) - 1
                    j = feasible[idx]
                
                route_cost += distance_matrix[current, j]
                visited.add(j)
                remaining_cap -= demands_arr[j]
                current = j
            
            # Add final return to depot
            route_cost += distance_matrix[current, 0]
            if route_cost < iter_best:
                iter_best = route_cost
        
        if iter_best < best_cost:
            best_cost = iter_best
        
        # Update pheromone
        pheromone *= 0.9
        if iter_best > 0:
            pheromone += 1.0 / iter_best
    
    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """Compute heuristic matrix using parameterized family with simulation-based tuning."""
    n = len(demands)
    t_start = time.time()
    deadline = t_start + 250.0  # Safety margin before 300s limit
    
    # Parameterized heuristic family:
    # eta[i,j] = 1 / (dist[i,j]^alpha * (1 + beta * demand_j/capacity)^gamma)
    # where alpha controls distance sensitivity, beta controls demand penalty, gamma controls penalty strength
    
    # Candidate parameter sets (alpha, beta, gamma)
    candidates = [
        (1.0, 0.0, 0.0),      # baseline: inverse distance
        (1.5, 0.0, 0.0),      # stronger distance preference
        (2.0, 0.0, 0.0),      # even stronger distance
        (1.0, 1.0, 1.0),      # moderate demand penalty
        (1.0, 2.0, 1.0),      # stronger demand penalty
        (1.0, 1.0, 2.0),      # stronger demand penalty (exponent)
        (1.5, 1.0, 1.0),      # combined
        (1.5, 2.0, 1.5),      # combined stronger
        (2.0, 1.0, 1.0),      # strong distance + demand
        (1.0, 0.5, 1.0),      # mild demand penalty
        (1.5, 0.5, 1.5),      # mild demand penalty, stronger distance
        (2.0, 1.5, 1.0),      # strong both
    ]
    
    # Also try a spatial clustering heuristic:
    # eta[i,j] = 1 / (dist[i,j] * (1 + dist_to_depot_j / max_dist_to_depot)^delta)
    # This encourages visiting nearby customers in coherent clusters
    
    # Compute auxiliary quantities
    dist_to_depot = distance_matrix[0, :]  # distance from depot to each node
    max_dd = dist_to_depot.max() if dist_to_depot.max() > 0 else 1.0
    demands_arr = demands.astype(float)
    
    # Evaluate candidates via simulation
    best_heuristic = 1.0 / (distance_matrix + 1e-9)
    best_score = float('inf')
    
    # First, quickly evaluate a subset
    for alpha, beta, gamma in candidates:
        if time.time() > deadline:
            break
        try:
            if beta == 0.0 and gamma == 0.0:
                h = 1.0 / (distance_matrix ** alpha + 1e-9)
            else:
                demand_factor = 1.0 + beta * (demands_arr / capacity)
                # For each edge (i,j), the demand penalty applies to j
                demand_penalty = np.ones((n, n))
                for j in range(1, n):
                    demand_penalty[:, j] = demand_factor[j] ** gamma
                h = 1.0 / (distance_matrix ** alpha * demand_penalty + 1e-9)
            
            # Ensure positive
            h = np.where(h > 0, h, 1e-9)
            
            score = _aco_simulate(distance_matrix, coordinates, demands, capacity, h, 
                                   n_ants=10, n_iters=10, seed=42)
            if score < best_score:
                best_score = score
                best_heuristic = h.copy()
        except Exception:
            continue
    
    # Try spatial clustering heuristics
    for delta in [0.5, 1.0, 1.5, 2.0]:
        if time.time() > deadline:
            break
        try:
            # eta[i,j] = 1 / (dist[i,j] * (1 + dist_to_depot[j]/max_dd)^delta)
            cluster_factor = np.ones((n, n))
            for j in range(1, n):
                cluster_factor[:, j] = (1.0 + dist_to_depot[j] / max_dd) ** delta
            h = 1.0 / (distance_matrix * cluster_factor + 1e-9)
            h = np.where(h > 0, h, 1e-9)
            
            score = _aco_simulate(distance_matrix, coordinates, demands, capacity, h,
                                   n_ants=10, n_iters=10, seed=42)
            if score < best_score:
                best_score = score
                best_heuristic = h.copy()
        except Exception:
            continue
    
    # Try combined: distance^alpha * demand_penalty * cluster_factor
    for alpha, beta, gamma, delta in [(1.0, 1.0, 1.0, 0.5), (1.5, 1.0, 1.0, 1.0), (1.0, 2.0, 1.0, 1.0)]:
        if time.time() > deadline:
            break
        try:
            demand_factor = 1.0 + beta * (demands_arr / capacity)
            demand_penalty = np.ones((n, n))
            for j in range(1, n):
                demand_penalty[:, j] = demand_factor[j] ** gamma
            
            cluster_factor = np.ones((n, n))
            for j in range(1, n):
                cluster_factor[:, j] = (1.0 + dist_to_depot[j] / max_dd) ** delta
            
            h = 1.0 / (distance_matrix ** alpha * demand_penalty * cluster_factor + 1e-9)
            h = np.where(h > 0, h, 1e-9)
            
            score = _aco_simulate(distance_matrix, coordinates, demands, capacity, h,
                                   n_ants=10, n_iters=10, seed=42)
            if score < best_score:
                best_score = score
                best_heuristic = h.copy()
        except Exception:
            continue
    
    # If we have time, do a finer search around the best parameters found
    if time.time() < deadline - 30:
        # Fine-tune: perturb the best heuristic slightly
        # Try scaling the best heuristic
        for scale in [0.5, 0.8, 1.2, 1.5, 2.0]:
            if time.time() > deadline:
                break
            try:
                h = best_heuristic * scale
                h = np.where(h > 0, h, 1e-9)
                score = _aco_simulate(distance_matrix, coordinates, demands, capacity, h,
                                       n_ants=10, n_iters=10, seed=42)
                if score < best_score:
                    best_score = score
                    best_heuristic = h.copy()
            except Exception:
                continue
    
    # Final validation
    best_heuristic = np.asarray(best_heuristic, dtype=float)
    if best_heuristic.shape != distance_matrix.shape:
        best_heuristic = 1.0 / (distance_matrix + 1e-9)
    best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
    
    return best_heuristic
# EVOLVE-BLOCK-END
