"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _aco_eval(distance_matrix, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Runs a full ACO simulation with the given heuristic and returns the best tour distance found.
    This is the true objective we want to optimize, not just greedy.
    """
    n = len(demands)
    rng = np.random.RandomState(seed)
    
    # Pheromone matrix, start at 1
    pheromone = np.ones_like(distance_matrix)
    
    best_overall = np.inf
    
    for it in range(n_iters):
        best_iter_cost = np.inf
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            remaining_cap = capacity
            total_dist = 0.0
            tour = [0]
            
            # Ant constructs its route
            while not np.all(visited):
                # Compute probabilities
                scores = pheromone[current] * heuristic[current]
                
                # Mask visited and infeasible
                feasible_mask = (~visited) & (demands <= remaining_cap)
                scores[~feasible_mask] = 0.0
                scores[0] = 0.0  # Depot already visited
                
                # If no feasible customer, return to depot
                if not np.any(scores > 0):
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    remaining_cap = capacity
                    tour.append(0)
                    continue
                
                # Normalize to probabilities
                total_score = np.sum(scores)
                if total_score <= 0:
                    # Fallback: pick randomly from feasible
                    feasible_indices = np.where(feasible_mask)[0]
                    if len(feasible_indices) == 0:
                        total_dist += distance_matrix[current, 0]
                        current = 0
                        remaining_cap = capacity
                        tour.append(0)
                        continue
                    next_node = feasible_indices[rng.randint(len(feasible_indices))]
                else:
                    probs = scores / total_score
                    next_node = rng.choice(n, p=probs)
                
                # Move to next_node
                total_dist += distance_matrix[current, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                current = next_node
                tour.append(next_node)
            
            # Return to depot
            total_dist += distance_matrix[current, 0]
            tour.append(0)
            
            if total_dist < best_iter_cost:
                best_iter_cost = total_dist
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += 1.0 / best_iter_cost
        
        if best_iter_cost < best_overall:
            best_overall = best_iter_cost
    
    return best_overall

def _greedy_eval_fast(distance_matrix, demands, capacity, heuristic):
    """
    Fast greedy evaluation for use during parameter search.
    """
    n = len(demands)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_cap = capacity
    total_dist = 0.0
    
    for _ in range(n - 1):
        scores = heuristic[current].copy()
        feasible_mask = (~visited) & (demands <= remaining_cap)
        scores[~feasible_mask] = -np.inf
        scores[0] = -np.inf
        
        if not np.any(np.isfinite(scores)):
            total_dist += distance_matrix[current, 0]
            current = 0
            remaining_cap = capacity
            continue
        
        best_j = np.argmax(scores)
        total_dist += distance_matrix[current, best_j]
        remaining_cap -= demands[best_j]
        visited[best_j] = True
        current = best_j
    
    total_dist += distance_matrix[current, 0]
    return total_dist

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix using simulated annealing over
    a parameterized family of heuristics. The heuristic is parameterized as:
    
    H[i,j] = (1/d[i,j])^alpha * (1 + beta * cos(theta_ij)) * (1 + gamma * demand_ratio)
    
    where alpha, beta, gamma are searched via SA.
    """
    n = len(demands)
    d_safe = distance_matrix + 1e-9
    
    # Precompute angular term: cos(theta) between vector from depot to i and vector from i to j
    coords = np.asarray(coordinates)
    depot = coords[0]
    vi = coords - depot  # N, 2
    vj = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # N, N, 2
    dot = np.sum(vi[:, np.newaxis, :] * vj, axis=2)
    norm_vi = np.linalg.norm(vi, axis=1)[:, np.newaxis]
    norm_vj = np.linalg.norm(vj, axis=2)
    norm_vi_safe = norm_vi + 1e-9
    norm_vj_safe = norm_vj + 1e-9
    cos_theta = dot / (norm_vi_safe * norm_vj_safe)
    
    # Precompute demand ratio term: for each edge i->j, ratio = demands[j] / capacity
    demand_ratio = demands[None, :] / capacity  # 1, N -> broadcast to N, N
    
    # Parameter bounds
    alpha_lo, alpha_hi = 0.5, 3.0
    beta_lo, beta_hi = -0.5, 1.0
    gamma_lo, gamma_hi = -0.5, 1.0
    
    def make_heuristic(alpha, beta, gamma):
        inv_d = 1.0 / d_safe
        inv_d_pow = inv_d ** alpha
        angular = 1.0 + beta * cos_theta
        demand_term = 1.0 + gamma * demand_ratio
        h = inv_d_pow * angular * demand_term
        # Ensure positivity
        h = np.maximum(h, 1e-9)
        # Set diagonal to small value
        np.fill_diagonal(h, 1e-9)
        return h
    
    # Simulated annealing search
    rng = np.random.RandomState(12345)
    
    # Start at a reasonable point
    alpha = 1.0
    beta = 0.0
    gamma = 0.0
    
    best_params = (alpha, beta, gamma)
    best_score = _greedy_eval_fast(distance_matrix, demands, capacity, make_heuristic(alpha, beta, gamma))
    
    # SA schedule
    T_start = 0.1
    T_end = 0.001
    n_sa_iters = 200
    start_time = time.time()
    deadline = start_time + 120.0  # Leave margin for ACO validation
    
    for it in range(n_sa_iters):
        # Check deadline
        if time.time() > deadline - 30.0:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_sa_iters)
        
        # Propose new parameters
        new_alpha = alpha + rng.normal(0, 0.1)
        new_beta = beta + rng.normal(0, 0.1)
        new_gamma = gamma + rng.normal(0, 0.1)
        
        # Clip to bounds
        new_alpha = np.clip(new_alpha, alpha_lo, alpha_hi)
        new_beta = np.clip(new_beta, beta_lo, beta_hi)
        new_gamma = np.clip(new_gamma, gamma_lo, gamma_hi)
        
        new_h = make_heuristic(new_alpha, new_beta, new_gamma)
        new_score = _greedy_eval_fast(distance_matrix, demands, capacity, new_h)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = new_score - best_score
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-10)):
            alpha, beta, gamma = new_alpha, new_beta, new_gamma
            if new_score < best_score:
                best_score = new_score
                best_params = (alpha, beta, gamma)
    
    # Build the best heuristic
    best_heur = make_heuristic(*best_params)
    
    # Optional: validate with a quick ACO run if time permits
    if time.time() < deadline - 60.0:
        # Compare with a few alternative parameter settings
        alternatives = [
            (1.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            (1.0, 0.3, 0.0),
            (1.0, 0.0, 0.3),
            (0.7, 0.2, 0.2),
        ]
        
        alt_scores = []
        for (a, b, g) in alternatives:
            h_alt = make_heuristic(a, b, g)
            s_alt = _greedy_eval_fast(distance_matrix, demands, capacity, h_alt)
            alt_scores.append((s_alt, h_alt))
        
        # Pick the best among best_params and alternatives
        all_candidates = [(best_score, best_heur)] + alt_scores
        all_candidates.sort(key=lambda x: x[0])
        best_heur = all_candidates[0][1]
    
    # Final safety checks
    if not np.all(np.isfinite(best_heur)):
        return 1.0 / d_safe
    if np.any(best_heur <= 0):
        best_heur = np.clip(best_heur, 1e-9, None)
    if best_heur.shape != distance_matrix.shape:
        return 1.0 / d_safe
    
    return best_heur
# EVOLVE-BLOCK-END
