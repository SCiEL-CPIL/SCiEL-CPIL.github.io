import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ant colony optimization of CVRP.
    Combines a demand-aware base heuristic with a structural refinement:
    we run a fast greedy proxy solver to pick the best combination of
    distance exponent (beta) and demand penalty strength (alpha), and also
    apply a mild depot-proximity bias to encourage tight return legs.
    """
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    n = dm.shape[0]
    
    # Deadline guard: return a safe default if we run out of time
    start_time = time.time()
    DEADLINE = 320.0  # 330s wall-clock limit, leave 10s margin
    
    # Default safe heuristic: inverse distance with mild demand penalty
    def safe_default():
        base = 1.0 / (dm + 1e-9)
        alpha = 0.1
        demand_penalty = 1.0 / (1.0 + alpha * dem)
        adjusted = base * demand_penalty[np.newaxis, :]
        adjusted[~np.isfinite(adjusted)] = 1e-9
        adjusted[adjusted <= 0] = 1e-9
        return adjusted
    
    # Candidate parameters to explore
    betas = [1.0, 1.2, 1.5, 2.0, 2.5]
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    def calculate_heuristic(beta, alpha, use_depot_bias=True):
        """Compute heuristic for given beta, alpha, and depot bias option."""
        with np.errstate(divide='ignore', invalid='ignore'):
            base = 1.0 / (dm ** beta)
        
        # Demand-aware penalty on destination nodes
        demand_penalty = 1.0 / (1.0 + alpha * dem)
        adjusted = base * demand_penalty[np.newaxis, :]
        
        # Depot proximity bias: boost edges from nodes close to the depot
        if use_depot_bias:
            depot_dist = dm[:, 0].copy()
            max_dd = np.max(depot_dist)
            if max_dd > 0:
                norm_dd = depot_dist / max_dd
            else:
                norm_dd = np.zeros(n)
            gamma = 0.1
            depot_bias = 1.0 + gamma * (1.0 - norm_dd)
            adjusted = adjusted * depot_bias[:, np.newaxis]
        
        # Ensure all values are positive and finite
        adjusted[~np.isfinite(adjusted)] = 1e-9
        adjusted[adjusted <= 0] = 1e-9
        
        return adjusted
    
    def proxy_solve(eta):
        """Greedy construction using eta as attractiveness."""
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0
        current = 0
        load = 0.0
        
        for _ in range(n - 1):
            best_j = -1
            best_score = -1.0
            
            for j in range(1, n):
                if not visited[j]:
                    if load + dem[j] <= capacity:
                        score = eta[current, j]
                        if score > best_score:
                            best_score = score
                            best_j = j
            
            if best_j == -1:
                total_dist += dm[current, 0]
                current = 0
                load = 0.0
            else:
                total_dist += dm[current, best_j]
                load += dem[best_j]
                visited[best_j] = True
                current = best_j
        
        if current != 0:
            total_dist += dm[current, 0]
            
        return total_dist
    
    # Search for best (beta, alpha) combination
    best_cost = float('inf')
    best_beta = 1.0
    best_alpha = 0.1
    best_use_bias = True
    
    # Evaluate a subset of combinations to stay within time budget
    # We'll do a coarse-to-fine search
    combinations = []
    for beta in betas:
        for alpha in alphas:
            combinations.append((beta, alpha, True))
            combinations.append((beta, alpha, False))
    
    # Shuffle to avoid bias in ordering, but keep deterministic via seed
    rng = np.random.RandomState(42)
    rng.shuffle(combinations)
    
    # Time budget for search: use up to 60% of available time
    search_time_limit = 180.0  # 3 minutes max for search
    
    for beta, alpha, use_bias in combinations:
        elapsed = time.time() - start_time
        if elapsed > search_time_limit:
            break
            
        eta = calculate_heuristic(beta, alpha, use_bias)
        cost = proxy_solve(eta)
        
        if cost < best_cost:
            best_cost = cost
            best_beta = beta
            best_alpha = alpha
            best_use_bias = use_bias
    
    # Final check: if we haven't found anything better, use default
    if best_cost == float('inf'):
        return safe_default()
    
    # Return the best heuristic found
    result = calculate_heuristic(best_beta, best_alpha, best_use_bias)
    
    # Final safety check
    result[~np.isfinite(result)] = 1e-9
    result[result <= 0] = 1e-9
    
    return result
