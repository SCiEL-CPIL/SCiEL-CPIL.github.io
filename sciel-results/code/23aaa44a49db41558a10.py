import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the edge-attractiveness matrix for the ACO heuristic.
    Uses an instance-specific optimization of heuristic parameters 
    (exponent and demand weight) based on greedy construction cost.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    try:
        d = np.asarray(distance_matrix, dtype=np.float64)
        N = d.shape[0]
        
        if not np.all(np.isfinite(d)):
            return np.ones_like(d)
            
        # Ensure demands are float for calculation
        dem = np.asarray(demands, dtype=np.float64)
        cap = float(capacity)
        
        # Parametric Family:
        # H[i, j] = 1 / (dist[i, j] + eps)^p * (1 + w * (C - dem[j]) / C)
        # p: distance exponent
        # w: demand weight (positive = prefer low demand, negative = prefer high demand)
        
        eps = 1e-9
        
        # Grid search for p and w
        # p range: 0.5 to 3.0
        # w range: -1.0 to 1.0
        p_values = np.linspace(0.5, 3.0, 15)
        w_values = np.linspace(-1.0, 1.0, 15)
        
        best_cost = np.inf
        best_p = 1.0
        best_w = 0.0
        
        # Pre-compute demand factor for each customer j
        # demand_factor[j] = (C - dem[j]) / C
        # Note: depot (0) has demand 0, so factor is 1.
        # We only apply this factor to the destination customer j in the heuristic.
        
        for p in p_values:
            if time.time() > deadline:
                break
            for w in w_values:
                if time.time() > deadline:
                    break
                
                # Calculate heuristic matrix for this (p, w)
                # dist^p
                d_safe = d + eps
                d_pow = d_safe ** p
                inv_d = 1.0 / d_pow
                
                # Apply demand weight to columns (destination j)
                # H[i, j] = inv_d[i, j] * (1 + w * factor[j])
                # factor[j] is (C - dem[j]) / C
                factors = (cap - dem) / cap
                # If cap is 0 or dem is 0, handle gracefully. 
                # In task, cap=50, dem in [1,9] for customers, 0 for depot.
                
                # We want to multiply column j by (1 + w * factors[j])
                # Create a vector of multipliers for each column j
                multipliers = 1.0 + w * factors
                
                # H = inv_d * multipliers[None, :]
                H = inv_d * multipliers[np.newaxis, :]
                
                # Ensure positive
                H = np.where(H > 0, H, 1e-9)
                
                # Evaluate H using a fast Greedy Construction
                # This simulates the "quality" of the heuristic by seeing how well
                # it guides a greedy route builder.
                
                cost = _evaluate_greedy(d, dem, cap, H)
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p
                    best_w = w
                    
        # Compute final heuristic with best parameters
        d_safe = d + eps
        d_pow = d_safe ** best_p
        inv_d = 1.0 / d_pow
        
        factors = (cap - dem) / cap
        multipliers = 1.0 + best_w * factors
        
        H_final = inv_d * multipliers[np.newaxis, :]
        
        # Ensure positive and finite
        H_final = np.where(np.isfinite(H_final) & (H_final > 0), H_final, 1e-9)
        
        return H_final
        
    except Exception:
        # Fallback to standard 1/d
        try:
            d = np.asarray(distance_matrix, dtype=np.float64)
            return 1.0 / (d + 1e-9)
        except:
            return np.ones((50, 50))

def _evaluate_greedy(d, dem, cap, H):
    """
    Fast greedy evaluation of a heuristic matrix.
    Returns total distance of the constructed routes.
    """
    N = d.shape[0]
    unvisited = set(range(1, N))
    current = 0
    load = 0.0
    total_dist = 0.0
    
    # We simulate the ACO ant behavior:
    # Pick next node j from unvisited such that load + dem[j] <= cap
    # Probability proportional to H[current, j] (or just argmax for greedy speed)
    # To be fast, we use argmax (greedy) rather than sampling.
    # This provides a lower bound on potential quality if the heuristic is good.
    
    while unvisited:
        # Find feasible neighbors
        # We need to check load constraint.
        # For speed, we can pre-filter or just check in loop.
        # N=50, so O(N) check is fine.
        
        best_j = -1
        best_h = -1.0
        
        for j in unvisited:
            if load + dem[j] <= cap:
                if H[current, j] > best_h:
                    best_h = H[current, j]
                    best_j = j
        
        if best_j == -1:
            # Return to depot
            total_dist += d[current, 0]
            current = 0
            load = 0.0
            # Retry finding a neighbor from depot
            # Loop continues, unvisited is same
            # To avoid infinite loop if no one fits (should not happen if total dem < cap * num_vehicles)
            # But if no one fits from depot, it's an error in problem definition or logic.
            # Assuming at least one customer fits.
            continue
            
        # Move to best_j
        total_dist += d[current, best_j]
        load += dem[best_j]
        current = best_j
        unvisited.remove(best_j)
        
    # Return to depot
    total_dist += d[current, 0]
    
    return total_dist
