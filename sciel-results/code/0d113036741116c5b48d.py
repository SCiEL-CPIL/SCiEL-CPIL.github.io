import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin
    
    n = distance_matrix.shape[0]
    dm = distance_matrix
    d = demands
    
    # 1. Compute Savings Matrix
    # s[i, j] = d[i, 0] + d[0, j] - d[i, j]
    # This represents the distance saved by merging routes to i and j into a single route segment i-j.
    
    # Vectorized savings calculation
    # d_i_0 = dm[:, 0]
    # d_0_j = dm[0, :]
    # s = d_i_0[:, None] + d_0_j[None, :] - dm
    
    d_i_0 = dm[:, 0]
    d_0_j = dm[0, :]
    
    # Initialize savings matrix
    savings = np.zeros((n, n), dtype=float)
    
    # Calculate for customer-customer pairs (i > 0, j > 0)
    # We can compute for all i, j and then mask
    s_all = d_i_0[:, None] + d_0_j[None, :] - dm
    
    # Mask out depot edges and diagonal?
    # For depot edges, savings is not defined in the same way, or is 0.
    # We will set heuristic for depot edges based on distance only.
    
    # Normalize savings for customer-customer pairs
    # Find max savings among valid pairs
    mask_customers = np.zeros((n, n), dtype=bool)
    mask_customers[1:, 1:] = True
    
    valid_savings = s_all[mask_customers]
    max_savings = np.max(valid_savings) if len(valid_savings) > 0 else 1.0
    
    if max_savings <= 0:
        max_savings = 1.0
        
    savings_norm = s_all / max_savings
    # Set savings to 0 where not applicable (depot edges)
    savings_norm[~mask_customers] = 0.0
    
    # 2. Base Heuristic: 1 / Distance
    # Avoid division by zero (though dm[0,0]=0, we don't use diagonal)
    base = 1.0 / (dm + 1e-9)
    
    # 3. Combine
    # eta[i, j] = base[i, j] * (1 + lambda * savings_norm[i, j])
    # This boosts edges that are both close and have high savings potential.
    
    # We need to tune lambda.
    # Let's define a function to evaluate a heuristic using a greedy ant.
    
    def _greedy_tour_cost(h, dm, demands, cap):
        current = 0
        remaining_cap = cap
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0
        
        while True:
            eligible = [j for j in range(1, n) if not visited[j] and demands[j] <= remaining_cap]
            if not eligible:
                total_dist += dm[current, 0]
                current = 0
                remaining_cap = cap
                if visited[1:].all():
                    break
                continue
            
            best_j = max(eligible, key=lambda j: h[current, j])
            total_dist += dm[current, best_j]
            remaining_cap -= demands[best_j]
            visited[best_j] = True
            current = best_j
            
            if visited[1:].all():
                total_dist += dm[current, 0]
                break
        
        return total_dist

    best_lambda = 1.0
    best_cost = float('inf')
    
    # Grid search for lambda
    # Lambda scales the savings boost.
    # If lambda is too high, it might favor high-saving edges even if they are far.
    # If lambda is 0, it's pure distance.
    
    lambdas = np.linspace(0, 5, 20)
    
    for lam in lambdas:
        if time.time() > deadline - 10:
            break
            
        h_test = base * (1.0 + lam * savings_norm)
        
        # Ensure positivity
        h_test = np.where(h_test > 0, h_test, 1e-9)
        
        cost = _greedy_tour_cost(h_test, dm, d, capacity)
        
        if cost < best_cost:
            best_cost = cost
            best_lambda = lam
            
    # Final Heuristic
    heuristic = base * (1.0 + best_lambda * savings_norm)
    
    # Ensure positivity and finiteness
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Shape check
    if heuristic.shape != dm.shape:
        heuristic = 1.0 / (dm + 1e-9)
        
    return heuristic
