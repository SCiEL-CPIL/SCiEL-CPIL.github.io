import numpy as np

def prime_number_theorem_partial_function():
    best_score = 0.0
    best_domain = np.array([1], dtype=np.int64)
    best_values = np.array([0.0], dtype=np.float64)
    
    # Heuristic approach: f(k) = -C * ln(k)/k
    # This maximizes the score for a given constraint tightness.
    # We scan for optimal K.
    
    K_values = [50, 100, 200, 500, 1000, 2000]
    
    for K in K_values:
        if K > 200000:
            break
            
        ks = np.arange(2, K + 1, dtype=np.float64)
        weights = np.log(ks) / ks
        
        x_max = 10 * K
        xs = np.arange(1, x_max + 1, dtype=np.float64)
        
        # Compute D(x) = sum_k weights[k] * {x/k}
        # Memory check: (K-1) * (10K) floats.
        # For K=2000: 1999 * 20000 ~ 40M elements ~ 320MB. Acceptable.
        # For K=5000: 4999 * 50000 ~ 250M elements ~ 2GB. Might be risky.
        # We stick to K <= 2000 in this vectorized block.
        
        if K > 2000:
            continue

        # Calculate fractional parts {x/k}
        # xs[None, :] / ks[:, None]
        # This creates a (K-1, 10K) array.
        ratios = xs[None, :] / ks[:, None]
        fractional_parts = ratios - np.floor(ratios)
        
        # D(x) is the dot product of weights and each column of fractional_parts
        D = np.dot(weights, fractional_parts)
        
        max_D = np.max(D)
        
        if max_D <= 0:
            continue
            
        C = 1.0 / max_D
        
        # Check bounds for f(k) = -C * weights
        f_vals = -C * weights
        
        # If any f(k) is outside [-10, 10], we are limited by the bound.
        # The constraint M(x) <= 1 might be loose if we reduce C to satisfy bounds.
        # But if we reduce C, M(x) decreases, so it's still feasible.
        # We want to maximize a = C * sum(weights^2).
        # So we want largest C.
        # C is limited by 1/max_D and by 10/max(weights).
        
        C_bound = 10.0 / np.max(weights)
        if C > C_bound:
            C = C_bound
            f_vals = -C * weights
            
        # Calculate score
        a = C * np.sum(weights * weights)
        
        if a > best_score:
            best_score = a
            domain = np.concatenate(([1], ks.astype(np.int64)))
            values = np.empty(K, dtype=np.float64)
            
            # f(1) calculation
            # f(1) = - sum_{k=2}^K f(k)/k = C * sum weights/k
            f1 = C * np.sum(weights / ks)
            values[0] = f1
            values[1:] = f_vals
            
            best_domain = domain
            best_values = values

    # Fallback
    if best_score < 0.1:
        best_domain = np.array([1, 2], dtype=np.int64)
        best_values = np.array([1.0, -2.0], dtype=np.float64)

    return best_domain, best_values
