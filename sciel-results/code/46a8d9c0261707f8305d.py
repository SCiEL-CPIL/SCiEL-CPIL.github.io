# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    DEADLINE = 600.0  # safety margin before 630s

    def check_feasibility_fast(domain, values, K):
        """Check M(x) <= 1.0001 for all x in [1, 10K] using efficient range updates."""
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        for k, f in zip(domain, values):
            if f == 0.0:
                continue
            # For each j, add f to M[j*k : (j+1)*k]
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k, max_x + 1)
                M[lo:hi] += f
                j += 1
        
        return np.max(M[1:max_x + 1])

    def compute_score(domain, values):
        """Compute the objective a = -sum_k f(k)*log(k)/k"""
        return -np.sum(values * np.log(domain) / domain)

    def compute_max_h_and_coeff(K, g_vals, ks_int):
        """
        Compute max_x h(x) where h(x) = sum_k g(k)*floor(x/k),
        and also compute a_coeff = -sum_k g(k)*log(k)/k for k>1.
        g_vals is for k=2..K (length K-1).
        """
        max_x = 10 * K
        floor_sum = np.zeros(max_x + 1, dtype=np.float64)
        
        # For each k from 2 to K, add g(k) * floor(x/k)
        for idx, k_int in enumerate(range(2, K + 1)):
            g_k = g_vals[idx]
            if g_k == 0.0:
                continue
            for j in range(1, max_x // k_int + 1):
                lo = j * k_int
                hi = min((j + 1) * k_int, max_x + 1)
                floor_sum[lo:hi] += g_k
        
        h_vals = floor_sum[1:max_x + 1]
        max_h = np.max(h_vals)
        
        # a_coeff = -sum_{k=2}^K g(k)*log(k)/k
        ks_float = ks_int.astype(np.float64)
        a_coeff = -np.sum(g_vals * np.log(ks_float[1:]) / ks_float[1:])
        
        return max_h, a_coeff

    def compute_f1(K, g_vals):
        """Compute f(1) = sum_{k=2}^K g(k)/k so that sum f(k)/k = 0"""
        ks_float = np.arange(2, K + 1, dtype=np.float64)
        return np.sum(g_vals / ks_float)

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Strategy 1: Exponential smoothing g(k) = -exp(-k/alpha)/k
    # Try various alpha and K
    alpha_values = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    K_values_exp = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    for alpha in alpha_values:
        if time.time() > DEADLINE - 60:
            break
        
        for K in K_values_exp:
            if time.time() > DEADLINE - 60:
                break
            
            if K < 2:
                continue
            
            ks_int = np.arange(2, K + 1, dtype=np.int64)
            ks_float = ks_int.astype(np.float64)
            
            # g(k) = -exp(-k/alpha)/k
            g_vals = -np.exp(-ks_float / alpha) / ks_float
            
            # Check if g_vals are reasonable
            if np.all(g_vals == 0):
                continue
            
            max_h, a_coeff = compute_max_h_and_coeff(K, g_vals, ks_int)
            
            if max_h <= 0:
                continue
            
            # c_max = 1.0001 / max_h
            c = 1.0001 / max_h
            
            # a = c * a_coeff (since a = -sum f(k)log(k)/k = -c*sum g(k)log(k)/k = c*a_coeff)
            a_val = c * a_coeff
            
            if a_val > best_score:
                # Build domain and values
                domain = np.concatenate([[1], ks_int])
                values = np.empty(K, dtype=np.float64)
                
                f1 = compute_f1(K, g_vals)
                values[0] = c * f1
                values[1:] = c * g_vals
                
                # Verify feasibility
                max_val = check_feasibility_fast(domain, values, K)
                
                if max_val <= 1.0001:
                    best_score = a_val
                    best_domain = domain.copy()
                    best_values = values.copy()
                else:
                    # If infeasible, try slightly smaller c
                    c = 0.999 * c
                    values[0] = c * f1
                    values[1:] = c * g_vals
                    max_val = check_feasibility_fast(domain, values, K)
                    if max_val <= 1.0001:
                        a_val = c * a_coeff
                        if a_val > best_score:
                            best_score = a_val
                            best_domain = domain.copy()
                            best_values = values.copy()
        
        if time.time() > DEADLINE - 60:
            break

    # Strategy 2: Power law g(k) = -1/k^p (keep from original, but with more K values)
    p_values = [1.0, 1.1, 1.2, 1.5, 2.0]
    K_values_pow = [100, 500, 1000, 5000, 10000, 50000]

    for p in p_values:
        if time.time() > DEADLINE - 60:
            break
        
        for K in K_values_pow:
            if time.time() > DEADLINE - 60:
                break
            
            if K < 2:
                continue
            
            ks_int = np.arange(2, K + 1, dtype=np.int64)
            ks_float = ks_int.astype(np.float64)
            
            # g(k) = -1/k^p
            g_vals = -1.0 / (ks_float ** p)
            
            max_h, a_coeff = compute_max_h_and_coeff(K, g_vals, ks_int)
            
            if max_h <= 0:
                continue
            
            c = 1.0001 / max_h
            a_val = c * a_coeff
            
            if a_val > best_score:
                domain = np.concatenate([[1], ks_int])
                values = np.empty(K, dtype=np.float64)
                
                f1 = compute_f1(K, g_vals)
                values[0] = c * f1
                values[1:] = c * g_vals
                
                max_val = check_feasibility_fast(domain, values, K)
                
                if max_val <= 1.0001:
                    best_score = a_val
                    best_domain = domain.copy()
                    best_values = values.copy()
                else:
                    c = 0.999 * c
                    values[0] = c * f1
                    values[1:] = c * g_vals
                    max_val = check_feasibility_fast(domain, values, K)
                    if max_val <= 1.0001:
                        a_val = c * a_coeff
                        if a_val > best_score:
                            best_score = a_val
                            best_domain = domain.copy()
                            best_values = values.copy()
        
        if time.time() > DEADLINE - 60:
            break

    # Strategy 3: Combined exponential-power: g(k) = -exp(-k/alpha) / k^p
    for alpha in [10, 50, 200, 1000]:
        if time.time() > DEADLINE - 30:
            break
        
        for p in [1.0, 1.5, 2.0]:
            if time.time() > DEADLINE - 30:
                break
            
            for K in [1000, 5000, 10000]:
                if time.time() > DEADLINE - 30:
                    break
                
                if K < 2:
                    continue
                
                ks_int = np.arange(2, K + 1, dtype=np.int64)
                ks_float = ks_int.astype(np.float64)
                
                g_vals = -np.exp(-ks_float / alpha) / (ks_float ** p)
                
                max_h, a_coeff = compute_max_h_and_coeff(K, g_vals, ks_int)
                
                if max_h <= 0:
                    continue
                
                c = 1.0001 / max_h
                a_val = c * a_coeff
                
                if a_val > best_score:
                    domain = np.concatenate([[1], ks_int])
                    values = np.empty(K, dtype=np.float64)
                    
                    f1 = compute_f1(K, g_vals)
                    values[0] = c * f1
                    values[1:] = c * g_vals
                    
                    max_val = check_feasibility_fast(domain, values, K)
                    
                    if max_val <= 1.0001:
                        best_score = a_val
                        best_domain = domain.copy()
                        best_values = values.copy()
                    else:
                        c = 0.999 * c
                        values[0] = c * f1
                        values[1:] = c * g_vals
                        max_val = check_feasibility_fast(domain, values, K)
                        if max_val <= 1.0001:
                            a_val = c * a_coeff
                            if a_val > best_score:
                                best_score = a_val
                                best_domain = domain.copy()
                                best_values = values.copy
                
                if time.time() > DEADLINE - 30:
                    break
            
            if time.time() > DEADLINE - 30:
                break
        
        if time.time() > DEADLINE - 30:
            break

    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: use the original seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
