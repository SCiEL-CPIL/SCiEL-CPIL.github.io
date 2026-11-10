import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0
    
    best_result = None
    best_a = -np.inf
    
    # We will try different K values
    # The optimal K might be small because the reward decays.
    # But P needs large k to be "cheap". So K should be large enough to provide cheap P.
    
    for K in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000]:
        if time.time() - start_time > deadline - 60:
            break
            
        k_vals = np.arange(2, K + 1)
        log_k_over_k = np.log(k_vals) / k_vals
        inv_k = 1.0 / k_vals
        
        # We want to select N (negative f) and P (positive f)
        # Constraint: 10 * (sum(N inv_k) - sum(P inv_k)) >= 1.0001
        # Objective: 10 * (sum(N log_k_over_k) - sum(P log_k_over_k))
        
        # Greedy approach:
        # Sort by log_k_over_k descending for N candidates
        # Sort by log_k_over_k ascending for P candidates
        
        n_indices = np.argsort(-log_k_over_k)
        p_indices = np.argsort(log_k_over_k)
        
        # We will build N and P
        # Start with empty
        sum_N = 0.0
        sum_P = 0.0
        obj_N = 0.0
        obj_P = 0.0
        
        N_set = set()
        P_set = set()
        
        # We need sum_N - sum_P >= 0.10001
        # Let's try to add elements to N first, then compensate with P
        
        # Actually, it's better to view it as:
        # Maximize Obj = 10 * (Sum_N_L - Sum_P_L)
        # Subject to 10 * (Sum_N_I - Sum_P_I) >= 1.0001
        
        # This is a fractional knapsack variant if we could split, but we are binary.
        # Let's just try a few simple configurations.
        
        # Config 1: All N from smallest k, P from largest k
        # Try different sizes for N
        
        for size_N in range(1, min(50, K)): # N is likely small
            # Take top size_N from N_candidates
            n_idx = n_indices[:size_N]
            s_N_I = np.sum(inv_k[n_idx])
            s_N_L = np.sum(log_k_over_k[n_idx])
            
            # Required P sum: s_P_I >= s_N_I - 0.10001
            # We want to minimize s_P_L subject to s_P_I >= target
            # This means we should pick P with smallest L/I ratio?
            # No, we pick P to satisfy sum I constraint with min sum L.
            # This is another knapsack.
            # Approximation: Pick P with smallest log_k_over_k.
            
            target_I = s_N_I - 0.10001
            if target_I <= 0:
                # No P needed
                s_P_I = 0.0
                s_P_L = 0.0
            else:
                # Sort P candidates by log_k_over_k ascending
                # We need sum I >= target_I
                # Greedy: pick smallest L/I? Or just smallest L?
                # Let's pick smallest L first.
                p_idx = p_indices
                s_P_I_curr = 0.0
                s_P_L_curr = 0.0
                p_selected = []
                for idx in p_idx:
                    if idx in N_set: # Avoid overlap, though N and P should be disjoint
                        continue
                    s_P_I_curr += inv_k[idx]
                    s_P_L_curr += log_k_over_k[idx]
                    p_selected.append(idx)
                    if s_P_I_curr >= target_I:
                        break
                s_P_I = s_P_I_curr
                s_P_L = s_P_L_curr
                P_set = set(p_selected)
            
            N_set = set(n_idx)
            
            # Check if valid (sum_N - sum_P >= 0.10001)
            if s_N_I - s_P_I < 0.10001:
                continue
                
            a = 10 * (s_N_L - s_P_L)
            
            if a > best_a:
                best_a = a
                # Construct f
                f = np.zeros(K + 1)
                f[N_set] = -10.0
                f[P_set] = 10.0
                # f(1) is determined by normalization
                # We return domain and values for k>=1
                # The function returns (domain, values)
                # domain should be [1, 2, ..., K]
                # values should be f(1), f(2), ..., f(K)
                
                # Calculate f(1)
                # S = sum f(k)/k
                # f(1) = S
                # Wait, f(1) is replaced by f(1) - S to make sum 0.
                # If we set f(1)=0 initially, then f(1)_adj = -S.
                # S = sum_{k=2}^K f(k)/k.
                # f(1)_adj = - (sum_{k=2}^K f(k)/k) = sum_{k=2}^K -f(k)/k.
                # Note: sum_{k=2}^K f(k)/k = 10 * (s_P_I - s_N_I).
                # So f(1)_adj = -10 * (s_P_I - s_N_I) = 10 * (s_N_I - s_P_I).
                
                # We need to check if f(1)_adj is within limits?
                # The problem says "values are clipped to [-10, 10]".
                # This applies to the input values.
                # If we return f(1) as part of values, it will be clipped.
                # Then S is calculated on clipped values.
                # Then f(1) is adjusted.
                
                # So we should return f(k) for k>=2 as -10 or 10.
                # And f(1) can be anything, say 0, because it gets clipped and then adjusted.
                # But wait, if f(1) is clipped, it becomes 0 if we return 0.
                # Then S = sum_{k=2} f(k)/k.
                # f(1)_final = 0 - S = -S.
                
                # So we just need to return f(k) for k>=2.
                
                domain = np.arange(1, K + 1)
                values = np.zeros(K + 1)
                values[N_set] = -10.0
                values[P_set] = 10.0
                # values[0] is index 1? No, domain is 1-indexed.
                # values array corresponds to domain.
                # domain[0] = 1. values[0] is f(1).
                # We set values[1:] for k=2..K.
                
                best_result = (domain, values)
                
        if time.time() - start_time > deadline - 30:
            break

    if best_result is None:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_result
