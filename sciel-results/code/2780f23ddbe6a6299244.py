# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin

    def compute_mobius(K):
        """Compute Mobius function mu(k) for k <= K using linear sieve."""
        mu = np.zeros(K + 1, dtype=np.int64)
        mu[1] = 1
        primes = []
        is_composite = np.zeros(K + 1, dtype=bool)
        for i in range(2, K + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > K:
                    break
                is_composite[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def check_mobius_feasibility(K, mu):
        """
        Check if the Mobius function (after f(1) adjustment) is feasible.
        Returns (max_M, a_value) if feasible check is computed, or (None, None) if not run.
        """
        f = mu[1:K+1].astype(np.float64)
        
        # S = sum_k f(k)/k
        k_vals = np.arange(1, K+1)
        S = np.sum(f / k_vals)
        
        max_x = 10 * K
        g = np.zeros(max_x + 1, dtype=np.float64)
        
        # Use NumPy slicing for efficiency
        for d in range(1, K + 1):
            val = f[d-1]
            if val != 0:
                g[d:max_x+1:d] += val
        
        M_raw = np.cumsum(g[1:max_x+1])
        x_vals = np.arange(1, max_x + 1)
        M_adj = M_raw - S * x_vals
        max_M = np.max(M_adj)
        
        if max_M <= 1.0001:
            a = -np.sum(f * np.log(k_vals) / k_vals)
            return max_M, a, f, k_vals
        else:
            return max_M, None, f, k_vals

    def greedy_adjust(K, mu):
        """
        Try to make Mobius feasible by zeroing out some values.
        Returns (f, a) if feasible, else (None, None).
        """
        f = mu[1:K+1].astype(np.float64)
        k_vals = np.arange(1, K+1)
        
        # Initial check
        max_M, a, _, _ = check_mobius_feasibility(K, mu)
        if a is not None:
            return f, a
            
        # If infeasible, we need to reduce max_M.
        # max_M is likely caused by large positive contributions in the tail.
        # Zeroing out large positive f(k) for large k might help, but hurts objective.
        # Zeroing out large negative f(k) might help if they cause M to dip too low? 
        # No, constraint is M(x) <= 1.0001.
        # M(x) = sum f(k) floor(x/k).
        # If M(x) is too high, we need to reduce the sum.
        # Since mu(k) is -1, 0, 1, we can only set to 0.
        
        # Strategy: Iterate through k in descending order of magnitude of contribution to max_M?
        # Or just iterate k and if removing it reduces max_M, do it?
        
        # This is expensive to check for each k.
        # Heuristic: The oscillations of sum_{d|n, d<=K} mu(d) are small.
        # If K is large enough, it should be feasible.
        # If K=200000 is infeasible, maybe K=100000 is?
        
        # Let's just return None here and rely on the K-reduction strategy in main.
        return None, None

    def try_lp_solution(K):
        """
        Solve LP for domain [1, K].
        """
        if K > 200:
            return None
        
        max_x = 10 * K
        n_vars = K
        
        c = np.zeros(n_vars)
        for k in range(2, K+1):
            c[k-1] = np.log(k) / k
        
        bounds = [(-10, 10)] * n_vars
        
        # Build A_ub matrix: for each x, row is coefficients of M(x)
        # M(x) = f(1)*(x-1) + sum_{k=2}^K f(k)*(floor(x/k) - x/k)
        A_ub = np.zeros((max_x, n_vars))
        b_ub = np.full(max_x, 1.0001)
        
        for x in range(1, max_x + 1):
            idx = x - 1
            A_ub[idx, 0] = x - 1  # coefficient of f(1)
            for k in range(2, K+1):
                A_ub[idx, k-1] = np.floor(x / k) - x / k
        
        try:
            from scipy.optimize import linprog
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                f_vals = result.x
                k_vals = np.arange(1, K+1)
                
                # Verify feasibility
                S = np.sum(f_vals / k_vals)
                f_adj = f_vals.copy()
                f_adj[0] = f_vals[0] - S
                
                g = np.zeros(max_x + 1, dtype=np.float64)
                for d in range(1, K + 1):
                    val = f_adj[d-1]
                    if val != 0:
                        g[d:max_x+1:d] += val
                M = np.cumsum(g[1:max_x+1])
                max_M = np.max(M)
                
                if max_M <= 1.0001:
                    a = -np.sum(f_vals * np.log(k_vals) / k_vals)
                    return f_vals, a, max_M
                else:
                    return None
            else:
                return None
        except Exception:
            return None

    # Strategy:
    # 1. Compute mu for K=200000 once.
    # 2. Check feasibility for K=200000.
    # 3. If infeasible, try smaller K using the precomputed mu.
    # 4. If all Mobius attempts fail, try LP for small K.
    
    best_f = None
    best_a = -np.inf
    best_k = 0
    
    K_max = 200000
    mu_max = compute_mobius(K_max)
    
    # Try Mobius with decreasing K
    K_candidates = [200000, 100000, 50000, 20000, 10000, 5000, 1000, 500, 100]
    
    for K_try in K_candidates:
        if time.time() > deadline - 60:
            break
            
        # Reuse mu_max if K_try <= K_max
        if K_try <= K_max:
            mu_subset = mu_max[:K_try+1]
        else:
            mu_subset = compute_mobius(K_try)
            
        max_M, a, f, k_vals = check_mobius_feasibility(K_try, mu_subset)
        
        if a is not None:
            # Feasible
            if a > best_a:
                best_f = f
                best_a = a
                best_k = K_try
            # Since we are going from large K to small K, and a generally increases with K for Mobius,
            # the first feasible one is likely the best. But let's be safe.
            # Actually, a = -sum mu(k) log(k)/k. This is approx log(K) - 1.
            # So larger K gives larger a.
            # So we should break on first success?
            # Yes, because we are iterating from largest K downwards.
            break
    
    # If Mobius failed for all, or if we want to check if LP is better (unlikely for small K)
    if best_f is None or best_a < 5.0: # 5.0 is roughly log(150)
        # Try LP for small K
        for K_lp in [100, 200]:
            if time.time() > deadline - 30:
                break
            result = try_lp_solution(K_lp)
            if result is not None:
                f_vals, a, max_M = result
                if a > best_a:
                    best_f = f_vals
                    best_a = a
                    best_k = K_lp
    
    if best_f is not None:
        domain = np.arange(1, best_k+1, dtype=np.int64)
        values = best_f.astype(np.float64)
        return domain, values
    
    # Fallback
    return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
