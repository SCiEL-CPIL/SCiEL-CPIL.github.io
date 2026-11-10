# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of positive integers to reals
    maximizing a = -sum_k f(k)*log(k)/k subject to M(x) <= 1.0001 for all x in [1,10K].
    
    Strategy: Use Möbius function as baseline (gives M(x)=1), then solve a small LP
    on a carefully selected subset of k values to exploit the 0.0001 slack.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    def _elapsed():
        return time.time() - start_time

    def _compute_M(f, k_vals, x_max):
        """Compute M(x) = sum f(k)*floor(x/k) for x in [1, x_max]."""
        M = np.zeros(x_max, dtype=np.float64)
        k_float = k_vals.astype(np.float64)
        for i in range(len(k_vals)):
            M += f[i] * (np.arange(1, x_max + 1) // int(k_vals[i]))
        return M

    # Step 1: Compute Möbius function up to a reasonable K
    K_mobius = 50000
    mu = np.zeros(K_mobius + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K_mobius + 1, dtype=bool)
    primes = []
    for i in range(2, K_mobius + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_mobius:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Step 2: Baseline solution using scaled Möbius
    # f(k) = c * mu(k) gives M(x) = c for all x
    # Max c = 1.0001
    # Metric = -c * sum mu(k)*log(k)/k ≈ c * 1 = c (since sum mu(k)log(k)/k -> -1)
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try different K values for the Möbius baseline
    for K in [10000, 20000, 30000, 40000, 50000]:
        if _elapsed() > deadline - 60:
            break
        
        k_vals = np.arange(1, K + 1, dtype=np.int64)
        c_max = 1.0001
        f_base = c_max * mu[1:K+1].astype(np.float64)
        
        # Check M(x) for this baseline
        x_max = 10 * K
        M_base = np.zeros(x_max, dtype=np.float64)
        x_range = np.arange(1, x_max + 1, dtype=np.int64)
        for i in range(K):
            M_base += f_base[i] * (x_range // int(k_vals[i]))
        
        if np.max(M_base) > 1.0001 + 1e-9:
            continue
        
        # Compute metric
        k_float = k_vals.astype(np.float64)
        log_k = np.log(k_float)
        metric = -np.sum(f_base * log_k / k_float)
        
        if metric > best_score:
            best_score = metric
            best_domain = k_vals.copy()
            best_values = f_base.copy()

    # Step 3: Try to improve by solving a small LP on a selected subset
    # Select k values with high "efficiency" = log(k)/k per unit M(x) cost
    if _elapsed() < deadline - 120:
        K_select = 2000
        k_all = np.arange(1, K_select + 1, dtype=np.int64)
        k_float_all = k_all.astype(np.float64)
        log_k_all = np.log(k_float_all)
        
        # Efficiency: we want to minimize f(k)*log(k)/k (i.e., make f(k) negative)
        # while keeping M(x) <= 1.0001
        # The "cost" of setting f(k) = -1 is the vector floor(x/k) for x in [1,10K]
        # The "benefit" is log(k)/k
        # We want high benefit/cost ratio
        
        # For a rough estimate, the "average cost" of f(k)=-1 is roughly 10K/k * (1/k) = 10K/k^2
        # Benefit is log(k)/k
        # Ratio ≈ log(k)/k / (10K/k^2) = k*log(k)/(10K)
        # This increases with k, so larger k are more efficient
        
        # Select top 200 k values by efficiency, plus include small k for constraint satisfaction
        eff = k_float_all * log_k_all / (10 * K_select)
        top_indices = np.argsort(eff)[-200:]
        small_indices = np.arange(0, 50)  # Include first 50 k values
        selected_indices = np.unique(np.concatenate([top_indices, small_indices]))
        selected_indices = np.sort(selected_indices)
        
        k_sel = k_all[selected_indices]
        m_sel = len(k_sel)
        k_sel_float = k_sel.astype(np.float64)
        log_k_sel = np.log(k_sel_float)
        
        # Build LP: maximize -sum f_sel(k) * log(k) / k
        # subject to: sum f_sel(k) * floor(x/k) <= 1.0001 for x in [1, 10*K_select]
        #             sum f_sel(k) / k = 0
        #             -10 <= f_sel(k) <= 10
        
        x_max_sel = 10 * K_select
        x_range_sel = np.arange(1, x_max_sel + 1, dtype=np.int64)
        
        # Constraint matrix
        A_ub = np.zeros((x_max_sel, m_sel), dtype=np.float64)
        for j in range(m_sel):
            A_ub[:, j] = x_range_sel // int(k_sel[j])
        
        b_ub = np.full(x_max_sel, 1.0001, dtype=np.float64)
        
        # Equality constraint: sum f(k)/k = 0
        A_eq = (1.0 / k_sel_float).reshape(1, -1)
        b_eq = np.array([0.0])
        
        # Objective: minimize sum f(k) * log(k) / k (equivalent to maximizing negative)
        c_obj = log_k_sel / k_sel_float
        
        bounds = [(-10.0, 10.0)] * m_sel
        
        try:
            result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method='highs',
                           options={'presolve': True, 'time_limit': 80})
            
            if result.success:
                f_sel = result.x.copy()
                # Extend to full domain: f(k) = 0 for k not in selected
                f_full = np.zeros(K_select, dtype=np.float64)
                f_full[selected_indices] = f_sel
                
                # Verify M(x)
                M_check = np.zeros(x_max_sel, dtype=np.float64)
                for i in range(K_select):
                    if f_full[i] != 0:
                        M_check += f_full[i] * (x_range_sel // int(k_all[i]))
                
                if np.max(M_check) <= 1.0001 + 1e-9:
                    metric_sel = -np.sum(f_full * log_k_all / k_float_all)
                    if metric_sel > best_score:
                        best_score = metric_sel
                        best_domain = k_all.copy()
                        best_values = f_full.copy()
        except Exception:
            pass

    # Step 4: Try a different structural approach - use only prime-related k values
    if _elapsed() < deadline - 60:
        # Use k values that are prime powers or have special structure
        K_pp = 5000
        # Generate prime powers up to K_pp
        prime_powers = set()
        for p in primes:
            if p > K_pp:
                break
            pp = p
            while pp <= K_pp:
                prime_powers.add(pp)
                pp *= p
        
        # Add 1 and some small composites
        for i in range(1, 101):
            prime_powers.add(i)
        
        k_pp = np.array(sorted(prime_powers), dtype=np.int64)
        k_pp = k_pp[k_pp <= K_pp]
        m_pp = len(k_pp)
        
        if m_pp < 10:
            k_pp = np.arange(1, K_pp + 1, dtype=np.int64)
            m_pp = K_pp
        
        k_pp_float = k_pp.astype(np.float64)
        log_k_pp = np.log(k_pp_float)
        
        x_max_pp = 10 * int(np.max(k_pp))
        x_range_pp = np.arange(1, x_max_pp + 1, dtype=np.int64)
        
        A_ub_pp = np.zeros((x_max_pp, m_pp), dtype=np.float64)
        for j in range(m_pp):
            A_ub_pp[:, j] = x_range_pp // int(k_pp[j])
        
        b_ub_pp = np.full(x_max_pp, 1.0001, dtype=np.float64)
        A_eq_pp = (1.0 / k_pp_float).reshape(1, -1)
        b_eq_pp = np.array([0.0])
        c_obj_pp = log_k_pp / k_pp_float
        bounds_pp = [(-10.0, 10.0)] * m_pp
        
        try:
            result_pp = linprog(c_obj_pp, A_ub=A_ub_pp, b_ub=b_ub_pp, A_eq=A_eq_pp, b_eq=b_eq_pp,
                              bounds=bounds_pp, method='highs',
                              options={'presolve': True, 'time_limit': 60})
            
            if result_pp.success:
                f_pp = result_pp.x.copy()
                M_check_pp = np.zeros(x_max_pp, dtype=np.float64)
                for i in range(m_pp):
                    M_check_pp += f_pp[i] * (x_range_pp // int(k_pp[i]))
                
                if np.max(M_check_pp) <= 1.0001 + 1e-9:
                    metric_pp = -np.sum(f_pp * log_k_pp / k_pp_float)
                    if metric_pp > best_score:
                        best_score = metric_pp
                        best_domain = k_pp.copy()
                        best_values = f_pp.copy()
        except Exception:
            pass

    # Fallback: minimal feasible solution
    if best_domain is None:
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([0.0, -1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END
