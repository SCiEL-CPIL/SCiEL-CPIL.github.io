# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def _mobius_upto(n):
    mu = np.ones(n + 1, dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            step = i
            for j in range(i * i, n + 1, step):
                is_prime[j] = False
            mu[i * i:n + 1:step] *= -1
            for j in range(i * i * i, n + 1, i * i):
                mu[j] = 0
    return mu


def _baseline():
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


def _feasible_fast(domain, values, K):
    """Fast feasibility check using divisor-sum prefix sums."""
    f = values
    ks = domain.astype(np.int64)
    
    # Build d[j] = sum_{k|j} f(k) for j in [1, K]
    d = np.zeros(K + 1, dtype=np.float64)
    for idx, k in enumerate(ks):
        kk = int(k)
        if kk <= K:
            d[kk:K+1:kk] += f[idx]
    
    prefix_d = np.cumsum(d[1:K+1])
    if prefix_d.max() > 1.0001:
        return False
    
    # Check x in [K+1, 10K]
    if 10*K > K:
        inc = np.zeros(10*K + 1, dtype=np.float64)
        for idx, k in enumerate(ks):
            kk = int(k)
            if kk <= 10*K:
                inc[kk:10*K+1:kk] += f[idx]
        
        M_K = prefix_d[-1] if K >= 1 else 0.0
        cum_inc = np.cumsum(inc[K+1:10*K+1])
        M_vals = M_K + cum_inc
        if M_vals.max() > 1.0001:
            return False
    
    return True


def _find_violated(z, K, tol=1e-6):
    """Find x values where M(x) > 1+tol. Returns list of x values."""
    ks = np.arange(1, K + 1, dtype=np.int64)
    
    d = np.zeros(K + 1, dtype=np.float64)
    for k_val in range(1, K + 1):
        if abs(z[k_val - 1]) > 1e-15:
            d[k_val:K+1:k_val] += z[k_val - 1]
    
    prefix_d = np.cumsum(d[1:K+1])
    violated_small = np.where(prefix_d > 1.0 + tol)[0]
    violated_xs = [int(idx + 1) for idx in violated_small]
    
    if 10*K > K:
        inc = np.zeros(10*K + 1, dtype=np.float64)
        for k_val in range(1, K + 1):
            if abs(z[k_val - 1]) > 1e-15:
                inc[k_val:10*K+1:k_val] += z[k_val - 1]
        
        M_K = prefix_d[-1] if K >= 1 else 0.0
        cum_inc = np.cumsum(inc[K+1:10*K+1])
        M_large = M_K + cum_inc
        violated_large = np.where(M_large > 1.0 + tol)[0]
        for idx in violated_large:
            violated_xs.append(int(K + 1 + idx))
    
    return violated_xs


def _solve_lp_for_K(K, deadline, max_constraints=5000):
    """Solve LP for given K using cutting-plane method."""
    t0 = time.time()
    ks = np.arange(1, K + 1, dtype=np.int64)
    kf = ks.astype(np.float64)
    c = np.log(kf) / kf
    
    # Start with a reasonable set of constraints
    # Include x=1, and some spread out points
    init_xs = set()
    init_xs.add(1)
    # Add some evenly spaced points
    n_init = min(200, K)
    for i in range(n_init):
        x = int((i + 1) * K / n_init)
        x = max(1, min(x, 10*K))
        init_xs.add(x)
    
    x_set = sorted(init_xs)
    z = None
    max_iter = 50
    tol = 1e-6
    
    for iteration in range(max_iter):
        if time.time() > deadline - 15:
            break
        
        n_cons = len(x_set)
        if n_cons > max_constraints:
            # Subsample constraints if too many
            step = max(1, n_cons // max_constraints)
            x_set = x_set[::step]
            n_cons = len(x_set)
        
        # Build constraint matrix efficiently
        A = np.zeros((n_cons, K), dtype=np.float64)
        for i, x in enumerate(x_set):
            A[i, :] = x // ks
        
        # Solve LP
        res = linprog(-c, A_ub=A, b_ub=np.ones(n_cons),
                      bounds=[(0.0, None)] * K, method="highs",
                      options={"presolve": True, "time_limit": 20.0})
        
        if not res.success:
            break
        
        z = res.x
        obj = float(np.dot(z, c))
        
        # Check for violated constraints
        violated = _find_violated(z, K, tol)
        
        if len(violated) == 0:
            break
        
        # Add most violated constraints
        # Sort by violation amount (approximate by value)
        existing = set(x_set)
        new_xs = [x for x in violated if x not in existing]
        
        if new_xs:
            # Add up to 100 new constraints per iteration
            if len(new_xs) > 100:
                # Pick diverse subset
                step = max(1, len(new_xs) // 100)
                new_xs = new_xs[::step][:100]
            
            x_set = sorted(existing | set(new_xs))
        
        # Limit total constraints
        if len(x_set) > max_constraints:
            step = max(1, len(x_set) // max_constraints)
            x_set = x_set[::step]
            x_set = sorted(x_set)
    
    if z is None:
        return None, 0.0
    
    return z, float(np.dot(z, c))


def prime_number_theorem_partial_function():
    """
    Construct a partial function f : subset of positive integers -> R that maximises
    a = -sum_k f(k)*log(k)/k under the step-function constraint
        M(x) = sum_k f(k)*floor(x/k) <= 1   for every integer x in [1, 10K],
    where K = max key in the domain.
    
    Uses iterative constraint generation (cutting-plane) to solve the LP efficiently.
    """
    t0 = time.time()
    DEADLINE = 590.0  # wall-clock safety margin before the 600 s kill
    
    best = _baseline()
    best_a = 1.0
    
    # Try LP for increasing K values with adaptive time allocation
    K_candidates = [500, 1000, 2000, 3000, 5000, 8000, 12000, 16000, 20000, 30000]
    
    for K in K_candidates:
        if time.time() > DEADLINE - 40:
            break
        
        try:
            z, a = _solve_lp_for_K(K, DEADLINE)
            
            if z is not None:
                # Final verification with actual returned f
                ks = np.arange(1, K + 1, dtype=np.int64)
                values = z.copy()
                values[0] = -values[1:].sum() / K  # enforce sum f(k)/k = 0
                
                if _feasible_fast(ks, values, K):
                    # Recompute a with the actual values
                    a_actual = float(-np.dot(values, np.log(ks.astype(np.float64)) / ks.astype(np.float64)))
                    if a_actual > best_a + 1e-9:
                        best = (ks, values)
                        best_a = a_actual
                        print(f"K={K}, a={a_actual:.6f}, time={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"K={K} failed: {e}")
            continue
    
    # If LP didn't improve significantly, try mu-based approach for larger K
    if best_a <= 1.02 + 1e-9 and time.time() < DEADLINE - 25:
        for K in (1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000):
            if time.time() > DEADLINE - 10:
                break
            try:
                mu = _mobius_upto(K)
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = mu.astype(np.float64).copy()
                values[0] = -values[1:].sum() / K
                
                if _feasible_fast(domain, values, K):
                    a = float(-np.dot(values, np.log(domain.astype(np.float64)) / domain.astype(np.float64)))
                    if a > best_a + 1e-9:
                        best = (domain, values)
                        best_a = a
                        print(f"mu K={K}, a={a:.6f}, time={time.time()-t0:.1f}s")
            except Exception:
                continue
    
    return best


# EVOLVE-BLOCK-END
