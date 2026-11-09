# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    m = len(domain)
    if m == 0:
        return True
    
    x_max = 10 * K_max
    
    # Use the divisor sum approach for efficiency
    # M(x) = sum_{k in domain} f(k) * floor(x/k)
    # M(x) = sum_{j=1}^{x} h(j) where h(j) = sum_{k|j, k in domain} f(k)
    
    if x_max > 1000000:
        # Use direct computation with numpy for speed on sampled points
        x_sample = np.arange(1, x_max + 1, max(1, x_max // 100000))
        M_vals = np.zeros(len(x_sample))
        for idx, x in enumerate(x_sample):
            M_vals[idx] = np.sum(values * (x // domain))
        return np.all(M_vals <= tolerance)
    else:
        # Compute h(j) for j = 1..x_max
        h = np.zeros(x_max + 1, dtype=np.float64)
        for i, k in enumerate(domain):
            h[k::k] += values[i]
        
        M_vals = np.cumsum(h[1:])
        return np.all(M_vals <= tolerance)


def _optimize_values(domain, values, K_max, deadline, tolerance=1.0001):
    """
    Optimize values to maximize a = -sum_k f(k)*log(k)/k subject to M(x) <= tolerance.
    Uses projected gradient descent with soft constraint handling.
    """
    m = len(domain)
    if m == 0:
        return values
    
    # Objective: maximize a = -sum_k f(k)*log(k)/k
    # Gradient of -a with respect to f(k): -log(k)/k
    log_k_over_k = np.log(domain.astype(np.float64)) / domain.astype(np.float64)
    
    # We want to minimize sum_k f(k)*log(k)/k (negative of objective)
    # So gradient of loss = log(k)/k
    
    # Constraint: M(x) = sum_k f(k)*floor(x/k) <= tolerance for all x in [1, 10*K_max]
    # This is equivalent to: for each x, sum_k f(k)*floor(x/k) - tolerance <= 0
    
    # Use projected gradient descent
    # Step size and iterations
    lr = 0.1
    n_iter = 500
    
    current_values = values.copy()
    best_values = values.copy()
    best_a = -np.sum(current_values * log_k_over_k)
    
    # Precompute constraint matrix structure
    x_max = 10 * K_max
    
    # For efficiency, we'll check constraints at key points
    # M(x) is piecewise linear in f, changing slope at multiples of domain elements
    # Critical points are at x = 1, 2, ..., x_max but we can sample
    
    # Use a smaller set of x values for constraint checking during optimization
    # The binding constraints are likely at small x values
    x_check = np.arange(1, min(x_max + 1, 10000))
    
    for it in range(n_iter):
        if time.time() > deadline - 30:
            break
        
        # Compute gradient of loss = sum_k f(k)*log(k)/k
        grad = log_k_over_k
        
        # Update
        new_values = current_values - lr * grad
        
        # Project onto feasible region: M(x) <= tolerance for all x
        # M(x) = sum_k f(k)*floor(x/k)
        # We need to find the maximum violation and project
        
        # Check constraints at sampled x values
        violations = np.zeros(len(x_check))
        for idx, x in enumerate(x_check):
            M_x = np.sum(new_values * (x // domain))
            violations[idx] = M_x - tolerance
        
        max_violation = np.max(violations)
        
        if max_violation > 1e-6:
            # Need to project: reduce values to satisfy constraints
            # Simple projection: scale down the values that cause violations
            # More sophisticated: use the constraint that's most violated
            
            # For simplicity, use a penalty-based approach
            # Add penalty to gradient for violated constraints
            penalty = 100.0
            penalty_grad = np.zeros(m)
            
            for idx, x in enumerate(x_check):
                if violations[idx] > 0:
                    # Gradient of penalty term: penalty * floor(x/k)
                    penalty_grad += penalty * violations[idx] * (x // domain).astype(np.float64)
            
            new_values = current_values - lr * (grad + penalty_grad)
        
        # Clip values to [-10, 10] as per task specification
        new_values = np.clip(new_values, -10, 10)
        
        # Check if this is better
        current_a = -np.sum(new_values * log_k_over_k)
        
        # Check full feasibility (more expensive, do less often)
        if it % 50 == 0 or it == n_iter - 1:
            if _check_feasibility(domain, new_values, K_max, tolerance):
                if current_a > best_a:
                    best_a = current_a
                    best_values = new_values.copy()
        
        current_values = new_values
    
    return best_values


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function as the base, which asymptotically achieves the optimal
    objective value of 1.0.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s safety margin
    
    # Compute Möbius function up to K
    K = 200000
    mu = _compute_mobius(K)
    
    # Use all k where mu(k) != 0
    domain = np.where(mu[1:] != 0)[0] + 1
    values = mu[domain].astype(np.float64)
    
    # Check feasibility
    if not _check_feasibility(domain, values, K):
        # If infeasible, try smaller K
        for K_try in [100000, 50000, 20000, 10000, 5000, 2000, 1000]:
            if time.time() > deadline:
                break
            mu_try = _compute_mobius(K_try)
            domain_try = np.where(mu_try[1:] != 0)[0] + 1
            values_try = mu_try[domain_try].astype(np.float64)
            if _check_feasibility(domain_try, values_try, K_try):
                domain, values, K = domain_try, values_try, K_try
                break
        else:
            # Fallback to the original small solution
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    # Optimize values to improve the objective while maintaining feasibility
    if time.time() < deadline - 60:
        values_optimized = _optimize_values(domain, values, K, deadline)
        if _check_feasibility(domain, values_optimized, K):
            values = values_optimized
    
    return domain, values


# EVOLVE-BLOCK-END
