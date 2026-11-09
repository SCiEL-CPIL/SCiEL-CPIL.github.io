# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_m_constraint(values_adj, K, x_max, time_budget_remaining):
    """
    Fast verification that M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, x_max].
    Uses vectorized difference-array construction.
    Returns (is_valid, max_M).
    """
    # Build diff array using vectorized operations
    # For each k with f_k != 0, we add f_k at positions k, 2k, ..., floor(x_max/k)*k
    # and subtract f_k at positions 2k, 3k, ..., (floor(x_max/k)+1)*k (if <= x_max+1)
    
    # Strategy: use the fact that for each k, the positions are arithmetic sequences.
    # We can batch-process by grouping k values that have the same floor(x_max/k) behavior,
    # but a simpler approach: iterate over k but vectorize the inner operations.
    
    # Actually, let's use a smarter approach:
    # M(x) = sum_{k=1}^{K} f(k) * floor(x/k)
    # = sum_{k=1}^{K} f(k) * sum_{j=1}^{floor(x/k)} 1
    # = sum_{j=1}^{x} sum_{k=1}^{min(K, x//j)} f(k)  [swapping sums, but this is O(x*K) worst case]
    
    # Better: use the difference array approach but vectorize it.
    # For each k, the contribution to M is:
    #   f(k) for x in [k, 2k-1], 2*f(k) for x in [2k, 3k-1], etc.
    # So M(x) = sum_{k=1}^{min(K,x)} f(k) * floor(x/k)
    
    # Difference array: M(x) - M(x-1) = sum_{k: k|x} f(k)  [i.e., sum of f(k) over divisors k of x]
    # Wait, that's not right. Let me reconsider.
    
    # M(x) = sum_{k=1}^{K} f(k) floor(x/k)
    # M(x) - M(x-1) = sum_{k=1}^{K} f(k) (floor(x/k) - floor((x-1)/k))
    # floor(x/k) - floor((x-1)/k) = 1 if k divides x, else 0
    # So M(x) - M(x-1) = sum_{k|x, k<=K} f(k)
    
    # This is the Dirichlet convolution of f with the indicator of 1!
    # M(x) - M(x-1) = (f * 1)(x) where * is Dirichlet convolution, restricted to k <= K.
    
    # So we can compute d(x) = sum_{k|x, k<=K} f(k) for x = 1, ..., x_max
    # using a sieve-like approach: for each k, add f(k) to all multiples of k.
    
    # This is O(x_max * log(x_max)) with a simple sieve, or we can vectorize it.
    
    d = np.zeros(x_max + 1, dtype=np.float64)
    
    # Vectorized: for each k, d[k::k] += f(k)
    # We can batch this by processing all k at once using numpy's fancy indexing,
    # but the total number of operations is sum_{k=1}^{K} floor(x_max/k) ~ x_max * log(K)
    
    # Let's do it in chunks to manage memory
    chunk_size = 100000
    for k_start in range(1, K + 1, chunk_size):
        k_end = min(k_start + chunk_size, K + 1)
        for k in range(k_start, k_end):
            f_k = values_adj[k - 1]
            if f_k != 0.0:
                d[k::k] += f_k
            if (time.time() > time_budget_remaining):
                return False, np.inf
    
    M = np.cumsum(d)
    max_M = np.max(M[1:x_max + 1])
    return max_M <= 1.0001, max_M


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy:
      1. Start with the Mobius function mu(k) for k=1..K as the initial solution.
      2. Verify the M(x) constraint using a fast vectorized difference-array method.
      3. If feasible, attempt to improve the objective via LP using scipy.optimize.linprog
         with a cutting-plane approach (start with a subset of constraints, add violated ones).
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 30.0

    K = 200000
    x_max = 10 * K

    # Compute Mobius function mu(k) for k = 1..K using a sieve
    mu = np.ones(K + 1, dtype=np.int64)
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
    mu[1] = 1

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    values_clipped = np.clip(values, -10.0, 10.0)

    S = np.sum(values_clipped / domain)
    values_adj = values_clipped.copy()
    values_adj[0] = values_adj[0] - S

    # Verify M(x) constraint
    elapsed = time.time() - start_time
    if elapsed > time_limit - safety_margin:
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    is_valid, max_M = _verify_m_constraint(values_adj, K, x_max, time_limit - safety_margin)
    
    if not is_valid:
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # Compute baseline metric
    log_k = np.log(domain)
    a_baseline = -np.sum(values_adj * log_k / domain)

    # Try to improve via LP with cutting plane
    elapsed = time.time() - start_time
    remaining = time_limit - safety_margin - elapsed
    
    if remaining > 60:
        try:
            values_lp = _solve_lp_improvement(values_adj, domain, K, x_max, remaining)
            if values_lp is not None:
                # Verify the LP solution
                is_valid_lp, _ = _verify_m_constraint(values_lp, K, x_max, time_limit - safety_margin)
                if is_valid_lp:
                    a_lp = -np.sum(values_lp * log_k / domain)
                    if a_lp > a_baseline:
                        values_adj = values_lp
                        a_baseline = a_lp
        except Exception:
            pass

    # Final check
    elapsed = time.time() - start_time
    if elapsed > time_limit - safety_margin:
        return domain, values

    is_valid_final, _ = _verify_m_constraint(values_adj, K, x_max, time_limit - safety_margin)
    if not is_valid_final:
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    return domain, values


def _solve_lp_improvement(values_init, domain, K, x_max, time_budget):
    """
    Try to improve the objective using scipy.optimize.linprog with a cutting-plane approach.
    
    The problem is:
    Maximize: -sum_k f(k) log(k)/k
    Subject to:
      -sum_k f(k)/k = 0  (the S adjustment makes this hold)
      M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, x_max]
      -10 <= f(k) <= 10 for all k (clipping)
    
    We use a cutting-plane method: start with a subset of x constraints, solve, 
    check all x, add violated constraints, repeat.
    """
    from scipy.optimize import linprog
    
    # Objective: minimize sum_k f(k) log(k)/k  (negative of what we want to maximize)
    c = (np.log(domain) / domain).astype(np.float64)
    
    # Equality constraint: sum_k f(k)/k = 0
    A_eq = (1.0 / domain).reshape(1, -1)
    b_eq = np.array([0.0])
    
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10.0, 10.0)] * K
    
    # Start with a coarse set of x constraints
    # x values: 1, 2, 4, 8, ..., up to x_max, plus some random samples
    x_coarse = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 
                         8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
                         2097152, 4194304, 8388608, 16777216, 33554432, 67108864,
                         134217728, 268435456, 536870912, 1073741824, 2147483648])
    x_coarse = x_coarse[x_coarse <= x_max]
    
    # Add some random x values
    np.random.seed(42)
    x_random = np.random.randint(1, x_max + 1, size=min(100, x_max))
    x_set = np.unique(np.concatenate([x_coarse, x_random]))
    
    # Build A_ub for these x values
    # M(x) = sum_k f(k) floor(x/k) <= 1.0001
    # For each x in x_set, the row is [floor(x/1), floor(x/2), ..., floor(x/K)]
    
    def build_A_ub(x_values):
        """Build constraint matrix for given x values."""
        n_constraints = len(x_values)
        A = np.zeros((n_constraints, K), dtype=np.float64)
        for i, x in enumerate(x_values):
            # floor(x/k) for k = 1, ..., K
            # Vectorized: for k > x, floor(x/k) = 0
            k_max = min(x, K)
            A[i, :k_max] = np.floor(x / np.arange(1, k_max + 1, dtype=np.float64))
        return A
    
    A_ub = build_A_ub(x_set)
    b_ub = np.full(len(x_set), 1.0001)
    
    deadline = time.time() + time_budget - 5.0  # 5s safety margin
    
    for iteration in range(20):
        if time.time() > deadline:
            break
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if not result.success:
            break
        
        f_opt = result.x
        
        # Check all x in [1, x_max] for violations
        # Sample x values to check
        n_check = min(10000, x_max)
        x_check = np.random.choice(x_max, size=n_check, replace=False)
        x_check = np.sort(x_check)
        
        # Compute M(x) for these x values
        # M(x) = sum_k f(k) floor(x/k)
        # Vectorized approach:
        violations = []
        for x in x_check:
            k_max = min(x, K)
            M_x = np.dot(f_opt[:k_max], np.floor(x / np.arange(1, k_max + 1, dtype=np.float64)))
            if M_x > 1.0001:
                violations.append(x)
        
        if len(violations) == 0:
            # No violations found in sample; do a more thorough check
            # Check multiples of small numbers and some structured set
            x_thorough = np.unique(np.concatenate([
                np.arange(1, min(1000, x_max + 1)),
                np.arange(0, x_max + 1, max(1, x_max // 1000)),
                violations
            ]))
            x_thorough = x_thorough[x_thorough <= x_max]
            
            found_violation = False
            for x in x_thorough:
                k_max = min(x, K)
                M_x = np.dot(f_opt[:k_max], np.floor(x / np.arange(1, k_max + 1, dtype=np.float64)))
                if M_x > 1.0001:
                    violations.append(x)
                    found_violation = True
            
            if not found_violation:
                return f_opt
        else:
            # Add violated constraints
            new_x = np.array(violations[:50])  # Add up to 50 new constraints
            new_A = build_A_ub(new_x)
            A_ub = np.vstack([A_ub, new_A])
            b_ub = np.concatenate([b_ub, np.full(len(new_x), 1.0001)])
    
    return None


# EVOLVE-BLOCK-END
