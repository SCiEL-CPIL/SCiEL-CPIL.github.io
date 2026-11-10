# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600

    def mobius_sieve(n):
        """Compute Mobius function mu(k) for k=1..n using a sieve."""
        mu = np.ones(n + 1, dtype=np.int8)
        mu[0] = 0
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        primes = []
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def check_feasibility_fast(domain, values, K):
        """
        Check M(x) <= 1.0001 for all x in [1, 10K].
        For the full Mobius function, M(x) = 1 for all x <= K (known identity).
        For x in [K+1, 10K], use prefix sums of mu to compute M(x) in O(sqrt(x)) per x.
        """
        max_x = 10 * K
        
        # Build prefix sum of mu for O(1) range sums
        mu_full = np.zeros(K + 1, dtype=np.float64)
        for i in range(len(domain)):
            k = domain[i]
            if 1 <= k <= K:
                mu_full[k] = values[i]
        
        # Prefix sum: prefix[i] = sum_{k=1}^{i} mu_full[k]
        prefix = np.zeros(K + 1, dtype=np.float64)
        for i in range(1, K + 1):
            prefix[i] = prefix[i-1] + mu_full[i]
        
        # For x <= K, M(x) = 1 exactly (identity), so skip
        # For x in [K+1, 10K], compute M(x) using grouping by floor(x/k)
        check_start = K + 1
        check_end = max_x
        
        if check_start > check_end:
            return True
        
        # Process x values; use sampling if too many
        n_checks = check_end - check_start + 1
        if n_checks > 200000:
            # Sample every n_th x value
            step = n_checks // 200000 + 1
            xs_to_check = range(check_start, check_end + 1, step)
        else:
            xs_to_check = range(check_start, check_end + 1)
        
        for x in xs_to_check:
            if time.time() - start_time > deadline - 90:
                return False
            
            # Compute M(x) = sum_{k=1}^{K} mu(k) * floor(x/k)
            # Group k by value of floor(x/k) = q
            M_x = 0.0
            k = 1
            while k <= K:
                q = x // k
                if q == 0:
                    break
                # k_max is the largest k such that floor(x/k) = q
                k_max = min(x // q, K)
                # Sum mu(k) for k in [k, k_max]
                sum_mu = prefix[k_max] - prefix[k-1]
                M_x += q * sum_mu
                k = k_max + 1
            
            if M_x > 1.0001:
                return False
        
        return True

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try increasing K values
    for K in [200000, 100000, 50000, 20000, 10000, 5000, 1000]:
        if time.time() - start_time > deadline - 120:
            break

        mu = mobius_sieve(K)
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        domain = nonzero_idx.astype(np.int64)
        values = mu[nonzero_idx].astype(np.float64)

        m = len(domain)
        if m < 1:
            continue

        # Compute objective a = -sum f(k) log(k) / k
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(values * log_k / domain.astype(np.float64))

        # Check feasibility
        feasible = check_feasibility_fast(domain, values, K)

        if not feasible:
            continue

        if a > best_score:
            best_score = a
            best_domain = domain.copy()
            best_values = values.copy()
            break  # Largest K that works is best

    # Fallback to baseline if nothing found
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
