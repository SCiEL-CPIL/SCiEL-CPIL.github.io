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
        For x <= K: M(x) = sum_{k<=x} mu(k) floor(x/k) = 1 exactly (if full Mobius).
        For x > K: need to verify.
        We check x in [K+1, 10K] using vectorized computation in batches.
        """
        max_x = 10 * K
        m = len(domain)
        
        # For x <= K, with full Mobius, M(x) = 1 exactly.
        # Only need to check x in [K+1, 10K]
        check_start = K + 1
        check_end = max_x
        
        if check_start > check_end:
            return True
        
        n_checks = check_end - check_start + 1
        
        # Process in batches to manage memory
        batch_size = 5000
        for batch_start in range(check_start, check_end + 1, batch_size):
            if time.time() - start_time > deadline - 90:
                return False
            
            batch_end = min(batch_start + batch_size - 1, check_end)
            xs = np.arange(batch_start, batch_end + 1, dtype=np.int64)
            n_x = len(xs)
            
            # Compute floor(x/k) for all x in batch and all k in domain
            # Shape: (n_x, m)
            # To save memory, process domain in chunks if needed
            chunk_size = min(m, 5000)
            M_batch = np.zeros(n_x, dtype=np.float64)
            
            for d_start in range(0, m, chunk_size):
                d_end = min(d_start + chunk_size, m)
                d_chunk = domain[d_start:d_end]
                v_chunk = values[d_start:d_end]
                
                # (n_x, 1) // (1, chunk) -> (n_x, chunk)
                floor_mat = xs[:, None] // d_chunk[None, :]
                M_batch += np.sum(v_chunk[None, :] * floor_mat, axis=1)
            
            if np.any(M_batch > 1.0001):
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

        # Estimate evaluator cost: m * 10 * K
        est_ops = m * 10 * K
        if est_ops > 5e9:  # too slow for evaluator
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
