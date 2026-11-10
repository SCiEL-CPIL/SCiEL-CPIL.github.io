import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 590.0  # safety margin before 600s kill

    # Use the largest feasible K to maximize the metric a = -sum f(k) log(k) / k
    K = 200000

    # Linear sieve for Möbius function μ(n)
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    is_composite = np.zeros(K + 1, dtype=bool)
    primes = []

    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > K:
                break
            is_composite[ip] = True
            if i % p == 0:
                mu[ip] = 0
                break
            else:
                mu[ip] = -mu[i]

        # Time guard: sieve for 200k is fast (<1s), but keep guard just in case
        if time.time() > deadline - 30:
            # If we somehow run out of time, just use what we have
            # Truncate domain to what's been computed
            K = i - 1
            break

    # Build domain and values
    # We can restrict to indices where mu != 0 to reduce m, but the constraint
    # M(x) = sum_k f(k) * floor(x/k) must hold for all x in [1, 10K].
    # Using all k=1..K with f(k) = mu(k) is the natural choice.
    # However, we can drop indices where mu(k)=0 since they contribute 0.
    # This reduces m significantly (about 30% are non-zero).
    
    # For correctness of M(x), we only need non-zero f(k).
    # Let's use only non-zero mu entries to keep m smaller (faster evaluation).
    nonzero_mask = mu[1:] != 0
    domain = np.arange(1, K + 1, dtype=np.int64)[nonzero_mask]
    values = mu[1:][nonzero_mask].astype(np.float64)
    
    # Set f(1) = 0 initially (evaluator will adjust it)
    if domain[0] == 1:
        values[0] = 0.0

    return domain, values
