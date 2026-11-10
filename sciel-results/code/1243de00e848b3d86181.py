# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1.0001 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    """
    start_time = time.time()
    deadline = 600  # seconds

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

    max_K = 200000
    mu = mobius_sieve(max_K)
    
    # Get indices where mu != 0
    nonzero_idx = np.where(mu[1:] != 0)[0] + 1
    # We want to consider prefixes of the domain.
    # The domain for a given K is nonzero_idx[nonzero_idx <= K].
    # We can compute the cumulative sum of the objective term for all nonzero indices.
    
    # Objective term for each k: -mu(k) * log(k) / k
    # Note: The evaluator calculates a = -sum f(k) log(k) / k.
    # If f(k) = mu(k), then term is -mu(k) log(k) / k.
    
    k_vals = nonzero_idx.astype(np.float64)
    mu_vals = mu[nonzero_idx].astype(np.float64)
    terms = -mu_vals * np.log(k_vals) / k_vals
    
    # Cumulative sum of terms
    # cum_sum[i] is the sum of terms[0..i]
    # This corresponds to the objective a for the domain consisting of nonzero_idx[0..i]
    # which is all k <= nonzero_idx[i] such that mu(k) != 0.
    # Let K_i = nonzero_idx[i].
    # The domain is {k in [1, K_i] | mu(k) != 0}.
    # This is exactly the domain for max key K_i.
    
    cum_sum = np.cumsum(terms)
    
    # Find the index i that maximizes cum_sum[i]
    # We must ensure that the resulting K is feasible.
    # Feasibility: M(x) <= 1.0001 for x in [1, 10*K].
    # For mu, this is generally true for K up to 200,000.
    # We pick the best K, then verify.
    
    best_idx = np.argmax(cum_sum)
    best_K = nonzero_idx[best_idx]
    best_score_val = cum_sum[best_idx]
    
    # It is possible that a smaller K has a higher score if the sum oscillates.
    # argmax finds the global max in the range [1, 200000].
    
    # We should also check if the max is at a K where the evaluator might be slow.
    # But K <= 200,000 is the limit.
    
    # Let's check a few top candidates to ensure robustness, in case of numerical noise
    # or if the global max is at a K that is "bad" for some reason (unlikely).
    
    # Get top 10 indices by cum_sum value
    top_indices = np.argsort(cum_sum)[::-1][:10]
    
    best_domain = None
    best_values = None
    best_final_score = -np.inf
    
    for idx in top_indices:
        K = nonzero_idx[idx]
        if time.time() - start_time > deadline - 60:
            break
            
        # Construct domain and values for this K
        mask = nonzero_idx <= K
        domain = nonzero_idx[mask]
        values = mu[nonzero_idx].astype(np.float64)[mask]
        
        m = len(domain)
        if m < 1:
            continue
        
        # Quick feasibility check
        # Check x = 1, and a sample of x in [1, 10*K]
        # For mu, M(1) = mu(1) = 1.
        # M(x) for x > 1 should be close to 0.
        # We check a sparse set to avoid O(m * 10K) in our code.
        # The evaluator does the full check.
        
        max_x = 10 * K
        # Check x=1
        M1 = np.sum(values * (1 // domain))
        if M1 > 1.0001:
            continue
            
        # Check some random/sample x
        if max_x > 100:
            check_xs = np.random.randint(1, max_x, size=min(100, max_x))
            # Ensure we check small x
            check_xs = np.concatenate([np.arange(1, 20), check_xs])
            check_xs = np.unique(check_xs)
            
            feasible = True
            # Vectorized check for multiple x is hard without matrix.
            # We do a loop over check_xs, but only 100-120 iterations.
            for x in check_xs:
                Mx = np.sum(values * (x // domain))
                if Mx > 1.0001:
                    feasible = False
                    break
            if not feasible:
                continue
        
        # Calculate exact score for this K
        # a = cum_sum[idx]
        # But wait, the evaluator adjusts f(1).
        # S = sum f(k)/k.
        # f(1) = mu(1) = 1.
        # S = 1/1 + sum_{k>1} mu(k)/k.
        # a_eval = -sum f'(k) log(k)/k.
        # f'(1) = f(1) - S.
        # f'(k) = f(k) for k > 1.
        # a_eval = - (f(1)-S) log(1)/1 - sum_{k>1} f(k) log(k)/k
        # Since log(1) = 0, the term for k=1 vanishes in the objective a.
        # So a_eval = - sum_{k>1} mu(k) log(k)/k.
        # This is exactly what cum_sum[idx] is, because the term for k=1 is 0.
        
        score = cum_sum[idx]
        
        if score > best_final_score:
            best_final_score = score
            best_domain = domain
            best_values = values

    if best_domain is None:
        # Fallback to K=1000
        K = 1000
        mask = nonzero_idx <= K
        domain = nonzero_idx[mask]
        values = mu[nonzero_idx].astype(np.float64)[mask]
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END
