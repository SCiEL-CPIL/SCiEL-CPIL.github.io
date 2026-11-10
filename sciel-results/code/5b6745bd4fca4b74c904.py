# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 630.0  # seconds total
    safety_margin = 30.0  # return by 600s

    def elapsed_ok():
        return (time.time() - start_time) < (deadline - safety_margin)

    def compute_mobius(K):
        """Compute the Mobius function mu(n) for n = 1..K using a sieve."""
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        primes = []
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False

        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                for p in primes:
                    if i * p > K:
                        break
                    is_prime[i * p] = False
                    if i % p == 0:
                        mu[i * p] = 0
                        break
                    else:
                        mu[i * p] = -mu[i]

        return mu[1:]  # mu[1], mu[2], ..., mu[K]

    def verify_M(f_values, K):
        """
        Verify that M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
        Returns True if feasible, False otherwise.
        """
        max_x = 10 * K
        # M(x) = sum_{k=1}^{K} f(k) * floor(x/k)
        # We can compute M(x) for all x using a difference array.
        # For a given k, f(k) * floor(x/k) is a step function that increases by f(k)
        # at each multiple of k.
        
        diff = np.zeros(max_x + 1, dtype=np.float64)
        
        # Optimization: skip k where f(k) is 0
        # Also, for very large K, this loop might be slow. 
        # However, K=200000 means max_x = 2,000,000.
        # The total number of operations is sum_{k=1}^K (max_x/k) ~ max_x * H_K ~ 2e6 * 12 ~ 24e6.
        # This is fast in NumPy if vectorized, but we are doing it in a loop over k.
        # Let's try to vectorize the addition to diff.
        
        # Actually, a pure Python loop over K=200,000 where each step does a NumPy slice operation
        # might be slow due to overhead. 
        # Let's estimate: 200,000 iterations. Each involves np.arange and an addition.
        # This might take ~1-2 seconds. Acceptable.
        
        for k_idx in range(K):
            k = k_idx + 1
            fk = f_values[k_idx]
            if fk == 0:
                continue
            # Add fk at positions k, 2k, 3k, ... up to max_x
            # diff[multiples] += fk
            multiples = np.arange(k, max_x + 1, k)
            diff[multiples] += fk
            
        M_vals = np.cumsum(diff)
        # M_vals[x] = M(x) for x = 1..max_x
        # Note: M(0) is not defined in the problem, but cumsum starts at index 0.
        # The problem asks for x in [1, 10K].
        if np.any(M_vals[1:] > 1.0001):
            return False
        return True

    def compute_objective(f_values, domain):
        """Compute a = -sum_k f(k) * log(k) / k."""
        logs = np.log(domain.astype(np.float64))
        return -np.sum(f_values * logs / domain.astype(np.float64))

    # Strategy:
    # 1. Compute mu(k) for k=1..200000.
    # 2. Set f(1) = 0. (The evaluator adjusts f(1) to satisfy sum f(k)/k = 0, 
    #    but if we set it to 0 initially, the adjustment S = sum_{k>1} f(k)/k.
    #    The objective is calculated on the *clipped* values *before* the f(1) adjustment?
    #    "values are clipped... Then f(1) is replaced... metric is a = -sum f(k) log(k)/k"
    #    Wait, the metric uses the final f(k) values?
    #    "metric is a = -sum_k f(k) * log(k) / k"
    #    Usually, this implies the final state. Let's assume the metric uses the final values.
    #    If we set f(1)=0 in our output, the evaluator computes S = sum_{k>1} mu(k)/k.
    #    Then it sets f(1) = 0 - S = -S.
    #    Then it computes a = -sum_{k=1}^K f_final(k) log(k)/k.
    #    Note that log(1)=0, so the f(1) term contributes 0 to the objective a.
    #    Therefore, the choice of f(1) in our output does not affect the objective a, 
    #    provided that the constraint sum f(k)/k = 0 is satisfied by the evaluator's adjustment.
    #    The constraint M(x) <= 1.0001 IS affected by f(1) because floor(x/1) = x.
    #    M(x) = f(1)*x + sum_{k>1} f(k) floor(x/k).
    #    If we output f(1)=0, the evaluator sets f(1) = -S.
    #    S = sum_{k=2}^K mu(k)/k.
    #    For large K, S is close to 0 (since sum mu(k)/k -> 0).
    #    Actually, sum_{k=1}^K mu(k)/k = 1 - sum_{k=1}^K (mu(k)/k - 1/k * 1_{k=1})? No.
    #    sum_{k=1}^K mu(k)/k approx 0.
    #    So f(1) will be small.
    #    
    #    However, the M(x) constraint is checked on the FINAL values.
    #    M(x) = f_final(1) * x + sum_{k=2}^K mu(k) floor(x/k).
    #    If we output f(1)=0, the evaluator computes f_final(1).
    #    We need to ensure that with f_final(1), M(x) <= 1.0001.
    #    
    #    Let's just output f(1)=0 and verify M(x) using the EXPECTED f_final(1) in our verification?
    #    Or does the evaluator check M(x) on the values WE return, before adjustment?
    #    "M(x) = sum_k f(k) * floor(x/k) is computed... if any M(x) > 1.0001 the function is infeasible"
    #    It doesn't explicitly say "after adjustment". But usually "the function" refers to the final state.
    #    However, the constraint sum f(k)/k = 0 is enforced by the evaluator.
    #    If the evaluator checks M(x) on the RAW values we return, and we return f(1)=0,
    #    then M(x) = sum_{k=2}^K mu(k) floor(x/k).
    #    For x=1, M(1) = mu(2)*0 + ... = 0? No.
    #    floor(1/1)=1, floor(1/2)=0.
    #    If we return f(1)=0, M(1) = 0*1 + 0 = 0.
    #    If we return f(1)=mu(1)=1, M(1) = 1.
    #    
    #    Let's look at the seed program. It returns specific values.
    #    The seed program's comment says: "f(1) is replaced by f(1) - S... so that afterwards sum_k f(k)/k = 0."
    #    And "M(x) ... is computed ... if any M(x) > 1.0001 ... infeasible".
    #    It is ambiguous whether M(x) is checked before or after the f(1) adjustment.
    #    Given "f(1) = 0 before this if 1 is not in the domain", it implies the adjustment is a preprocessing step.
    #    Then "M(x) ... is computed". This likely refers to the adjusted function.
    #    
    #    To be safe, we should construct f such that the ADJUSTED function satisfies M(x) <= 1.0001.
    #    Since we can control f(1) in our output, and the evaluator adjusts it, 
    #    we should output f(1) such that the adjusted f(1) makes M(x) small.
    #    Actually, the evaluator sets f(1) = f_out(1) - S.
    #    So we can set f_out(1) = S. Then f_final(1) = 0.
    #    If f_final(1) = 0, then M(x) = sum_{k=2}^K mu(k) floor(x/k).
    #    This is the "tail" of the Mobius sum.
    #    For x=1, M(1) = 0.
    #    For x=2, M(2) = mu(2)*1 + mu(3)*0... = -1.
    #    For x=3, M(3) = mu(2)*1 + mu(3)*1 = -1 -1 = -2.
    #    Wait, sum_{d|n} mu(d) = 0 for n>1.
    #    sum_{k=1}^infty mu(k) floor(x/k) = 1 for x=1, 0 for x>1.
    #    So sum_{k=2}^K mu(k) floor(x/k) = 1 - mu(1)*1 - (tail) = 0 - tail (for x>1).
    #    For x=1, sum_{k=2}^K mu(k) floor(1/k) = 0.
    #    So if f_final(1)=0, M(x) is the negative of the tail sum.
    #    The tail sum is small.
    #    
    #    So, strategy:
    #    1. Compute mu(k) for k=1..K.
    #    2. Set f_out(k) = mu(k) for k>1.
    #    3. Set f_out(1) = S, where S = sum_{k=2}^K mu(k)/k.
    #       This ensures f_final(1) = 0.
    #    4. Verify M(x) <= 1.0001 for the function with f_final(1)=0 and f_final(k)=mu(k).
    #       Note: The evaluator clips values to [-10, 10]. mu(k) is in {-1, 0, 1}, so clipping has no effect.
    #    
    #    If verification fails (unlikely for large K, but possible for small K or due to floating point),
    #    we can reduce K.

    K_max = 200000
    mu_full = compute_mobius(K_max)
    
    # We will try K from K_max downwards
    best_score = -np.inf
    best_result = None
    
    # We can test a few K values.
    # Since the objective a increases with K (more terms), we want the largest feasible K.
    # The tail sum decreases as K increases, so feasibility should improve or stay stable.
    # Actually, the tail sum T_K(x) = sum_{k>K} mu(k) floor(x/k).
    # M_final(x) = -T_K(x) (approx).
    # We need |T_K(x)| <= 1.0001.
    # As K increases, T_K(x) becomes smaller.
    # So if K=1000 is feasible, K=200000 should be more feasible.
    # Let's just try K_max first.
    
    K = K_max
    domain = np.arange(1, K + 1, dtype=np.int64)
    f_values = mu_full[:K].astype(np.float64)
    
    # Calculate S = sum_{k=2}^K mu(k)/k
    # f_values[0] is mu(1). f_values[1:] are mu(2)..mu(K)
    S = np.sum(f_values[1:] / domain[1:])
    
    # Set f_out(1) = S so that f_final(1) = 0
    f_values[0] = S
    
    # Verify M(x) for the FINAL function (f_final(1)=0, f_final(k)=mu(k) for k>1)
    # We construct the final values for verification
    f_verify = f_values.copy()
    f_verify[0] = 0.0
    
    if verify_M(f_verify, K):
        score = compute_objective(f_values, domain)
        best_score = score
        best_result = (domain.copy(), f_values.copy())
    else:
        # If K_max is not feasible (shouldn't happen, but just in case), try smaller K
        # Or maybe the issue is that for small x, the tail is not small enough?
        # No, tail gets smaller with larger K.
        # If K_max fails, it's a bug or numerical issue.
        # Let's try K=10000
        K = 10000
        domain = np.arange(1, K + 1, dtype=np.int64)
        f_values = mu_full[:K].astype(np.float64)
        S = np.sum(f_values[1:] / domain[1:])
        f_values[0] = S
        f_verify = f_values.copy()
        f_verify[0] = 0.0
        if verify_M(f_verify, K):
            score = compute_objective(f_values, domain)
            best_score = score
            best_result = (domain.copy(), f_values.copy())
        else:
            # Fallback to seed
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            best_result = (domain, values)

    if best_result is not None:
        return best_result

    # Absolute fallback
    domain = np.array([1], dtype=np.int64)
    values = np.array([0.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
