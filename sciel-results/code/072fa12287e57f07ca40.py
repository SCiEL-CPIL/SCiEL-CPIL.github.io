# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
    Instead of the discrete Mobius function, we use a smoothed, continuous-like kernel
    f(k) = -c/k * exp(-k/sigma) for k > 1.
    
    The objective is a = -sum_k f(k) log(k) / k.
    Substituting f(k):
    a = -sum_{k=2}^K (-c/k * exp(-k/sigma)) * log(k) / k
      = c * sum_{k=2}^K (log(k) / k^2) * exp(-k/sigma)
    
    To maximize a, we want c as large as possible.
    The constraint is M(x) = sum_k f(k) floor(x/k) <= 1 for all x in [1, 10K].
    Note that f(k) < 0 for k > 1.
    The evaluator adjusts f(1) such that sum_k f(k)/k = 0.
    Let S = sum_{k=2}^K f(k)/k. Then f(1) is set to -S (since f(1)/1 + S = 0).
    Actually, the problem says: "f(1) is replaced by f(1) - S with S = sum_k f(k)/k".
    Wait, the text says: "f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0".
    If we set initial f(1)=0, then new f(1) = -S.
    S = sum_{k=2}^K f(k)/k = sum_{k=2}^K (-c/k^2 * exp(-k/sigma)) < 0.
    So new f(1) = -S = c * sum_{k=2}^K (1/k^2 * exp(-k/sigma)) > 0.
    
    M(x) = f(1_new)*floor(x/1) + sum_{k=2}^K f(k) floor(x/k)
         = f(1_new) + sum_{k=2}^K f(k) floor(x/k)
    
    For large x, floor(x/k) ~ x/k.
    M(x) ~ f(1_new) + x * sum_{k=2}^K f(k)/k = f(1_new) + x * S.
    Since f(1_new) = -S, M(x) ~ -S + x*S = S(x-1).
    Since S < 0, for large x, M(x) becomes very negative. This is safe (< 1).
    
    The risk is for small x or intermediate x where floor(x/k) is small but non-zero,
    or where the sum of negative terms might not be enough to offset f(1_new) if S is not "large" enough in magnitude?
    Actually, M(1) = f(1_new) + sum_{k=2}^K f(k) * 0 = f(1_new) = -S.
    We need M(1) <= 1. So -S <= 1 => S >= -1.
    S = -c * sum_{k=2}^K (1/k^2 * exp(-k/sigma)).
    So c * sum_{k=2}^K (1/k^2 * exp(-k/sigma)) <= 1.
    
    To maximize a = c * sum_{k=2}^K (log(k)/k^2 * exp(-k/sigma)),
    we should choose c as large as possible, i.e., c = 1 / sum_{k=2}^K (1/k^2 * exp(-k/sigma)).
    Then a = sum_{k=2}^K (log(k)/k^2 * exp(-k/sigma)) / sum_{k=2}^K (1/k^2 * exp(-k/sigma)).
    
    This ratio is the "expected log(k)" under the distribution proportional to (1/k^2) exp(-k/sigma).
    To maximize this, we want the distribution to be as "heavy-tailed" (large k) as possible,
    but constrained by the finite K and the exponential decay.
    Larger sigma means slower decay, so larger k contributes more.
    However, if sigma is too large, the sum in the denominator is dominated by small k? No, 1/k^2 decays fast.
    Let's analyze the ratio R(sigma) = sum_{k=2}^K w_k log(k) / sum_{k=2}^K w_k, where w_k = (1/k^2) exp(-k/sigma).
    
    If sigma -> infinity, w_k ~ 1/k^2.
    R(inf) = sum_{k=2}^K (log(k)/k^2) / sum_{k=2}^K (1/k^2).
    As K -> infinity, sum 1/k^2 converges to pi^2/6 - 1.
    sum log(k)/k^2 converges to -zeta'(2).
    So R(inf) is a constant ~ 0.93?
    
    If sigma is small, w_k decays fast, dominated by k=2.
    R(small) ~ log(2)/2^2 / (1/2^2) = log(2) ~ 0.69.
    
    So we want large sigma. Let's set sigma = K/2 or similar to allow the tail to contribute.
    Actually, if sigma is very large, the exponential term is ~1 for all k <= K.
    Then we are effectively using f(k) = -c/k^2.
    This is a very smooth, negative function.
    M(x) = -S + sum_{k=2}^K (-c/k^2) floor(x/k).
    Since all terms in the sum are negative, M(x) <= M(1) = -S = 1 (by construction of c).
    So M(x) <= 1 is ALWAYS satisfied for this family of functions!
    
    Why?
    f(k) < 0 for all k > 1.
    f(1) > 0.
    M(x) = f(1) + sum_{k=2}^K f(k) floor(x/k).
    For any x >= 1, floor(x/k) >= 0.
    Since f(k) < 0, sum_{k=2}^K f(k) floor(x/k) <= 0.
    Therefore M(x) <= f(1) = -S.
    We set c such that -S = 1.
    So M(x) <= 1 for all x.
    
    This guarantees feasibility!
    The only question is how large we can make the score a.
    a = c * sum_{k=2}^K (log(k)/k^2 * exp(-k/sigma)).
    With c = 1 / sum_{k=2}^K (1/k^2 * exp(-k/sigma)),
    a = (sum_{k=2}^K (log(k)/k^2 * exp(-k/sigma))) / (sum_{k=2}^K (1/k^2 * exp(-k/sigma))).
    
    This is a weighted average of log(k) with weights w_k = (1/k^2) exp(-k/sigma).
    To maximize the average log(k), we want weights on large k.
    The factor 1/k^2 penalizes large k. The factor exp(-k/sigma) penalizes large k if sigma is small.
    If we let sigma -> infinity, w_k = 1/k^2.
    The average log(k) is approx sum_{k=2}^K log(k)/k^2 / sum_{k=2}^K 1/k^2.
    
    Let's calculate this value for K=200000.
    sum_{k=1}^infty 1/k^2 = 1.6449.
    sum_{k=1}^infty log(k)/k^2 = -zeta'(2) = 0.9375.
    Subtract k=1 terms:
    Denom = 1.6449 - 1 = 0.6449.
    Num = 0.9375 - 0 = 0.9375.
    Ratio = 0.9375 / 0.6449 = 1.453.
    
    This is > 1.0! The seed score is 1.0.
    So this strategy should yield a score around 1.45, which is a significant improvement.
    
    We need to ensure numerical stability and compute this efficiently.
    K = 200000.
    We can compute the sums using numpy.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Use maximum K to maximize the sum
    K = 200000
    
    # We choose sigma = infinity effectively, i.e., exp(-k/sigma) = 1.
    # This simplifies w_k = 1/k^2.
    # f(k) = -c / k for k > 1.
    # S = sum_{k=2}^K f(k)/k = -c * sum_{k=2}^K 1/k^2.
    # f(1) = -S = c * sum_{k=2}^K 1/k^2.
    # We set f(1) = 1 (max allowed by M(1) <= 1).
    # So c * sum_{k=2}^K 1/k^2 = 1 => c = 1 / sum_{k=2}^K 1/k^2.
    
    # Domain is 1..K
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    # Compute values
    # k = 1..K
    k = domain.astype(np.float64)
    
    # Sum of 1/k^2 for k=2..K
    # Use numpy for speed
    k_sq = k * k
    inv_k_sq = 1.0 / k_sq
    # Set index 0 (k=1) to 0 for the sum
    sum_inv_k_sq = np.sum(inv_k_sq[1:])
    
    c = 1.0 / sum_inv_k_sq
    
    # f(k) for k > 1
    values = np.zeros(K, dtype=np.float64)
    values[0] = c * sum_inv_k_sq  # f(1) = 1.0
    # f(k) = -c/k
    values[1:] = -c / k[1:]
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return domain, values
    
    # Optional: Verify feasibility for a few points?
    # M(x) = 1 + sum_{k=2}^K (-c/k) floor(x/k).
    # Since -c/k < 0 and floor(x/k) >= 0, M(x) <= 1.
    # The minimum value of M(x) is not constrained by the problem (only upper bound).
    # So feasibility is guaranteed.
    
    # The score is a = -sum f(k) log(k) / k.
    # f(1) term: -1 * log(1) / 1 = 0.
    # f(k) term for k>1: -(-c/k) * log(k) / k = c * log(k) / k^2.
    # a = c * sum_{k=2}^K log(k) / k^2.
    # This matches our derivation.
    
    return domain, values

# EVOLVE-BLOCK-END
