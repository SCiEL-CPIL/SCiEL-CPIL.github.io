# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
import sys

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600.0  # seconds total
    safety_margin = 30.0  # return by 570s

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
                        mu[i * p] = 0  # i*p has p^2 as factor
                        break
                    else:
                        mu[i * p] = -mu[i]

        return mu[1:]  # mu[1], mu[2], ..., mu[K]

    def verify_M(f_values, K):
        """
        Verify that M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
        """
        max_x = 10 * K
        if max_x <= 200000:
            # Compute M(x) for all x using vectorized approach
            # M(x) = sum_{k=1}^{K} f(k) * floor(x/k)
            # For x from 1 to max_x:
            diff = np.zeros(max_x + 1, dtype=np.float64)
            for k_idx in range(K):
                k = k_idx + 1
                fk = f_values[k_idx]
                if fk == 0:
                    continue
                # Add fk at positions k, 2k, 3k, ... up to max_x
                multiples = np.arange(k, max_x + 1, k)
                diff[multiples] += fk
            M_vals = np.cumsum(diff)
            # M_vals[x] = M(x) for x = 1..max_x
            if np.any(M_vals > 1.0001):
                return False
            return True
        else:
            # For large K, just check sampled x values
            check_xs = set()
            for x in range(1, min(10000, max_x + 1)):
                check_xs.add(x)
            for x in range(10000, max_x + 1, max(1, (max_x - 10000) // 1000)):
                check_xs.add(x)
            check_xs = sorted(check_xs)
            for x in check_xs:
                lim = min(K, x)
                M_x = np.sum(f_values[:lim] * (x // np.arange(1, lim + 1)))
                if M_x > 1.0001:
                    return False
            return True

    def compute_objective(f_values, domain):
        """Compute a = -sum_k f(k) * log(k) / k."""
        logs = np.log(domain.astype(np.float64))
        return -np.sum(f_values * logs / domain.astype(np.float64))

    # Start with a small K and increase
    best_score = -np.inf
    best_result = None

    # Try K = 1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000
    K_candidates = [1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]

    for K in K_candidates:
        if not elapsed_ok():
            break

        try:
            mu = compute_mobius(K)
            domain = np.arange(1, K + 1, dtype=np.int64)
            f_values = mu.astype(np.float64)

            # Verify feasibility
            if not verify_M(f_values, K):
                # Try with a smaller effective K or skip
                continue

            score = compute_objective(f_values, domain)
            if score > best_score:
                best_score = score
                best_result = (domain.copy(), f_values.copy())
        except Exception:
            continue

    # If we haven't found a good result with the simple Mobius approach,
    # try LP for small K to potentially squeeze out a better score.
    # The LP approach is computationally expensive, so we limit it to small K.
    if best_score < 1.001: # Only if Mobius didn't do well enough
        for K in [5, 10, 20, 50]:
            if not elapsed_ok():
                break
            try:
                # Objective: maximize a = -sum f(k) log(k) / k
                # In linprog, we minimize c^T x. So c = - (log(k) / k)
                domain = np.arange(1, K + 1, dtype=np.int64)
                logs = np.log(domain.astype(np.float64))
                c = -(logs / domain.astype(np.float64))

                # Constraints:
                # 1. sum f(k) / k = 0  =>  sum f(k) * (1/k) = 0
                # This is an equality constraint.
                A_eq = np.zeros((1, K))
                A_eq[0, :] = 1.0 / domain.astype(np.float64)
                b_eq = np.array([0.0])

                # 2. M(x) <= 1.0001 for all x in [1, 10K]
                # M(x) = sum_{k=1}^{min(K,x)} f(k) * floor(x/k)
                # This is a set of linear inequalities: A_ub x <= b_ub
                max_x = 10 * K
                # We don't need to check all x if K is small, but for K=50, 10K=500 is fine.
                # For K=100, 10K=1000 is fine.
                # For K=200, 10K=2000 might be a bit large for dense LP, but HiGHS is fast.
                # Let's sample x if K is large, but for K<=50, we can use all x.
                # Actually, the constraint is tightest at x where the sum is largest.
                # But to be safe, we include all x in [1, 10K].
                
                A_ub = np.zeros((max_x, K))
                b_ub = np.ones(max_x) * 1.0001
                
                for x in range(1, max_x + 1):
                    for k in range(1, min(K, x) + 1):
                        A_ub[x-1, k-1] = x // k
                
                # Bounds for f(k):
                # The task says "values are clipped to [-10, 10]".
                # So f(k) must be in [-10, 10].
                bounds = [(-10.0, 10.0)] * K

                # Solve LP
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                
                if res.status == 0:
                    f_lp = res.x
                    # Check feasibility again just in case
                    if verify_M(f_lp, K):
                        score_lp = compute_objective(f_lp, domain)
                        if score_lp > best_score:
                            best_score = score_lp
                            best_result = (domain.copy(), f_lp.copy())
            except Exception as e:
                pass

    if best_result is not None:
        return best_result

    # Fallback: the seed program
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
