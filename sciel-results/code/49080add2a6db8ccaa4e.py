import numpy as np
import time

def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max] after clipping and f(1) adjustment."""
    clipped = np.clip(values, -10.0, 10.0)
    S = np.sum(clipped / domain.astype(np.float64))
    x_max = 10 * K_max
    h = np.zeros(x_max + 1, dtype=np.float64)
    for i in range(len(domain)):
        k = int(domain[i])
        f_val = clipped[i]
        if k == 1:
            f_val = f_val - S
        if k > x_max:
            continue
        h[k::k] += f_val
    M_vals = np.cumsum(h[1:])
    if len(M_vals) == 0:
        return True
    return np.max(M_vals) <= tolerance

def _compute_a(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    clipped = np.clip(values, -10.0, 10.0)
    k_float = domain.astype(np.float64)
    S = np.sum(clipped / k_float)
    adjusted = clipped.copy()
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        adjusted[idx_1[0]] = clipped[idx_1[0]] - S
    if len(k_float) == 0:
        return 0.0
    return -np.sum(adjusted * np.log(k_float) / k_float)

def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes a = -sum_k f(k) log(k)/k
    subject to M(x) <= 1 for all x in [1, 10K], after the f(1) adjustment.
    Uses a greedy / iterative LP approach to build a good partial function.
    """
    start_time = time.time()
    deadline = 590  # Safety margin before 600s kill

    # Seed
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_domain = seed_domain.copy()
    best_values = seed_values.copy()
    best_a = _compute_a(seed_domain, seed_values)

    # Try small LPs with increasing K
    K_list = [100, 200, 300, 500, 700]
    
    for K in K_list:
        if time.time() - start_time > deadline - 10:
            break
        try:
            from scipy.optimize import linprog
            x_arr = np.arange(1, 10 * K + 1, dtype=np.int64)
            j_arr = np.arange(1, K + 1, dtype=np.int64)
            A_ub = (x_arr[:, None] - np.remainder(x_arr[:, None], j_arr[None, :])) // j_arr[None, :]
            b_ub = np.ones(10 * K, dtype=np.float64)
            c = np.zeros(K)
            k_vals = np.arange(1, K + 1)
            c[1:] = np.log(k_vals[1:]) / k_vals[1:]
            bounds = [(-10.0, 10.0)] * K
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                f_vals = res.x
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = f_vals.astype(np.float64)
                if _check_feasibility(domain, values, K):
                    a_val = _compute_a(domain, values)
                    if a_val > best_a:
                        best_a = a_val
                        best_domain = domain.copy()
                        best_values = values.copy()
        except Exception:
            continue

    # Iterative improvement: try to add new keys to domain one at a time
    # using a small LP for a few candidate keys
    max_iter = 5
    for _ in range(max_iter):
        if time.time() - start_time > deadline - 5:
            break
        current_K = int(best_domain.max()) if len(best_domain) > 0 else 1
        # Try candidates up to max(current_K + 100, 2000)
        max_try = max(current_K + 200, 2000)
        improved = False
        for new_k in range(current_K + 1, max_try + 1):
            if time.time() - start_time > deadline - 5:
                break
            if new_k > 200000:
                break
            # Try adding new_k to domain with value 1.0 and re-solve a small LP
            # on the union of current domain and {new_k}
            # But solving an LP on a growing domain each time is expensive.
            # Instead, just test if adding new_k with value v (optimistic) helps.
            # Use a simple heuristic: if f(k) can be negative, adding it with -10 might help.
            test_domain = np.append(best_domain, new_k)
            test_domain.sort()
            # Set new value to -10 (most negative)
            test_values = np.append(best_values, -10.0)
            # Reorder values to match sorted domain
            idx = np.argsort(test_domain)
            test_domain = test_domain[idx]
            test_values = test_values[idx]
            if _check_feasibility(test_domain, test_values, int(test_domain.max())):
                a_val = _compute_a(test_domain, test_values)
                if a_val > best_a + 1e-6:
                    best_a = a_val
                    best_domain = test_domain.copy()
                    best_values = test_values.copy()
                    improved = True
                    # Continue trying to add more in the same iteration
                # else: this new_k with -10 doesn't help; skip
        if not improved:
            break

    return best_domain, best_values
