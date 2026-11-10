# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n=1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
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
    return mu


def _compute_M_vectorized(domain, values, max_x):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x]
    using a vectorized difference-array approach.
    """
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = int(domain[idx])
        f_k = values[idx]
        if f_k == 0.0:
            continue
        
        # j ranges from 1 to max_x // k
        max_j = max_x // k
        if max_j == 0:
            continue
        
        # For each j, the interval is [j*k, min((j+1)*k - 1, max_x)]
        # Contribution is f_k * j
        # Start positions: k, 2k, 3k, ..., max_j * k
        starts = np.arange(1, max_j + 1, dtype=np.int64) * k
        # End positions (exclusive): (j+1)*k for j=1..max_j-1, max_x+1 for last
        ends = np.arange(2, max_j + 2, dtype=np.int64) * k
        ends = np.minimum(ends, max_x + 1)
        # Values to add: f_k * 1, f_k * 2, ..., f_k * max_j
        vals = f_k * np.arange(1, max_j + 1, dtype=np.float64)
        
        # Use np.add.at for scattered updates (handles duplicates safely)
        np.add.at(diff, starts, vals)
        np.add.at(diff, ends, -vals)
    
    M = np.cumsum(diff)
    return M[:max_x + 1]


def _solve_lp(K, time_limit=30.0):
    """
    Solve the LP for a given K to find the optimal f(k) values.
    Returns (domain, values) or None if failed.
    """
    start = time.time()
    
    max_x = 10 * K
    n_vars = K
    n_cons = max_x
    
    # A_ub is n_cons x n_vars
    # A_ub[i, j] = floor((i+1) / (j+1))
    x_vals = np.arange(1, max_x + 1)[:, np.newaxis]
    k_vals = np.arange(1, K + 1)[np.newaxis, :]
    A_ub = (x_vals // k_vals).astype(np.float64)
    
    b_ub = np.ones(max_x)
    
    # Objective function c
    # Minimize sum c_i f_i
    # c_i = log(i+1) / (i+1)
    c = np.log(np.arange(1, K + 1)) / np.arange(1, K + 1)
    
    # Equality constraint: sum f(k) / k = 0
    A_eq = (1.0 / np.arange(1, K + 1)).reshape(1, -1)
    b_eq = np.array([0.0])
    
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10.0, 10.0)] * K
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            return None
            
        f_vals = res.x
        
        # Filter out very small values to reduce domain size
        # The evaluator handles any distinct positive integers.
        # However, keeping all K is fine as long as K <= 200000.
        # But for cleanliness and to avoid numerical noise in M(x) check,
        # we can keep all or filter. Let's keep all for now, as LP solution is dense.
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = f_vals.astype(np.float64)
        
        return domain, values
        
    except Exception:
        return None


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    best_result = None
    best_a = -np.inf

    # Strategy:
    # 1. Try LP for small K (fast, exact optimal for that K).
    # 2. Try Mobius for larger K (heuristic, scales better, but suboptimal for finite K).
    # 3. Pick the best feasible result.

    # --- Phase 1: LP Optimization ---
    # LP is feasible for K up to ~1000-2000 depending on hardware, but gets slow.
    # K=200 is very fast and good. K=500 is okay. K=1000 might be slow.
    # Let's try a few sizes.
    lp_Ks = [200, 500, 1000]
    
    for K in lp_Ks:
        if time.time() - start_time > deadline - 120:
            break
            
        res = _solve_lp(K, time_limit=deadline - 120 - (time.time() - start_time))
        if res is not None:
            domain, values = res
            
            # Check feasibility with the same strictness as the evaluator
            # The evaluator clips values to [-10, 10] first.
            clipped_values = np.clip(values, -10.0, 10.0)
            
            # Evaluator logic:
            # 1. Clip
            # 2. S = sum_k f(k)/k over clipped
            # 3. f(1) = f(1) - S (if 1 in domain, else 0 - S)
            # 4. Check M(x) <= 1.0001
            # 5. Compute a
            
            # Let's simulate the evaluator exactly to compute 'a' and check feasibility
            # Note: The LP already enforces sum f(k)/k = 0 and M(x) <= 1.
            # But the evaluator's adjustment of f(1) is specific.
            # Wait, the LP enforces sum f(k)/k = 0.
            # The evaluator computes S = sum f(k)/k. If S=0, then f(1) is unchanged.
            # So the LP solution should satisfy the evaluator's normalization step trivially.
            
            # However, we must verify M(x) <= 1.0001 with the *evaluator's* f values.
            # Since S=0, f_eval = f_lp.
            # So we just need to check M(x) for f_lp.
            
            # But wait, the LP constraint is M(x) <= 1.
            # The evaluator allows M(x) <= 1.0001.
            # So LP solutions are feasible.
            
            # Compute a
            k_float = domain.astype(np.float64)
            a = -np.sum(clipped_values * np.log(k_float) / k_float)
            
            if a > best_a:
                best_a = a
                best_result = (domain, clipped_values)

    # --- Phase 2: Mobius Heuristic ---
    # If we have time, try larger K with Mobius.
    # Mobius is not optimal for finite K (it doesn't minimize the objective exactly under constraints),
    # but it approaches 1.0 as K -> infinity.
    # For small K, LP is better. For large K, Mobius might be better than small-K LP?
    # Actually, LP with K=1000 should be quite good.
    # Let's see if Mobius with K=5000 or 10000 beats LP with K=1000.
    
    mobius_Ks = [2000, 5000, 10000, 20000, 50000]
    
    for K in mobius_Ks:
        if time.time() - start_time > deadline - 60:
            break
            
        mu = _compute_mobius(K)
        mask = mu[1:K + 1] != 0
        domain = np.where(mask)[0] + 1
        values = mu[domain].astype(np.float64)
        
        if len(domain) == 0:
            continue
        
        # Simulate Evaluator
        clipped_values = np.clip(values, -10.0, 10.0)
        
        # Compute S
        # S = sum_k f(k)/k
        # f(1) is mu(1) = 1.
        # S = sum_{k in domain} clipped_values[k] / k
        
        k_float = domain.astype(np.float64)
        S = np.sum(clipped_values / k_float)
        
        # Adjust f(1)
        # "f(1) is replaced by f(1) - S ... (f(1) = 0 before this if 1 is not in the domain)"
        # In Mobius, 1 is always in the domain (mu(1)=1).
        # So f(1) becomes 1 - S.
        
        # We need to construct the final values array for M(x) and a.
        # Let's create a dict or array for final values.
        # Since domain is sorted, we can find index of 1.
        idx_1 = np.where(domain == 1)[0]
        
        final_values = clipped_values.copy()
        if len(idx_1) > 0:
            final_values[idx_1[0]] -= S
        else:
            # If 1 not in domain, f(1) = 0 - S = -S.
            # We would need to add 1 to domain?
            # The task says "domain holds distinct positive integers".
            # "f(1) is replaced by ... (f(1) = 0 before this if 1 is not in the domain)"
            # This implies if 1 is not in domain, we treat it as if f(1)=-S for the sum check?
            # "so that afterwards sum_k f(k)/k = 0"
            # If 1 is not in domain, the sum is over the existing k.
            # The instruction says "f(1) is replaced by f(1) - S".
            # If 1 is not in domain, f(1) is effectively 0. So new f(1) = -S.
            # Does this add 1 to the domain?
            # "domain holds distinct positive integers ... values holds the finite reals f(k)"
            # If the evaluator adds 1 to the domain, m changes?
            # "returning (domain, values): two 1-D arrays of equal length m >= 1"
            # The evaluator takes MY domain and values.
            # It computes S from MY domain/values.
            # Then it says "f(1) is replaced by...".
            # If 1 is not in my domain, does it add it?
            # Usually, "partial function" implies domain is fixed.
            # But the normalization step "f(1) = 0 before this if 1 is not in the domain" suggests
            # that the evaluator might be considering the function defined on 1..K implicitly?
            # No, it says "sum_k f(k)/k" over the clipped values.
            # If 1 is not in domain, it's not in the sum.
            # Then "f(1) is replaced by f(1) - S".
            # If 1 is not in domain, this operation is void? Or does it add 1?
            # Given the Mobius context, 1 is always included.
            # Let's assume 1 is in domain for Mobius.
            pass
        
        # Check M(x) with final_values
        max_x = 10 * K
        M = _compute_M_vectorized(domain, final_values, max_x)
        
        if np.max(M[1:max_x + 1]) > 1.0001:
            continue
        
        # Compute a
        # a = -sum_k f(k) * log(k) / k
        # Using final_values
        a = -np.sum(final_values * np.log(k_float) / k_float)
        
        if a > best_a:
            best_a = a
            best_result = (domain, final_values)

    if best_result is None:
        # Fallback to the seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_result


# EVOLVE-BLOCK-END
