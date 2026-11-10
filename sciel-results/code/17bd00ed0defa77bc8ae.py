# EVOLVE-BLOCK-START
import numpy as np
import time
from math import log


def _compute_mobius(N):
    """Compute the Möbius function mu(n) for n = 1..N."""
    mu = np.ones(N + 1, dtype=np.float64)
    primes = []
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            # Mark multiples
            for j in range(i * i, N + 1, i):
                if j <= N:
                    is_prime[j] = False
            # Apply Möbius rule: mu(n) = -mu(n/i) if i is a prime factor
            # Actually, standard sieve for mu:
            mu[i] = -1
            for p in primes:
                if p * i > N:
                    break
                mu[p * i] = -mu[i]
                if i % p == 0:
                    mu[p * i] = 0
                    break
    return mu


def _is_feasible(domain, values, K_max=200000, x_limit_mult=10, tol=1.0001):
    """
    Check if M(x) = sum_k f(k) * floor(x/k) <= tol for all x in [1, x_limit_mult * max(domain)].
    Note: The constraint is on M(x) for all real x. Since M(x) is a step function increasing
    with x (if f has positive parts) or not monotonic generally, we must check at discontinuities.
    However, the task says "for every integer x in [1, 10K]". Wait, the prompt says:
    "M(x) = sum_k f(k) * floor(x/k) is computed for every integer x in [1, 10K]"
    Actually, floor(x/k) is constant between integers? No, x is the variable.
    If x is real, floor(x/k) changes at multiples of k.
    The max of M(x) over real x in [1, 10K] occurs just before a discontinuity?
    Let's re-read: "M(x) = sum_k f(k) * floor(x/k) is computed for every integer x in [1, 10K]".
    It explicitly says "for every integer x". So we only need to check integer x.
    """
    if len(domain) == 0:
        return True
    
    K = np.max(domain)
    x_max = 10 * K
    
    # We need to check M(x) for x = 1, ..., x_max.
    # M(x) = sum_{k in domain} f(k) * floor(x/k)
    # This can be computed efficiently if we iterate x? No, that's O(x_max * m).
    # x_max can be up to 2,000,000. m can be up to 200,000.
    # 2e6 * 2e5 is too slow.
    
    # Alternative: M(x) is a step function. The values change when x crosses a multiple of some k.
    # However, the prompt says "computed for every integer x".
    # Let's check the cost: "cost about m * 10K". This implies the evaluator does O(m * 10K).
    # If K=200000, 10K=2e6. m=200000. Product is 4e11. That would take forever.
    # Wait, "cost about m * 10K" might be an approximation or I am misinterpreting.
    # If the evaluator actually runs in 600s, maybe K is small or m is small?
    # Or maybe the evaluation is vectorized?
    # Let's assume we need to be efficient.
    
    # If we use numpy:
    # x = np.arange(1, x_max + 1)
    # M = np.zeros_like(x, dtype=float)
    # for k, f_k in zip(domain, values):
    #     M += f_k * (x // k)
    # This is O(m * x_max). If m=1000, x_max=2000000, that's 2e9 ops. Might be borderline in Python loop but ok in numpy?
    # But doing it inside the solver loop is too slow if we do it often.
    
    # Heuristic for feasibility check during optimization:
    # Only check a subset of x or use a smarter check.
    # Or, notice that if we use the Mobius function, M(x) = 1 for x >= 1 (if domain is large enough).
    # Specifically, sum_{d|n} mu(d) = 0 for n>1, 1 for n=1.
    # sum_{k<=x} mu(k) floor(x/k) = sum_{n<=x} sum_{k|n, k<=x} mu(k) = sum_{n<=x} 1 = x? No.
    # Identity: sum_{d|n} mu(d) = 1 if n=1 else 0.
    # sum_{k=1}^x mu(k) floor(x/k) = sum_{k=1}^x mu(k) sum_{j=1}^{floor(x/k)} 1 = sum_{n=1}^x sum_{k|n} mu(k).
    # The inner sum is 1 if n=1, 0 otherwise.
    # So the total sum is 1 for any x >= 1.
    # So M(x) = 1 for all x >= 1 if the domain includes all k up to x.
    # If the domain is finite (up to K), then for x > K, floor(x/k) is still there.
    # sum_{k in D} mu(k) floor(x/k).
    # If D = {1..K}, then for x >= K, M(x) = 1.
    # For x < K, M(x) = 1 as well?
    # sum_{k=1}^K mu(k) floor(x/k). If K >= x, this is sum_{k=1}^x mu(k) floor(x/k) = 1.
    # So if we use the full Mobius function up to K, M(x) is exactly 1 for all x >= 1.
    # This satisfies M(x) <= 1.0001.
    
    # So, any prefix of the Mobius function is feasible?
    # Let D = {1, ..., K}. f(k) = mu(k).
    # M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # If x <= K, M(x) = 1.
    # If x > K, M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # We know sum_{k=1}^x mu(k) floor(x/k) = 1.
    # So sum_{k=1}^K mu(k) floor(x/k) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
    # The tail sum is small?
    # Actually, the Mertens function M(n) = sum_{k=1}^n mu(k) is small (O(sqrt(n))).
    # The error term is related to the Mertens function.
    # It is known that sum_{k=1}^K mu(k) floor(x/k) = 1 + O(x/K * log(x/K)?) No.
    # Let's just check numerically for the specific K we choose.
    
    # For the purpose of the solver, if we stick to the Mobius function, it's likely feasible.
    # We can verify feasibility at the end.
    
    # For a quick check during optimization, we can sample x.
    # But since we are refining the Mobius family, we might just trust it or check a few points.
    
    # Let's implement a fast check for the specific case of Mobius or similar.
    # Or just check x = 1, 2, ..., min(100, x_max) and some large x.
    
    # Actually, the prompt says "if any M(x) > 1.0001 ... scores 0".
    # So we must be sure.
    
    # Let's use a vectorized check for a subset of x if m is small, or a smarter method.
    # If m is large, we can't iterate over all x in Python.
    
    # However, if we are just using the Mobius function, we can compute the max M(x) efficiently?
    # Not easily.
    
    # Let's assume the candidate solutions will be close to Mobius.
    # We will do a full check at the end of the optimization.
    # During optimization, we will use a relaxed constraint or a sampled check.
    
    pass # Placeholder, will be implemented in the main function logic


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600 - 10  # 590s safety margin
    
    # Strategy:
    # 1. Compute Mobius function up to K_max.
    # 2. The objective is a = -sum f(k) log(k) / k.
    # 3. Constraint: sum f(k)/k = 0 (handled by evaluator by adjusting f(1), but we should aim for it or account for it).
    #    Actually, the evaluator replaces f(1) with f(1) - S where S = sum f(k)/k.
    #    So the effective f(1) becomes 0 if we don't include 1? No, "f(1) = 0 before this if 1 is not in the domain".
    #    If 1 is in the domain, f(1) is adjusted.
    #    The metric is computed on the CLIPPED values?
    #    "values are clipped to [-10, 10]. Then f(1) is replaced..."
    #    So we should provide values in [-10, 10] ideally, or they will be clipped.
    #    Mobius values are -1, 0, 1. So they are safe.
    
    # 4. Feasibility: M(x) <= 1.0001.
    #    For f(k) = mu(k), M(x) = 1 for all x >= 1 (if domain is 1..K and x<=K, or if tail is negligible).
    #    Actually, for x > K, M(x) = 1 - error. The error can be positive or negative.
    #    We need M(x) <= 1.0001.
    #    If the error is positive, we might violate it.
    #    The error is sum_{k=K+1}^x mu(k) floor(x/k).
    #    This is related to the Mertens function.
    #    To be safe, we might need to adjust f(k) slightly or choose K such that the error is negative or small.
    
    # 5. Optimization:
    #    We want to maximize a = -sum f(k) log(k) / k.
    #    With f(k) = mu(k), a = -sum mu(k) log(k) / k.
    #    This sum converges to 1.
    #    Can we do better than mu(k)?
    #    The constraints are:
    #    - sum f(k)/k = S (will be subtracted from f(1)).
    #    - M(x) <= 1.0001.
    #    
    #    Let's try to optimize f(k) for k=1..K.
    #    Since the evaluator adjusts f(1), the effective constraint on the other f(k) is:
    #    M(x) = sum_{k in D, k != 1} f(k) floor(x/k) + f(1)_eff * floor(x/1)
    #    f(1)_eff = f(1) - sum_{k in D} f(k)/k.
    #    So M(x) = sum_{k in D, k != 1} f(k) floor(x/k) + (f(1) - sum_{k in D} f(k)/k) * x
    #    = f(1) * x - x * sum_{k in D} f(k)/k + sum_{k in D, k != 1} f(k) floor(x/k)
    #    = f(1) * x + sum_{k in D, k != 1} f(k) (floor(x/k) - x/k)
    #    
    #    Note that floor(x/k) - x/k is in (-1, 0].
    #    Let g(k) = floor(x/k) - x/k.
    #    M(x) = f(1) * x + sum_{k=2}^K f(k) g_k(x).
    #    
    #    We want M(x) <= 1 for all x.
    #    And we want to maximize a = -sum f(k) log(k)/k.
    #    Note that a depends on f(1) too: -f(1) log(1)/1 - sum_{k=2} f(k) log(k)/k = -sum_{k=2} f(k) log(k)/k.
    #    So f(1) does not affect the objective!
    #    This means we can choose f(1) arbitrarily (within clipping bounds) to help satisfy constraints?
    #    Yes! f(1) is a free variable that doesn't cost in the objective.
    #    
    #    So the problem reduces to:
    #    Choose f(2), ..., f(K) to maximize -sum_{k=2}^K f(k) log(k)/k.
    #    Subject to:
    #    For all x in [1, 10K]:
    #    f(1) * x + sum_{k=2}^K f(k) (floor(x/k) - x/k) <= 1.
    #    
    #    And f(1) must be such that the final f(1) (after adjustment) is in [-10, 10]?
    #    "values are clipped to [-10, 10]". So the input values are clipped.
    #    So f(1) input must be in [-10, 10].
    #    
    #    Since f(1) multiplies x, and x goes up to 10K, f(1) must be very small (close to 0) to not violate the constraint for large x, unless the sum term cancels it.
    #    But the sum term is bounded?
    #    g_k(x) is in (-1, 0].
    #    If we choose f(k) for k>=2, we can determine the required f(1).
    #    
    #    Let's fix f(2)..f(K).
    #    Then for each x, we need f(1) <= (1 - sum_{k=2} f(k) g_k(x)) / x.
    #    So f(1) <= min_x (1 - sum_{k=2} f(k) g_k(x)) / x.
    #    Also f(1) >= -10 (clipping).
    #    
    #    To maximize feasibility, we can set f(1) to the maximum allowed value (closest to 10) or just check if the max allowed value is >= -10.
    #    Actually, we just need SOME f(1) in [-10, 10] to work.
    #    So we need:
    #    max_{f1 in [-10,10]} min_x (f1 * x + sum_{k=2} f(k) g_k(x)) <= 1
    #    This is equivalent to:
    #    For all x, there exists f1 in [-10, 10] such that f1 * x + S(x) <= 1.
    #    This requires:
    #    If x > 0, f1 <= (1 - S(x)) / x.
    #    We need the lower bound of f1 (which is -10) to be <= the upper bound required for all x.
    #    So we need: -10 <= (1 - S(x)) / x for all x.
    #    => -10 x <= 1 - S(x)
    #    => S(x) - 1 <= 10 x
    #    => sum_{k=2} f(k) g_k(x) - 1 <= 10 x
    #    Since g_k(x) is negative, if f(k) are positive, S(x) is negative.
    #    This constraint is likely easy to satisfy.
    #    
    #    Wait, we also need f(1) to be the one that the evaluator uses.
    #    The evaluator calculates S = sum f(k)/k.
    #    Then f(1)_new = f(1) - S.
    #    Then it checks M(x) using f(1)_new.
    #    My derivation above assumed f(1)_eff is the variable.
    #    Let's restart the derivation with variables u_k = f(k) for k=2..K and v = f(1).
    #    S = v/1 + sum_{k=2} u_k/k.
    #    f(1)_eff = v - S = v - (v + sum_{k=2} u_k/k) = - sum_{k=2} u_k/k.
    #    So f(1)_eff is DETERMINED by u_k!
    #    f(1)_eff = - U, where U = sum_{k=2}^K u_k/k.
    #    
    #    So M(x) = (-U) * x + sum_{k=2}^K u_k floor(x/k).
    #    M(x) = sum_{k=2}^K u_k (floor(x/k) - x/k) + (-U) * (x - 0)? No.
    #    M(x) = f(1)_eff * x + sum_{k=2}^K u_k floor(x/k)
    #    = -U * x + sum_{k=2}^K u_k floor(x/k)
    #    = sum_{k=2}^K u_k floor(x/k) - x sum_{k=2}^K u_k/k
    #    = sum_{k=2}^K u_k (floor(x/k) - x/k).
    #    
    #    So M(x) = sum_{k=2}^K u_k h_k(x), where h_k(x) = floor(x/k) - x/k.
    #    Note h_k(x) is in (-1, 0].
    #    
    #    Constraint: M(x) <= 1 for all x in [1, 10K].
    #    Since h_k(x) <= 0, if we choose u_k >= 0, then M(x) <= 0 <= 1.
    #    So any non-negative u_k is feasible!
    #    
    #    Objective: a = -sum_{k=2}^K u_k log(k)/k.
    #    To maximize a, we want to make -u_k log(k)/k large.
    #    Since log(k)/k > 0 for k > 1, we want u_k to be as negative as possible?
    #    No, a = - sum u_k * (log k / k).
    #    If u_k is negative, -u_k is positive, so a increases.
    #    So we want u_k to be negative.
    #    But if u_k is negative, M(x) = sum u_k h_k(x).
    #    h_k(x) is negative. u_k is negative. Product is positive.
    #    So M(x) will be positive.
    #    We need M(x) <= 1.
    #    
    #    So we want to choose u_k (negative) to maximize -sum u_k log(k)/k
    #    subject to sum u_k h_k(x) <= 1 for all x.
    #    
    #    This is a linear programming problem!
    #    Maximize c^T u
    #    Subject to A u <= 1
    #    
    #    Where c_k = -log(k)/k.
    #    A_{x,k} = h_k(x) = floor(x/k) - x/k.
    #    
    #    We also have clipping: u_k in [-10, 10].
    #    
    #    Since h_k(x) <= 0, and we want u_k negative, let v_k = -u_k >= 0.
    #    Maximize sum v_k log(k)/k
    #    Subject to sum v_k (-h_k(x)) <= 1  => sum v_k (x/k - floor(x/k)) <= 1.
    #    Let w_k(x) = x/k - floor(x/k) = {x/k}. This is in [0, 1).
    #    Constraint: sum_{k=2}^K v_k {x/k} <= 1 for all x.
    #    v_k in [0, 10].
    #    
    #    This is a nice LP.
    #    The constraints are for x = 1, ..., 10K.
    #    Number of constraints: 10K.
    #    Number of variables: K-1.
    #    
    #    We can solve this LP using scipy.optimize.linprog.
    #    
    #    However, K can be up to 200,000.
    #    10K = 2,000,000 constraints.
    #    This is a huge LP.
    #    
    #    But maybe we don't need all x.
    #    The function {x/k} is periodic? No.
    #    But the max of sum v_k {x/k} might occur at specific x.
    #    
    #    Let's try a smaller K first to see if we can improve on the seed.
    #    Seed score is 1.0.
    #    The seed uses a specific set of values.
    #    
    #    Let's implement the LP solver for small K (e.g., K=1000) and see the score.
    #    Then increase K if time permits.

    def solve_lp(K):
        # Variables: v_2, ..., v_K (length K-1)
        # Maximize sum v_k log(k)/k
        # Subject to sum v_k {x/k} <= 1 for x=1..10K
        # v_k in [0, 10]
        
        n_vars = K - 1
        if n_vars <= 0:
            return None, None
        
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks  # Objective coefficients (to maximize)
        
        # Constraints: A v <= b
        # A has shape (10K, K-1)
        # This is too large to build explicitly if K is large.
        # We need a smarter way or a smaller K.
        
        # Let's try K=100 first.
        # 1000 constraints, 99 variables. Easy.
        
        if K > 500:
            # Too large for simple dense LP in Python with scipy if not optimized?
            # Actually, 5000 x 499 is 2.5M floats. 20MB. OK.
            # But solving it might take time.
            pass
            
        x_max = 10 * K
        xs = np.arange(1, x_max + 1)
        
        # Build A
        # A[i, j] corresponds to x = xs[i], k = ks[j]
        # {xs[i] / ks[j]}
        
        # Vectorized construction
        # A = (xs[:, None] / ks[None, :]) % 1
        # This creates a 10K x (K-1) matrix.
        
        A = (xs[:, None] / ks[None, :]) % 1.0
        b = np.ones(x_max)
        
        # Bounds
        bounds = [(0, 10)] * n_vars
        
        # Solve
        from scipy.optimize import linprog
        # linprog minimizes c^T x. We want to maximize. So use -c.
        res = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if res.success:
            v = res.x
            u = -v
            # f(1) is determined: f(1)_eff = -sum u_k/k = sum v_k/k.
            # But we return f(1) such that after adjustment it becomes f(1)_eff.
            # f(1)_eff = f(1) - sum_{k=2} f(k)/k = f(1) - sum_{k=2} u_k/k.
            # We want f(1)_eff = sum v_k/k.
            # So f(1) = sum v_k/k + sum u_k/k = sum v_k/k - sum v_k/k = 0.
            # So we can set f(1) = 0.
            
            f1 = 0.0
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = np.zeros(K, dtype=np.float64)
            values[0] = f1
            values[1:] = u
            
            return domain, values
        else:
            return None, None

    # Try increasing K
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Start with small K
    for K in [10, 50, 100, 200, 500]:
        if time.time() > deadline:
            break
        
        try:
            domain, values = solve_lp(K)
            if domain is not None:
                # Evaluate score
                # Clip values
                v_clipped = np.clip(values, -10, 10)
                
                # Calculate S
                S = np.sum(v_clipped / domain)
                
                # Adjust f(1)
                # If 1 is in domain (it is, at index 0)
                v_clipped[0] = v_clipped[0] - S
                
                # Recalculate M(x) to check feasibility?
                # We trusted the LP, but let's verify for the specific K.
                # And calculate objective a.
                
                # Objective a = -sum f(k) log(k) / k
                # Note: log(1) = 0, so f(1) doesn't contribute.
                a = -np.sum(v_clipped * np.log(domain) / domain)
                
                # Check feasibility quickly
                # M(x) = sum v_clipped[k] floor(x/k)
                # We know from LP construction that sum v_k {x/k} <= 1.
                # M(x) = sum u_k (floor(x/k) - x/k) + f1_eff * x? No.
                # M(x) = sum_{k=2} u_k {x/k} * (-1) + ... wait.
                # My LP constraint was sum v_k {x/k} <= 1.
                # M(x) = sum u_k (floor(x/k) - x/k) = - sum v_k {x/k}.
                # So M(x) = - sum v_k {x/k}.
                # Since sum v_k {x/k} <= 1, M(x) >= -1.
                # We need M(x) <= 1.
                # This is always true if M(x) is not very large positive.
                # Wait, if v_k are positive, {x/k} is positive, sum is positive.
                # M(x) = - (positive). So M(x) is negative.
                # So M(x) <= 1 is satisfied.
                
                # So the LP solution is feasible.
                
                if a > best_score:
                    best_score = a
                    best_domain = domain
                    best_values = v_clipped # Return the clipped/adjusted values?
                    # The evaluator clips and adjusts.
                    # So we should return the ORIGINAL values before clipping/adjustment?
                    # "values are clipped... Then f(1) is replaced..."
                    # So the evaluator does this.
                    # We should return the values we generated.
                    # If we return f(1)=0 and u_k, the evaluator will clip (no change) and adjust f(1).
                    # So we return the pre-adjustment values.
                    best_values = values 
                    
        except Exception as e:
            print(f"Error for K={K}: {e}")
            continue

    # Fallback to seed if no LP solution found
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
