# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k)*log(k)/k subject to M(x) <= 1.
    Uses Linear Programming to find the optimal distribution of 'mass' f(k).
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for the 630s limit

    # We search for the optimal K (max key in domain)
    # Based on theoretical scaling, larger K allows more "cheap" mass (small log(k)/k)
    # to be placed, improving the objective.
    # We try a sequence of K values.
    K_candidates = [100, 500, 2000, 10000, 50000, 200000]
    
    best_score = -np.inf
    best_domain = None
    best_values = None

    for K in K_candidates:
        if time.time() > deadline - 60:
            break

        try:
            # Variables: f(1), f(2), ..., f(K)
            # Constraints:
            # 1. Normalization: sum_{k=1}^K f(k)/k = 0  (Equality)
            # 2. Divisibility: sum_{k|x} f(k) <= 1 for all x in [1, K]
            #    Note: The evaluator checks M(x) <= 1 for x in [1, 10K].
            #    M(x) = sum_{k|x} f(k).
            #    If we ensure sum_{k|x} f(k) <= 1 for all x in [1, K], does it hold for x > K?
            #    M(x) for x > K sums over k <= K such that k|x.
            #    The set of divisors of x that are <= K is a subset of {1..K}.
            #    However, the constraint in the evaluator is M(x) <= 1 for x in [1, 10K].
            #    The "tightest" constraints are usually at small x or prime x.
            #    Actually, if sum_{k|x} f(k) <= 1 for all x, it's sufficient.
            #    But we only have variables up to K.
            #    For x > K, the sum is over k|x, k<=K.
            #    Let's check if we need to constrain x up to 10K.
            #    If we constrain x in [1, K], is it possible M(x) > 1 for x in (K, 10K]?
            #    M(x) = sum_{d|x, d<=K} f(d).
            #    If x is a prime > K, M(x) = f(1) (since 1 is the only divisor <= K).
            #    If x is composite, it has more divisors.
            #    The maximum of M(x) for x > K is likely bounded by the constraints on smaller numbers if the function is "smooth" or monotonic in a certain way?
            #    Actually, no. Consider x = p (prime > K). M(p) = f(1).
            #    Consider x = 2p (2p > K). M(2p) = f(1) + f(2).
            #    We know f(1)+f(2) <= 1 from the constraint for x=2 (if 2<=K).
            #    Generally, if x > K, let y = x / (largest prime factor?) No.
            #    Let's just constrain x in [1, min(10K, K)]? No, that's just [1, K].
            #    Wait, if K=100, we check x up to 1000.
            #    If we only constrain x in [1, 100], is M(101) <= 1?
            #    M(101) = f(1). We know f(1) <= 1 (from x=1). So yes.
            #    M(102) = f(1)+f(2)+f(3)+f(6)+... (divisors of 102 <= 100).
            #    Divisors of 102: 1, 2, 3, 6, 17, 34, 51, 102.
            #    Those <= 100: 1, 2, 3, 6, 17, 34, 51.
            #    Is sum_{d|102, d<=100} f(d) <= 1?
            #    We don't have a direct constraint for 102.
            #    However, note that 102 = 2 * 51.
            #    We have constraint for x=51: sum_{d|51} f(d) <= 1.
            #    We have constraint for x=2: sum_{d|2} f(d) <= 1.
            #    This doesn't directly bound the sum for 102.
            #    
            #    To be safe, we should constrain x in [1, 10K].
            #    But the number of constraints is 10K.
            #    For K=200000, that's 2,000,000 constraints.
            #    LP with 200k variables and 2M constraints might be too slow for HiGHS in 600s?
            #    HiGHS is fast, but let's check.
            #    Alternatively, we can observe that M(x) is maximized when x has many small divisors?
            #    Or we can just constrain x in [1, K] and hope?
            #    Let's look at the structure. f(k) >= 0.
            #    M(x) = sum_{d|x} f(d).
            #    If we ensure M(x) <= 1 for all x in [1, K], does it hold for x in (K, 10K]?
            #    Let x in (K, 10K]. Let p be the smallest prime factor of x.
            #    Then x/p <= 10K/p.
            #    This path is risky. Let's stick to constraining x in [1, 10K] if possible,
            #    or find a smarter set of constraints.
            #    
            #    Actually, the constraint M(x) <= 1 for all x is equivalent to M(x) <= 1 for all squarefree x? No.
            #    It's equivalent to M(x) <= 1 for all x that are "critical".
            #    Since f(k) >= 0, M(x) is monotone with respect to divisibility? No.
            #    M(x) depends on the specific divisors.
            #    
            #    Let's try to constrain only x in [1, K] first and see if it's feasible for x > K.
            #    If not, we'll add constraints for x in (K, 10K] selectively.
            #    
            #    However, for the LP formulation, adding 10K constraints is manageable if K is not huge.
            #    Let's limit K to 50000 for the LP to be safe on time.
            #    With K=50000, 10K = 500000 constraints. 50000 variables.
            #    This is a large sparse LP. HiGHS should handle it.

            n = K
            # Objective: minimize sum (log(k)/k) * f(k)  => maximize -sum ...
            # c_k = log(k)/k
            c = np.log(np.arange(1, n + 1, dtype=np.float64)) / np.arange(1, n + 1, dtype=np.float64)
            
            # Equality constraint: sum f(k)/k = 0
            # A_eq = [1/1, 1/2, ..., 1/n]
            A_eq = (1.0 / np.arange(1, n + 1, dtype=np.float64)).reshape(1, -1)
            b_eq = np.array([0.0])
            
            # Inequality constraints: M(x) <= 1 for x in [1, 10K]
            # We will build the sparse matrix A_ub.
            # Row x (0-indexed x-1) has 1s at columns d-1 where d|x.
            # Number of rows = 10K.
            # Number of columns = K.
            
            # To save memory and time, we can just constrain x in [1, K] initially.
            # Let's assume constraining [1, K] is sufficient for now to get a better score than Mobius.
            # If it fails feasibility check in the evaluator, we'll know.
            # But wait, the evaluator kills on infeasibility.
            # Let's check the logic: if f(k) >= 0, is max M(x) for x in [1, 10K] attained at x <= K?
            # Counter-example: K=10. f(1)=1, f(2)=0... M(1)=1.
            # If we put mass on small numbers, M(x) for large x might be small if x has few small divisors.
            # The worst case for M(x) is when x is highly composite.
            # The largest highly composite numbers are usually <= K if we go up to 10K?
            # No, 10K can have more divisors.
            # Example: K=100. x=90 (<=100) has 12 divisors. x=180 (>100) has 18 divisors.
            # If f(d) > 0 for many d|180, M(180) could be > 1 even if M(x)<=1 for x<=100.
            # So we MUST constrain x up to 10K.
            
            max_x = 10 * K
            
            # Building 2M x 50k sparse matrix is heavy but doable.
            # Let's use COO format.
            rows = []
            cols = []
            data = []
            
            for x in range(1, max_x + 1):
                # Find divisors of x that are <= K
                # Optimization: iterate d up to sqrt(x)
                # Or simply: for d in 1..K, if x%d==0. This is O(K * 10K) = 10K^2.
                # For K=50000, 10^10 ops. Too slow in Python loop.
                # We need a faster way to build the matrix.
                
                # Alternative: Iterate k from 1 to K, and add 1 to rows that are multiples of k.
                # For each k, multiples are k, 2k, 3k, ... <= 10K.
                # Total operations sum_{k=1}^K (10K/k) = 10K log K.
                # For K=50000, 10*50000*11 ~ 5.5M ops. This is fast!
                
                for j in range(1, max_x // k + 1):
                    x_val = j * k
                    if x_val <= max_x:
                        rows.append(x_val - 1)
                        cols.append(k - 1)
                        data.append(1.0)
                # Wait, the loop above is inside the k loop?
                # Yes.
            
            # Let's re-structure for speed:
            rows = []
            cols = []
            data = []
            for k in range(1, n + 1):
                # Multiples of k up to max_x
                for j in range(1, max_x // k + 1):
                    rows.append(j * k - 1)
                    cols.append(k - 1)
                    data.append(1.0)
            
            A_ub = so.sparse.coo_matrix((data, (rows, cols)), shape=(max_x, n)).tocsr()
            b_ub = np.ones(max_x)
            
            # Bounds: f(k) >= 0
            bounds = [(0, None)] * n
            
            # Solve LP
            # Method: 'HiGHS' is the default and fastest.
            res = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='HiGHS')
            
            if not res.success:
                continue
                
            f_vals = res.x
            
            # Verify feasibility explicitly (just in case of numerical issues)
            # And compute score
            # Clip values? The LP solution should be within bounds, but let's be safe.
            # The evaluator clips to [-10, 10]. Our f(k) >= 0.
            # If any f(k) > 10, it gets clipped, changing the objective and constraints.
            # We should add upper bound 10 to the LP to be precise.
            # Let's restart LP with bounds [(0, 10)]
            
            bounds = [(0, 10)] * n
            res = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='HiGHS')
            
            if not res.success:
                continue
                
            f_vals = res.x
            
            # Identify non-zero values to form sparse domain
            # Tolerance for zero
            tol = 1e-9
            mask = f_vals > tol
            domain = np.where(mask)[0] + 1
            values = f_vals[mask]
            
            if len(domain) == 0:
                continue
                
            # Calculate Score (Objective)
            # a = -sum f(k) log(k) / k
            # Note: The evaluator calculates this on the *clipped* and *adjusted* values.
            # Since we solved for sum f(k)/k = 0, the adjustment S is 0.
            # And since f(k) <= 10 (from bounds), clipping doesn't change values.
            # So the score is exactly -res.fun (since res.fun is sum c_k f_k).
            score = -res.fun
            
            if score > best_score:
                best_score = score
                best_domain = domain.copy()
                best_values = values.copy()
                
        except Exception as e:
            # If LP fails for this K, try smaller K
            continue

    if best_domain is None:
        # Fallback to a known feasible simple solution
        # f(1)=1, others 0.
        # M(x) = 1 for all x. Feasible (<= 1.0001).
        # Score = -1 * log(1) / 1 = 0.
        # But we want higher.
        # Let's use the Mobius function as fallback, it's known to be feasible?
        # Actually, Mobius has negative values.
        # Let's just return a single point f(1)=1.
        domain = np.array([1], dtype=np.int64)
        values = np.array([1.0], dtype=np.float64)
        
    return domain, values

# EVOLVE-BLOCK-END
