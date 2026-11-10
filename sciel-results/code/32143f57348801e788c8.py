# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_mobius(n):
    """Compute Möbius function mu(k) for k = 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
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

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    Strategy: Greedy Coordinate Descent on a sparse support of square-free integers.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    # Parameters
    K = 200000
    max_x = 10 * K
    limit = 1.0001
    
    # 1. Compute Möbius function
    mu = compute_mobius(K)
    
    # 2. Select Support Set
    # We want to maximize -sum f(k) log(k) / k.
    # Since log(k)/k > 0, we want f(k) to be as negative as possible for large k.
    # Constraint M(x) = sum f(k) floor(x/k) <= 1.
    # To keep M(x) low (negative), we prefer f(k) < 0.
    # For square-free k, mu(k) is +/- 1.
    # If we pick k with mu(k) = -1, then setting f(k) < 0 is "natural" in some sense?
    # Actually, let's just pick a large set of square-free numbers.
    # To save memory and time, we will focus on a subset of indices.
    # Let's use the first 50,000 square-free numbers.
    
    square_free_indices = np.where(mu != 0)[0]
    # We need at least 1 element.
    if len(square_free_indices) == 0:
         return np.array([1], dtype=np.int64), np.array([0.0], dtype=np.float64)
    
    # Take a manageable subset to allow for fast iterative updates.
    # 50k variables is reasonable for numpy vectorized ops.
    max_vars = 50000
    if len(square_free_indices) > max_vars:
        # Prioritize larger k? Or just random? 
        # Larger k has larger log(k)/k initially? No, log(k)/k decreases.
        # But larger k affects fewer x in M(x).
        # Let's just take the first max_vars square-free numbers.
        support_indices = square_free_indices[:max_vars]
    else:
        support_indices = square_free_indices
        
    m = len(support_indices)
    
    # 3. Precompute Coefficients for M(x)
    # M(x) = sum_{i=0}^{m-1} f(i) * floor(x / k_i)
    # We need to check M(x) <= limit for all x in [1, 10K].
    # Checking all 10K x 50k is 500M ops. Too slow for iterative.
    # Strategy: Check a subset of x, or use a fast approximation.
    # Actually, M(x) is a step function. The maximum of M(x) likely occurs at jumps or boundaries.
    # However, we can just check all x if we vectorize the update of M(x) efficiently?
    # No, updating M(x) for all x when one f(i) changes requires adding f(i)*floor(x/k_i) to all x.
    # That's O(10K) per update. If we do 1000 updates, that's 10^7 ops. Feasible.
    
    # Let's initialize f to 0.
    f = np.zeros(m, dtype=np.float64)
    
    # Precompute floor(x / k_i) for all i and all x?
    # Memory: 50k * 2M floats = 100GB. Too much.
    # We must compute floor(x/k) on the fly or in chunks.
    
    # Alternative:
    # We can maintain the current M(x) array.
    # M_current = np.zeros(max_x + 1)
    # When we change f[i] by delta, we add delta * floor(x/k_i) to M_current.
    # floor(x/k_i) can be computed quickly.
    
    M_current = np.zeros(max_x + 1, dtype=np.float64)
    
    # Objective to maximize: A = -sum f(i) * log(k_i) / k_i
    # Let w_i = -log(k_i) / k_i. Note w_i > 0.
    # Maximize sum f(i) * w_i.
    # Wait, the metric is a = -sum f(k) log(k) / k.
    # So we want to maximize this value.
    # If f(k) is negative, -f(k) is positive, so a increases.
    # So we want f(k) to be as negative as possible.
    
    k_vals = support_indices + 1
    weights = -np.log(k_vals.astype(np.float64)) / k_vals.astype(np.float64)
    
    # Greedy Loop
    # We will iterate through variables, trying to decrease f(i) (make it more negative)
    # as much as possible without violating M(x) <= limit.
    
    # To speed up, we can process in batches or just iterate.
    # Let's do a few passes.
    
    num_passes = 10
    if m > 20000:
        num_passes = 5
        
    for _ in range(num_passes):
        if time.time() > deadline - 60:
            break
            
        # Shuffle indices to avoid bias
        order = np.random.permutation(m)
        
        for idx in order:
            if time.time() > deadline - 60:
                break
                
            k = k_vals[idx]
            w = weights[idx]
            
            # Current f[idx]
            current_f = f[idx]
            
            # We want to decrease f[idx] (increase negativity).
            # Let f_new = current_f - step.
            # Change in M(x) is -step * floor(x/k).
            # We need M_current(x) - step * floor(x/k) <= limit for all x.
            # This implies step * floor(x/k) >= M_current(x) - limit for all x where floor(x/k) > 0.
            # If M_current(x) <= limit, then M_current(x) - limit <= 0.
            # Since step > 0 and floor(x/k) >= 0, the condition is automatically satisfied for those x?
            # No, if M_current(x) is already close to limit, we can't decrease f further.
            # Specifically, for any x where floor(x/k) > 0:
            # M_new(x) = M_current(x) - step * floor(x/k)
            # We require M_new(x) <= limit.
            # M_current(x) - step * floor(x/k) <= limit
            # step * floor(x/k) >= M_current(x) - limit
            
            # Let R(x) = M_current(x) - limit.
            # If R(x) <= 0, the constraint is satisfied for any step >= 0.
            # If R(x) > 0, we need step >= R(x) / floor(x/k).
            # So the maximum step we can take is determined by the MINIMUM of R(x)/floor(x/k) over all x where R(x)>0?
            # No, we are DECREASING f.
            # Wait.
            # Objective: Maximize sum f(i) w_i.
            # w_i > 0.
            # If we DECREASE f(i), the objective DECREASES.
            # We want to INCREASE the objective.
            # So we want to INCREASE f(i) (make it less negative or positive).
            
            # Let's re-read the objective.
            # a = -sum f(k) log(k) / k.
            # If f(k) is negative, -f(k) is positive.
            # So we want f(k) to be NEGATIVE.
            # To make a LARGER, we need f(k) to be MORE NEGATIVE.
            # So we want to DECREASE f(k).
            
            # Constraint: M(x) <= limit.
            # M(x) = sum f(j) floor(x/k_j).
            # If we decrease f(i) by step (step > 0), M(x) decreases by step * floor(x/k_i).
            # Since M(x) decreases, it becomes EASIER to satisfy M(x) <= limit.
            # So if the current state is feasible, decreasing f(i) keeps it feasible!
            
            # Wait, is that right?
            # If M(x) <= 1, and we subtract a positive amount, M(x) becomes smaller (more negative).
            # So it is still <= 1.
            # So we can decrease f(i) indefinitely?
            # No, because of the f(1) adjustment and the clipping?
            # The problem states:
            # 1. values are clipped to [-10, 10].
            # 2. f(1) is adjusted so that sum f(k)/k = 0.
            # 3. M(x) is computed.
            
            # If we set f(i) = -10 for all i, then M(x) is very negative.
            # But f(1) must be adjusted.
            # S = sum f(k)/k. If all f(k) = -10, S is negative.
            # f(1) = f(1) - S. If f(1) was -10, f(1) becomes -10 - (negative) = -10 + |S|.
            # This might push f(1) out of [-10, 10] or change M(x) significantly.
            
            # Also, we are maximizing a = -sum f(k) log(k)/k.
            # If we set all f(k) = -10, a is large positive.
            # Is there a constraint preventing this?
            # "if any M(x) > 1.0001 the function is infeasible".
            # If M(x) is very negative, it is feasible.
            # So why isn't the answer just all -10?
            
            # Let's check the f(1) adjustment again.
            # f(1) = 0 initially if 1 not in domain.
            # If 1 IS in domain, f(1) is updated.
            # If we include 1 in the support, we must handle it.
            # If we EXCLUDE 1 from the support, f(1) is 0.
            # Then S = sum_{k>1} f(k)/k.
            # f(1) is not in the domain, so it doesn't contribute to M(x) directly?
            # M(x) = sum_{k in domain} f(k) floor(x/k).
            # If 1 is not in domain, f(1) is effectively 0 in the sum.
            # But the adjustment says "f(1) is replaced by f(1) - S".
            # If 1 is not in domain, does it enter the domain?
            # "f(1) = 0 before this if 1 is not in the domain".
            # This implies f(1) is treated as a value in the calculation of S, and then potentially added to the domain?
            # "domain holds distinct positive integers".
            # If we start with 1 not in domain, and the adjustment creates a non-zero f(1), does 1 become part of the domain for the M(x) calculation?
            # "M(x) = sum_k f(k) * floor(x/k)". The sum is over k.
            # Usually "partial function" implies the domain is fixed.
            # But the text says "f(1) is replaced... so that afterwards sum_k f(k)/k = 0".
            # And "M(x) is computed... for every integer x".
            # If 1 was not in the domain, f(1) was 0.
            # After adjustment, f(1) is non-zero.
            # Does the sum for M(x) include k=1?
            # "sum_k f(k)". If f is a partial function, the sum is over the domain.
            # If the domain doesn't change, and 1 wasn't in it, then f(1) is not in the sum?
            # But the adjustment formula suggests f(1) is a value in the system.
            # Let's assume the domain is dynamic or 1 is always included if non-zero.
            # To be safe, let's INCLUDE 1 in the support.
            
            # If we include 1, we can't just set all to -10.
            # Because f(1) will be adjusted.
            # Let's stick to the GCD idea but be careful.
            
            # Actually, if we can set f(k) = -10 for all k > 1, and f(1) is adjusted,
            # M(x) = f(1)*x + sum_{k>1} f(k) floor(x/k).
            # If f(k) = -10, the sum is very negative.
            # f(1) = 0 - S = -S.
            # S = f(1)_initial(0)/1 + sum_{k>1} -10/k = -10 * H_K.
            # So f(1) = 10 * H_K.
            # H_200000 is approx log(200000) + gamma = 12 + 0.57 = 12.5.
            # f(1) = 125.
            # But values are clipped to [-10, 10]!
            # So f(1) becomes 10.
            # Then S is recalculated? No, "f(1) is replaced... so that afterwards sum = 0".
            # This implies the adjustment is done on the raw values, then clipping?
            # "Evaluation: values are clipped to [-10, 10]. Then f(1) is replaced..."
            # Order:
            # 1. Clip values to [-10, 10].
            # 2. Calculate S = sum f(k)/k (using clipped values).
            # 3. f(1) = f(1) - S.
            # 4. Check if sum f(k)/k = 0?
            #    If f(1) is changed, the sum changes by -S.
            #    New sum = Old Sum - S = S - S = 0.
            #    So the normalization holds.
            # 5. BUT, is f(1) clipped again?
            #    "values are clipped... Then f(1) is replaced".
            #    It does not say f(1) is clipped again.
            #    So f(1) can be outside [-10, 10].
            
            # If f(1) can be large, then M(x) has a term f(1)*x.
            # If f(1) is positive (which it will be if S is negative), M(x) grows linearly.
            # M(x) > 1 will happen for large x.
            # So we CANNOT have S negative.
            # We need S >= 0 so that f(1) = -S <= 0.
            # If f(1) <= 0, then M(x) = f(1)*x + ...
            # The linear term is negative, helping feasibility.
            
            # So the strategy is:
            # Choose f(k) for k > 1 such that S = sum_{k>1} f(k)/k is POSITIVE.
            # Then f(1) = -S is negative.
            # This makes M(x) more negative, allowing us to push f(k) for k>1 to be more negative?
            # Wait, if f(k) for k>1 are negative, S is negative.
            # We need S to be positive.
            # So we need some f(k) to be positive.
            
            # Objective: Maximize -sum f(k) log(k)/k.
            # We want f(k) negative for large k (high weight).
            # But we need S > 0.
            # So we need some positive f(k) with small weight (small log(k)/k) to compensate?
            # Or small k?
            # log(k)/k is max at k=e.
            # For k=2, log(2)/2 = 0.34.
            # For k=10, log(10)/10 = 0.23.
            # For k=100, log(100)/100 = 0.046.
            
            # So we should set f(k) = -10 for large k (to maximize objective).
            # And set f(k) = 10 for small k (to make S positive) to allow f(1) to be negative.
            
            # Let's refine the GCD:
            # 1. Start with f(k) = 0.
            # 2. Identify "good" indices (large k) to decrease.
            # 3. Identify "bad" indices (small k) to increase (to keep S positive).
            # 4. Iterate.
            
            # This is getting complex. Let's just implement a simple heuristic:
            # Set f(k) = -10 for k > K_threshold.
            # Set f(k) = 10 for k < K_threshold.
            # Tune K_threshold.
            
            # Let's try this simpler approach first.
            pass # End of thinking, moving to code

    # Simple Heuristic Implementation
    # Try different thresholds for switching from +10 to -10.
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # We must include 1 in the domain for the f(1) adjustment to work as intended?
    # If 1 is not in domain, f(1)=0. S = sum_{k>1} f(k)/k.
    # f(1) becomes -S.
    # If 1 is not in domain, does M(x) include f(1)?
    # "M(x) = sum_k f(k) floor(x/k)".
    # If the domain is fixed and 1 is not in it, then f(1) is not in the sum.
    # But the adjustment "f(1) is replaced" implies it's a value in the function.
    # I will assume 1 MUST be in the domain for the adjustment to affect M(x).
    # If 1 is not in the domain, the adjustment is meaningless for M(x).
    # So I will always include 1.
    
    # Support: All integers 1..K?
    # No, that's 200k variables.
    # Let's just use all integers 1..K.
    
    # We will test a few configurations.
    # Config 1: f(k) = -10 for all k > 1, f(1) adjusted.
    # Config 2: f(k) = -10 for k > T, f(k) = 10 for k <= T.
    
    # Let's try Config 2 with various T.
    
    for T in [10, 100, 1000, 5000, 10000]:
        if time.time() > deadline - 30:
            break
            
        # Construct f
        # f is array of length K
        f = np.zeros(K, dtype=np.float64)
        
        # k = 1..K
        # Indices 0..K-1 correspond to k=1..K
        
        # For k <= T (indices 0..T-1), set f = 10
        # For k > T (indices T..K-1), set f = -10
        
        if T > 0:
            f[:T] = 10.0
        if T < K:
            f[T:] = -10.0
            
        # Clip to [-10, 10] (already done)
        
        # Calculate S
        k_arr = np.arange(1, K+1, dtype=np.float64)
        S = np.sum(f / k_arr)
        
        # Adjust f(1)
        f[0] = f[0] - S
        
        # Check feasibility
        # M(x) = sum f(k) floor(x/k)
        # We need M(x) <= 1.0001 for all x in [1, 10K].
        
        # Fast check:
        # M(x) is dominated by f(1)*x if f(1) is large.
        # f(1) = 10 - S (if T>=1) or -S (if T=0).
        # If T is small, S is approx 10*H_T - 10*H_K.
        # H_K is large. So S is negative.
        # f(1) = 10 - (negative) = 10 + |S|. Large positive.
        # Then M(x) ~ f(1)*x.
        # M(10K) ~ f(1)*10K.
        # If f(1) > 0, this will be > 1.
        
        # So we need S to be POSITIVE and large enough to make f(1) negative?
        # f(1) = 10 - S.
        # If S > 10, f(1) < 0.
        # S = sum_{k=1}^T 10/k + sum_{k=T+1}^K -10/k
        # S = 10 (H_T - H_K).
        # H_T - H_K is negative.
        # So S is always negative.
        # So f(1) is always > 10.
        # So M(x) will be huge positive.
        
        # Conclusion: We cannot have f(k) = -10 for large k if we want S > 0.
        # We need MORE positive mass than negative mass in 1/k.
        
        # So we should set f(k) = 10 for a LARGE range, and f(k) = -10 for a SMALL range?
        # But we want to maximize -sum f(k) log(k)/k.
        # We want f(k) negative for large k.
        # Contradiction.
        
        # Unless... we don't include 1 in the domain?
        # If 1 is NOT in the domain, f(1) is not in M(x).
        # Then we can set f(k) = -10 for all k.
        # S = sum -10/k = negative.
        # f(1) = 0 - S = positive.
        # But f(1) is not in the domain, so it doesn't affect M(x).
        # Does the adjustment add 1 to the domain?
        # "f(1) is replaced...".
        # If the domain is fixed, and 1 is not in it, then f(1) is just a dummy value?
        # "domain holds distinct positive integers".
        # If I return domain without 1, and values without f(1), then the evaluator:
        # 1. Clips values.
        # 2. Calculates S = sum f(k)/k over the domain.
        # 3. "f(1) is replaced by f(1) - S".
        #    If 1 is not in the domain, f(1) is 0.
        #    So f(1) becomes -S.
        # 4. "so that afterwards sum_k f(k)/k = 0".
        #    This sum is over the domain? Or over all k including 1?
        #    If the domain doesn't change, the sum is over the domain.
        #    The sum over the domain is S.
        #    The adjustment changes f(1). If 1 is not in the domain, the sum over the domain is UNCHANGED.
        #    So the condition "sum_k f(k)/k = 0" would FAIL if S != 0.
        #    Unless the evaluator adds 1 to the domain?
        
        # Given the ambiguity, the safest bet is to INCLUDE 1 in the domain.
        
        # So we must have S > 10 to make f(1) negative.
        # S = 10 * (Sum_{k in Pos} 1/k - Sum_{k in Neg} 1/k).
        # We need Sum_{k in Pos} 1/k > Sum_{k in Neg} 1/k + 1.
        
        # To maximize objective, we want Neg to be large k.
        # To satisfy constraint, we want Pos to be small k (high 1/k).
        
        # Let's try:
        # Pos: k = 1..M
        # Neg: k = M+1..K
        # S = 10 (H_M - (H_K - H_M)) = 10 (2 H_M - H_K).
        # We need 2 H_M - H_K > 1.
        # H_K ~ log K.
        # H_M ~ log M.
        # 2 log M - log K > 1 => log(M^2/K) > 1 => M^2/K > e => M > sqrt(eK).
        
        # If K=200000, M > sqrt(543000) ~ 737.
        
        # So if we set M = 1000, S should be positive and large.
        # Let's test M=1000.
        
        M_val = 1000
        f = np.zeros(K, dtype=np.float64)
        f[:M_val] = 10.0
        f[M_val:] = -10.0
        
        S = np.sum(f / k_arr)
        f[0] = f[0] - S
        
        # Check M(x)
        # M(x) = f(1)*x + sum_{k=2}^K f(k) floor(x/k)
        # f(1) is negative.
        # The sum is negative.
        # So M(x) should be negative.
        
        # Calculate Score
        score = -np.sum(f * np.log(k_arr) / k_arr)
        
        # Verify Feasibility (Sampled)
        # Check x = 1, 10, 100, 1000, 10000, 100000, 2000000
        test_x = [1, 10, 100, 1000, 10000, 100000, 2000000]
        feasible = True
        for x in test_x:
            if x > max_x: continue
            # Compute M(x)
            # M(x) = sum f(k) floor(x/k)
            # Vectorized
            kk = np.arange(1, K+1)
            floors = x // kk
            M_x = np.sum(f * floors)
            if M_x > 1.0001:
                feasible = False
                break
                
        if feasible and score > best_score:
            best_score = score
            best_domain = np.arange(1, K+1, dtype=np.int64)
            best_values = f.copy()
            
    if best_domain is None:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    return best_domain, best_values

# EVOLVE-BLOCK-END
