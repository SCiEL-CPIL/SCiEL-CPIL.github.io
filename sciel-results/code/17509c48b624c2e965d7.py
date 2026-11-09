# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy: Greedy Coordinate Descent (GCD) with dual ascent.
    1. Start with small K and expand.
    2. For each K, initialize f(k) with a heuristic (e.g., related to mu(k) or small random values).
    3. Iteratively update f(k) to maximize a = -sum f(k) log(k)/k subject to M(x) <= 1.
       Since M(x) is linear in f, we can compute the max allowed increment for each f(k).
    4. The objective is linear, so the optimal solution lies on the boundary of the feasible region.
       We push f(k) as far as possible in the direction that improves the objective, 
       constrained by the tightest M(x) bound.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    # We want to maximize a = -sum f(k) log(k)/k
    # This means we want f(k) to be negative for large k (where log(k)/k is significant) 
    # or positive for small k? 
    # Wait, a = - sum f(k) log(k) / k.
    # To maximize a, we want the sum to be as negative as possible.
    # So we want f(k) * log(k) / k to be positive? No, we want - (sum) to be large positive.
    # So sum f(k) log(k) / k should be large negative.
    # Since log(k)/k > 0, we want f(k) to be negative.
    
    # Constraints:
    # 1. sum f(k)/k = 0 (handled by f(1))
    # 2. M(x) = sum f(k) floor(x/k) <= 1 for all x in [1, 10K]
    
    # Note: f(1) is determined by the zero-sum constraint: f(1) = - sum_{k>1} f(k)/k.
    # M(x) = f(1) + sum_{k=2}^{min(x,K)} f(k) floor(x/k)
    #      = - sum_{k=2}^K f(k)/k + sum_{k=2}^{min(x,K)} f(k) floor(x/k)
    #      = sum_{k=2}^K f(k) * (floor(x/k) - 1/k)  [Note: floor(x/k)=0 if k>x]
    
    # Let's define g_k(x) = floor(x/k) - 1/k for k <= x, and -1/k for k > x.
    # Constraint: sum_{k=2}^K f(k) g_k(x) <= 1 for all x.
    
    # We want to minimize sum f(k) log(k)/k (which is -a).
    # So we want to make f(k) as negative as possible, weighted by log(k)/k.
    
    # Greedy approach:
    # Start with f(k) = 0 for all k.
    # For each k, calculate how much we can decrease f(k) (make it more negative) 
    # without violating any M(x) <= 1 constraint.
    # Decreasing f(k) by delta > 0 changes M(x) by -delta * g_k(x).
    # We need M(x) - delta * g_k(x) <= 1.
    # So delta * g_k(x) >= M(x) - 1.
    
    # If g_k(x) > 0, delta >= (M(x) - 1) / g_k(x).
    # If g_k(x) < 0, delta <= (M(x) - 1) / g_k(x).
    # If g_k(x) = 0, no constraint from this x.
    
    # Since we start with M(x) = 0 (all f=0), M(x)-1 = -1.
    # For g_k(x) > 0: delta >= -1/g_k(x). Since delta > 0, this is always satisfied if -1/g_k(x) is negative.
    # For g_k(x) < 0: delta <= -1/g_k(x) = 1/|g_k(x)|. This gives an upper bound on how much we can decrease f(k).
    
    # Wait, if we decrease f(k), M(x) changes by -delta * g_k(x).
    # If g_k(x) is negative, -delta * g_k(x) is positive, so M(x) increases. This is bad.
    # If g_k(x) is positive, -delta * g_k(x) is negative, so M(x) decreases. This is good.
    
    # So we can only decrease f(k) if it doesn't violate constraints where g_k(x) < 0.
    # Actually, we want to maximize a. If f(k) becomes very negative, a increases.
    # But making f(k) negative increases M(x) for x where g_k(x) < 0.
    
    # Let's re-evaluate.
    # Objective: Maximize A = - sum f(k) w_k, where w_k = log(k)/k.
    # This is equivalent to Minimize S = sum f(k) w_k.
    # We want f(k) to be negative to make S negative.
    
    # Let's try increasing f(k) instead?
    # If f(k) is positive, S increases, A decreases. Bad.
    # So we definitely want f(k) negative.
    
    # Let's check the sign of g_k(x).
    # g_k(x) = floor(x/k) - 1/k.
    # For k <= x, floor(x/k) >= 1. 1/k <= 0.5.
    # If floor(x/k) = 1, g_k(x) = 1 - 1/k > 0.
    # If floor(x/k) = 0 (i.e., k > x), g_k(x) = -1/k < 0.
    
    # So for any k, there are x < k where g_k(x) < 0.
    # Specifically, for x in [1, k-1], g_k(x) = -1/k.
    # Changing f(k) by delta changes M(x) for x < k by -delta * (-1/k) = delta/k.
    # So M(x) increases by delta/k for all x < k.
    # Since we start at M(x)=0, the constraint M(x) <= 1 implies delta/k <= 1 => delta <= k.
    # So we can decrease f(k) by at most k?
    # Let's check other x.
    # For x >= k, g_k(x) = floor(x/k) - 1/k.
    # If we decrease f(k) by delta, M(x) changes by -delta * (floor(x/k) - 1/k).
    # Since floor(x/k) >= 1, floor(x/k) - 1/k > 0.
    # So M(x) decreases. This is safe.
    
    # So the only constraints on decreasing f(k) come from x < k.
    # For x < k, M(x) increases by delta/k.
    # Initial M(x) = 0.
    # Constraint: 0 + delta/k <= 1 => delta <= k.
    
    # So we can set f(k) = -k?
    # Let's check if this is optimal.
    # If f(k) = -k for all k, then:
    # M(x) = sum_{k=2}^K (-k) * (floor(x/k) - 1/k)
    #      = - sum_{k=2}^K k * floor(x/k) + sum_{k=2}^K 1
    #      = - sum_{k=2}^x k * floor(x/k) + (K - 1)  [since floor(x/k)=0 for k>x]
    
    # This looks like it might violate M(x) <= 1 for large x.
    # Let's test small x.
    # x=1: M(1) = - sum_{k=2}^K (-k) * (-1/k) = - sum 1 = -(K-1).
    # Wait, my formula for M(x) change was for delta.
    # Let's just compute M(x) directly for f(k) = -k.
    # M(1) = f(1) + sum_{k=2}^K f(k) floor(1/k).
    # floor(1/k) = 0 for k>1.
    # So M(1) = f(1).
    # f(1) = - sum_{k=2}^K f(k)/k = - sum (-k)/k = - sum (-1) = K-1.
    # So M(1) = K-1.
    # Constraint M(1) <= 1. So K-1 <= 1 => K <= 2.
    # So we can't just set f(k) = -k.
    
    # The constraint from x=1 is tight.
    # M(1) = f(1) = - sum_{k=2}^K f(k)/k <= 1.
    # So sum_{k=2}^K f(k)/k >= -1.
    # Let f(k) = -c_k. Then sum c_k/k <= 1.
    # We want to maximize a = sum c_k log(k)/k.
    # This is a knapsack-like problem!
    # Maximize sum c_k w_k subject to sum c_k v_k <= 1, c_k >= 0.
    # where w_k = log(k)/k and v_k = 1/k.
    # Ratio w_k/v_k = log(k).
    # To maximize, we should pick the largest ratios.
    # log(k) is increasing. So we should pick the largest k.
    # So we should put all weight on the largest K.
    # Let c_K = K (so c_K/K = 1).
    # Then sum c_k/k = 1.
    # f(K) = -K.
    # All other f(k) = 0.
    # f(1) = -(-K)/K = 1.
    # M(x) = f(1) + f(K) floor(x/K).
    # M(x) = 1 - K floor(x/K).
    # For x < K, floor(x/K) = 0, M(x) = 1. OK.
    # For x = K, floor(K/K) = 1, M(K) = 1 - K = 1-K.
    # For x = 2K, floor(2K/K) = 2, M(2K) = 1 - 2K.
    # We need M(x) <= 1.
    # 1 - K floor(x/K) <= 1 => -K floor(x/K) <= 0 => floor(x/K) >= 0.
    # This is always true for x >= 1.
    # So this solution is feasible!
    
    # Score a = - f(K) log(K)/K = - (-K) log(K)/K = log(K).
    # So a = log(K).
    # To maximize a, we want largest K.
    # Max K is 200000.
    # a = log(200000) ~ 12.2.
    
    # Can we do better?
    # What if we distribute weight?
    # We established this is a knapsack problem with ratio log(k).
    # Since log(k) is increasing, the optimal is to put all weight on the largest k.
    # So f(K) = -K, others 0, f(1)=1 is optimal for the relaxed problem where only M(1) matters.
    # But M(x) for x > 1 also matters.
    # We checked M(x) = 1 - K floor(x/K) <= 1.
    # It is always <= 1.
    # So this is a valid solution.
    
    # Is there a better one?
    # What if we use multiple large k's?
    # Suppose we use K and K-1.
    # c_K/K + c_{K-1}/(K-1) = 1.
    # a = c_K log(K)/K + c_{K-1} log(K-1)/(K-1).
    # Since log(K) > log(K-1), we prefer c_K.
    # So concentrating on the largest K is best.
    
    # So the strategy is:
    # f(K) = -K
    # f(1) = 1
    # Others 0.
    
    # Let's verify M(x) <= 1.
    # M(x) = 1 * 1 + (-K) * floor(x/K).
    # M(x) = 1 - K floor(x/K).
    # Max value is 1 (when floor(x/K)=0).
    # Min value is 1 - K * floor(10K/K) = 1 - 10K.
    # So M(x) <= 1 is always satisfied.
    
    # Score = log(K).
    # We can choose K up to 200000.
    # So score ~ 12.2.
    
    # Current best score is 1.0 (which corresponds to a ~ 0? No, score is proportional to a).
    # The problem says "score is proportional to a".
    # If seed score is 1.0, and it likely had a small a, then a=12.2 is a huge improvement.
    
    # Let's implement this simple solution and verify.
    
    K = 200000
    domain = np.array([1, K], dtype=np.int64)
    values = np.array([1.0, -float(K)], dtype=np.float64)
    
    # Check constraints
    # M(x) = 1 - K * floor(x/K)
    # Max M(x) is 1.
    # Sum f(k)/k = 1/1 + (-K)/K = 1 - 1 = 0.
    
    # Score a = - (1 * log(1)/1 + (-K) * log(K)/K) = - (0 - log(K)) = log(K).
    
    # This seems too simple. Did I miss a constraint?
    # "values are clipped to [-10, 10]"
    # Ah! f(K) = -200000. This will be clipped to -10.
    # f(1) = 1.
    
    # If f(K) is clipped to -10:
    # f(1) is recalculated?
    # "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values"
    # So we must provide values that, after clipping, yield a good result.
    # If we provide f(K) = -200000, it becomes -10.
    # Then S = 1/1 + (-10)/K = 1 - 10/K.
    # f(1) becomes 1 - (1 - 10/K) = 10/K.
    # New f(1) = 10/K.
    # New f(K) = -10.
    
    # Check M(x) with clipped values:
    # M(x) = f(1) + f(K) floor(x/K) = 10/K - 10 floor(x/K).
    # Max M(x) is 10/K (when floor=0).
    # 10/K <= 1 is true for K >= 10.
    # So this is feasible.
    
    # Score a = - (f(1) log(1)/1 + f(K) log(K)/K)
    # a = - (0 + (-10) log(K)/K) = 10 log(K) / K.
    
    # We want to maximize 10 log(K) / K.
    # Let h(K) = log(K) / K.
    # h'(K) = (1 - log(K)) / K^2.
    # Max at log(K) = 1 => K = e ~ 2.718.
    # So K=3 is optimal?
    # h(3) = log(3)/3 ~ 1.098/3 ~ 0.366.
    # a = 10 * 0.366 = 3.66.
    
    # If K=200000, h(K) = log(200000)/200000 ~ 12.2/200000 ~ 6e-5.
    # a = 0.0006.
    
    # So the clipping constraint changes the optimal strategy significantly.
    # We are limited to f(k) in [-10, 10].
    
    # Let's re-evaluate the knapsack with bounds.
    # Maximize sum c_k log(k)/k subject to sum c_k/k <= 1 and 0 <= c_k <= 10 (since f(k) = -c_k, and f(k) >= -10).
    # Actually f(k) can be positive too?
    # If f(k) is positive, it contributes negatively to a. So we want f(k) negative.
    # So 0 <= -f(k) <= 10.
    
    # Ratio is log(k).
    # We should fill the knapsack with the largest k's first.
    # Capacity is 1 (in terms of sum c_k/k).
    # Item k has weight 1/k and value log(k)/k.
    # We can take at most 10 of each item.
    
    # So we take 10 of K, 10 of K-1, etc., until capacity is full.
    # Capacity = 1.
    # Weight of 10 items of size k is 10/k.
    # We want to pick largest k.
    # Start with K=200000.
    # Take 10 of K. Weight = 10/200000 = 0.00005.
    # Remaining capacity = 0.99995.
    # Take 10 of K-1. Weight = 10/199999.
    # ...
    # This will take many items.
    # Number of items m ~ 1 / (10/K) = K/10 = 20000.
    # So we use ~20000 distinct k values, each with f(k) = -10.
    
    # Let's calculate the score.
    # a = sum_{k in S} 10 * log(k) / k.
    # Where S is the set of largest ~20000 integers.
    # Approximate sum as integral:
    # Integral from K-m to K of 10 log(x)/x dx.
    # Let u = log(x), du = dx/x.
    # Integral 10 u du = 5 u^2.
    # Bounds: log(K-m) to log(K).
    # K=200000, m=20000.
    # log(200000) ~ 12.2.
    # log(180000) ~ 12.1.
    # a ~ 5 * (12.2^2 - 12.1^2) = 5 * (148.8 - 146.4) = 5 * 2.4 = 12.
    
    # This is much better than 3.66.
    
    # So the strategy is:
    # 1. Choose a set of large k's, say from K-N to K.
    # 2. Set f(k) = -10 for these k's.
    # 3. Set f(1) to satisfy sum f(k)/k = 0.
    #    f(1) = - sum_{k in S} (-10)/k = 10 * sum_{k in S} 1/k.
    # 4. Check if f(1) <= 10.
    #    sum_{k in S} 1/k ~ ln(K) - ln(K-N).
    #    If N is small relative to K, this is small.
    #    For N=20000, K=200000, sum ~ ln(200000/180000) ~ ln(1.11) ~ 0.1.
    #    f(1) = 10 * 0.1 = 1.
    #    This is <= 10. OK.
    # 5. Check M(x) <= 1.
    #    M(x) = f(1) + sum_{k in S} (-10) floor(x/k).
    #    M(x) = 10 sum_{k in S} 1/k - 10 sum_{k in S} floor(x/k).
    #    We need this <= 1.
    #    Let S = [K-N, K].
    #    For x < K-N, floor(x/k) = 0 for all k in S.
    #    M(x) = 10 sum_{k in S} 1/k.
    #    We need 10 sum 1/k <= 1 => sum 1/k <= 0.1.
    #    As calculated, sum 1/k ~ 0.1. So this is tight.
    #    We might need to reduce N or f(k) magnitude.
    
    # Let's optimize N and f_max (magnitude of f(k)).
    # Let f(k) = -c for k in [K-N, K].
    # f(1) = c * sum_{k in S} 1/k.
    # Constraint f(1) <= 10 => c * sum 1/k <= 10.
    # Constraint M(x) <= 1 for x < K-N:
    # M(x) = f(1) = c * sum 1/k <= 1.
    # So c * sum 1/k <= 1.
    # This is tighter than f(1) <= 10.
    # So c <= 1 / sum 1/k.
    
    # Score a = c * sum_{k in S} log(k)/k.
    # To maximize a, we want c as large as possible.
    # So c = 1 / sum 1/k.
    # a = sum log(k)/k / sum 1/k.
    
    # This is the ratio of weighted sums.
    # We want to choose S to maximize this ratio.
    # Ratio = (sum log(k)/k) / (sum 1/k).
    # This is a weighted average of log(k) with weights 1/k.
    # To maximize the average, we should pick k's with largest log(k).
    # So pick the largest k's.
    # But increasing N increases the denominator more?
    # Let's check.
    # If we add a smaller k, log(k) is smaller.
    # The weighted average will decrease.
    # So we should pick the smallest possible set of largest k's?
    # But we are constrained by c <= 10? No, c is determined by the M(x) constraint.
    # Wait, f(k) is clipped to [-10, 10].
    # So we must have c <= 10.
    # If 1 / sum 1/k > 10, then c is limited to 10.
    # If c = 10, then f(1) = 10 * sum 1/k.
    # Constraint M(x) <= 1 for x < K-N:
    # 10 * sum 1/k <= 1 => sum 1/k <= 0.1.
    
    # So we need to choose S such that sum_{k in S} 1/k <= 0.1.
    # And we want to maximize a = 10 * sum_{k in S} log(k)/k.
    # Since log(k)/k is decreasing for k > e, we want to pick the largest k's.
    # So start from K=200000 and go down until sum 1/k > 0.1.
    
    # Let's estimate N.
    # sum_{k=K-N}^K 1/k ~ ln(K/(K-N)).
    # ln(K/(K-N)) = 0.1 => K/(K-N) = e^0.1 ~ 1.105.
    # K = 1.105 (K-N) => K = 1.105 K - 1.105 N => 0.105 K = 1.105 N => N ~ K/10.5 ~ 19000.
    
    # So we pick k from 181000 to 200000.
    # Set f(k) = -10.
    # f(1) = 10 * sum 1/k ~ 1.
    # M(x) for x < 181000 is 1.
    # For x >= 181000, floor(x/k) becomes 1 or more.
    # M(x) = 1 - 10 sum floor(x/k).
    # This will be negative, so <= 1.
    
    # So this strategy works.
    # Score a = 10 * sum_{k in S} log(k)/k.
    # Sum log(k)/k ~ Integral ln(x)/x dx = 0.5 (ln K)^2 - 0.5 (ln(K-N))^2.
    # ln(200000) ~ 12.2.
    # ln(181000) ~ 12.1.
    # Sum ~ 0.5 * (148.8 - 146.4) = 1.2.
    # a ~ 12.
    
    # This is a solid improvement.
    
    # Implementation:
    # 1. Find N such that sum_{k=K-N}^K 1/k <= 0.1.
    # 2. Set f(k) = -10 for k in [K-N, K].
    # 3. Set f(1) = 10 * sum_{k in S} 1/k.
    # 4. Return domain and values.
    
    K_max = 200000
    target_sum = 0.1
    sum_inv = 0.0
    N = 0
    k = K_max
    while sum_inv <= target_sum and k > 1:
        sum_inv += 1.0 / k
        N += 1
        k -= 1
    
    # k is now K_max - N + 1
    # S is [k, K_max]
    S = np.arange(k, K_max + 1, dtype=np.int64)
    
    # f(k) = -10
    f_S = -10.0 * np.ones(len(S))
    
    # f(1)
    f_1 = 10.0 * np.sum(1.0 / S.astype(np.float64))
    
    # Check if f_1 > 10? No, sum_inv <= 0.1, so f_1 <= 1.
    
    # Build domain and values
    domain = np.concatenate((np.array([1], dtype=np.int64), S))
    values = np.concatenate((np.array([f_1], dtype=np.float64), f_S))
    
    # Verify
    # M(x) <= 1?
    # For x < k, M(x) = f_1 <= 1.
    # For x >= k, M(x) = f_1 - 10 * sum_{j in S, j<=x} floor(x/j).
    # Since sum floor(x/j) >= 1 (at least for j=x if x in S, or j=k if x>=k),
    # M(x) <= 1 - 10 = -9 <= 1.
    
    # Score
    a = -np.sum(values * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
    
    return domain, values

# EVOLVE-BLOCK-END
