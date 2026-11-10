# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def prime_number_theorem_partial_function():
    """
    Uses the Möbius function to construct a feasible solution.
    f(k) = mu(k) for k=2..K, f(1) adjusted to satisfy sum f(k)/k = 0.
    M(x) = 1 for all x >= 1 (if K large enough relative to x, but here x <= 10K).
    Actually M(x) = sum_{k=1}^K mu(k) floor(x/k).
    For x <= K, M(x) = 1 if x is a square? No.
    sum_{k=1}^x mu(k) floor(x/k) = 1 for all x >= 1.
    This is a standard identity.
    So if we set f(k) = mu(k) for all k in domain, and domain covers 1..K,
    then M(x) = 1 for all x <= K.
    For x > K, M(x) = sum_{k=1}^K mu(k) floor(x/k).
    This is not necessarily 1. It is the count of square-free integers <= x with some weight?
    Actually, the identity sum_{d|n} mu(d) = 1 if n=1 else 0 implies:
    sum_{k=1}^x mu(k) floor(x/k) = 1 for all x >= 1.
    Wait, this identity holds for the infinite sum?
    No, the sum is finite here up to K.
    If K >= x, then floor(x/k) = 0 for k > x. So the sum is over k=1..x.
    And sum_{k=1}^x mu(k) floor(x/k) = 1 is a known identity.
    So if our domain is {1, ..., K}, and we evaluate M(x) for x in [1, 10K],
    then for x <= K, M(x) = 1.
    For x > K, M(x) = sum_{k=1}^K mu(k) floor(x/k).
    We need M(x) <= 1.0001.
    Is M(x) <= 1 for x > K?
    M(x) is the summatory function of mu.
    M(x) = sum_{n=1}^x mu(n).
    Wait, M(x) in the problem is sum_k f(k) floor(x/k).
    If f(k) = mu(k), then M_problem(x) = sum_{k=1}^K mu(k) floor(x/k).
    We know that sum_{k=1}^infty mu(k) floor(x/k) = 1.
    So M_problem(x) = 1 - sum_{k=K+1}^infty mu(k) floor(x/k).
    For x <= 10K, the tail sum is small?
    Actually, the error term in the Prime Number Theorem relates to M(x).
    But here we have a finite sum.
    Let's check the constraint.
    If we use mu(k), the objective a = -sum mu(k) log(k)/k.
    sum mu(k) log(k)/k is related to the derivative of zeta or something?
    sum_{k=1}^infty mu(k)/k = 0.
    sum_{k=1}^infty mu(k) log(k)/k = ?
    It is known that sum mu(k) log(k)/k converges to a specific value?
    Actually, sum mu(k)/k^s = 1/zeta(s).
    Derivative at s=1?
    d/ds (1/zeta(s)) at s=1.
    1/zeta(s) = product (1 - p^{-s}).
    Log(1/zeta(s)) = sum log(1-p^{-s}) = -sum sum p^{-ns}/n.
    d/ds Log(1/zeta(s)) = sum sum p^{-ns} log(p) / n.
    At s=1, this diverges?
    No, 1/zeta(s) has a zero at s=1? No, zeta(s) has a pole, so 1/zeta(s) has a zero.
    So 1/zeta(s) ~ s-1.
    So d/ds (1/zeta(s)) at s=1 is 1?
    So sum mu(k) log(k)/k = -1?
    Let's check numerically for small K.
    
    The key is: does M(x) <= 1.0001 hold for x up to 10K?
    M_problem(x) = 1 - tail.
    If tail > 0, then M < 1. If tail < 0, then M > 1.
    We need M <= 1.0001.
    So we need tail >= -0.0001.
    tail = sum_{k=K+1}^infty mu(k) floor(x/k).
    For x <= 10K, floor(x/k) is non-zero only for k <= x.
    So tail = sum_{k=K+1}^x mu(k) floor(x/k).
    The magnitude of this sum is roughly O(x / log(x)) or similar?
    Actually, the error term for the PNT is M(x) = o(x).
    But here the "M(x)" is different.
    Let's just try it.
    If it fails, we can scale down or adjust.
    But scaling down reduces the score.
    The score is proportional to a.
    a = -sum f(k) log(k)/k.
    If we use mu(k), a is fixed (for large K).
    If we scale f(k) by c, a scales by c.
    But M(x) scales by c.
    If M(x) <= 1.0001, we can set c=1.
    If M(x) > 1.0001, we must scale down, reducing score.
    
    Let's check if M(x) > 1 for x > K with mu(k).
    The identity sum_{k=1}^x mu(k) floor(x/k) = 1 is exact for the infinite sum?
    No, it is exact for the sum up to x.
    sum_{k=1}^x mu(k) floor(x/k) = sum_{k=1}^x mu(k) sum_{j=1}^{floor(x/k)} 1
    = sum_{j=1}^x sum_{k=1}^{floor(x/j)} mu(k)
    = sum_{j=1}^x M(floor(x/j))
    where M(n) = sum_{k=1}^n mu(k).
    This sum is 1.
    So sum_{k=1}^K mu(k) floor(x/k) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
    The term sum_{k=K+1}^x mu(k) floor(x/k) is the error.
    For x=10K, this error might be significant.
    
    However, the task allows K up to 200000.
    We can choose K to be large enough such that the error is small.
    Or we can just use the LP approach which guarantees feasibility.
    The current LP approach scores 1.0.
    The inspiration 1 scores 1.0664.
    The inspiration 1 likely used a better LP or a different objective.
    
    Let's try the Mu function approach with a large K, say 5000.
    And check feasibility.
    If feasible, compute score.
    If not feasible, scale down.
    
    Actually, the LP approach in the current code is slow for large K.
    The Mu approach is O(K log K) to compute mu, and O(K * 10K) to check M(x)?
    No, M(x) check is O(K * 10K) which is 10 * K^2.
    For K=5000, that's 250 million operations. Might be too slow in Python loops.
    But we can vectorize the M(x) check.
    M(x) = sum_k f(k) floor(x/k).
    This is a matrix multiplication?
    A[i, j] = floor((i+1) / (j+1)).
    M = A @ f.
    A is 10K x K.
    For K=5000, A is 50000 x 5000 = 250M elements.
    Memory is 250M * 8 bytes = 2GB. Too much.
    
    We need a smarter way to check M(x) or a smaller K.
    Or use the LP which only solves for the necessary constraints?
    The LP has 10K constraints.
    For K=100, 1000 constraints.
    For K=1000, 10000 constraints.
    
    Let's stick to the LP but optimize it.
    Or use the Mu function with a moderate K and check M(x) efficiently.
    
    How to check M(x) efficiently?
    M(x) = sum_{k=1}^K f(k) floor(x/k).
    We can compute this for all x in [1, 10K] using a loop over k.
    For each k, add f(k) * floor(x/k) to M(x) for all x.
    floor(x/k) is a step function.
    We can update M(x) in O(10K/k) time for each k?
    Total time sum_{k=1}^K 10K/k = 10K log K.
    For K=10000, 10*10000*10 = 1e6. Very fast.
    
    So:
    1. Compute mu(k) for k=1..K.
    2. Set f(k) = mu(k) for k=2..K, f(1)=0 initially.
    3. Compute S = sum f(k)/k.
    4. f(1) = -S.
    5. Check if f(1) is in [-10, 10]. (S is small, so yes).
    6. Compute M(x) for x=1..10K using the O(10K log K) method.
    7. Check max M(x) <= 1.0001.
    8. If feasible, compute score.
    9. If not, scale down f(k) for k>1 by factor alpha.
       Then re-compute S, f(1), M(x).
       Actually, if we scale f(k) for k>1 by alpha, and f(1) adjusts,
       M(x) = alpha * M_mu(x) + f(1) * 1 (since floor(x/1)=x? No, floor(x/1)=x).
       Wait, f(1) term in M(x) is f(1) * floor(x/1) = f(1) * x.
       This is large!
       The constraint sum f(k)/k = 0 means f(1) = -sum_{k>1} f(k)/k.
       M(x) = f(1)*x + sum_{k>1} f(k) floor(x/k).
       = -x * sum_{k>1} f(k)/k + sum_{k>1} f(k) floor(x/k).
       = sum_{k>1} f(k) (floor(x/k) - x/k).
       = - sum_{k>1} f(k) {x/k}.
       So M(x) = - sum_{k>1} f(k) {x/k}.
       We need M(x) <= 1.0001.
       So - sum f(k) {x/k} <= 1.0001.
       sum f(k) {x/k} >= -1.0001.
       
       If f(k) = mu(k), then M(x) = - sum mu(k) {x/k}.
       We know sum mu(k) floor(x/k) = 1 - tail.
       sum mu(k) (x/k - {x/k}) = 1 - tail.
       x sum mu(k)/k - sum mu(k) {x/k} = 1 - tail.
       M(x) = - sum mu(k) {x/k} = x sum mu(k)/k - (1 - tail).
       sum mu(k)/k is close to 0.
       So M(x) is close to -1 + tail.
       We need M(x) <= 1.0001.
       -1 + tail <= 1.0001 => tail <= 2.0001.
       This is always true since tail is small.
       We also need M(x) to be not too negative? No, only upper bound.
       But wait, the constraint is M(x) <= 1.0001.
       Is there a lower bound? No.
       But the score is -sum f(k) log(k)/k.
       If we scale f(k) by alpha, score scales by alpha.
       We want to maximize alpha such that M(x) <= 1.0001.
       M_alpha(x) = alpha * M_mu(x).
       So we need alpha * max_x M_mu(x) <= 1.0001.
       alpha <= 1.0001 / max_x M_mu(x).
       Also f(1) must be in [-10, 10].
       f(1) = -alpha * S_mu.
       So |alpha * S_mu| <= 10.
       
       So we can choose alpha = min(1.0001 / max_M, 10 / |S_mu|).
       Then score = alpha * a_mu.
       
       Let's implement this.
    """
    start_time = time.time()
    deadline = start_time + 580

    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0

    def compute_mu(K):
        mu = np.zeros(K + 1, dtype=np.int8)
        mu[1] = 1
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
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

    def check_M(mu, K, alpha):
        # f(k) = alpha * mu(k) for k>1
        # f(1) = -alpha * S where S = sum_{k=2}^K mu(k)/k
        # M(x) = alpha * sum_{k=2}^K mu(k) (floor(x/k) - x/k)
        #      = -alpha * sum_{k=2}^K mu(k) {x/k}
        # We want max M(x) for x in [1, 10K].
        
        S = np.sum(mu[2:K+1] / np.arange(2, K+1, dtype=np.float64))
        
        # Compute M(x) for x=1..10K
        max_x = 10 * K
        M_vals = np.zeros(max_x + 1) # 1-indexed
        
        # M(x) = -alpha * sum_{k=2}^K mu(k) {x/k}
        # Let's compute sum_{k=2}^K mu(k) {x/k} for all x.
        # {x/k} = x/k - floor(x/k).
        # sum mu(k) x/k = x * sum mu(k)/k = x * S.
        # sum mu(k) floor(x/k).
        # Let T(x) = sum_{k=2}^K mu(k) floor(x/k).
        # Then sum mu(k) {x/k} = x*S - T(x).
        # M(x) = -alpha * (x*S - T(x)) = alpha * (T(x) - x*S).
        
        # Compute T(x) for all x in [1, 10K].
        # T(x) = sum_{k=2}^K mu(k) floor(x/k).
        # Use the O(10K log K) method.
        T = np.zeros(max_x + 1)
        for k in range(2, K + 1):
            m = mu[k]
            if m == 0:
                continue
            # floor(x/k) for x=1..max_x
            # We can update T in blocks.
            # For x in [j*k, (j+1)*k - 1], floor(x/k) = j.
            # j ranges from 0 to max_x//k.
            for j in range(1, max_x // k + 1):
                start = j * k
                end = min((j + 1) * k - 1, max_x)
                T[start:end+1] += m * j
        
        # Now M(x) = alpha * (T(x) - x*S) for x=1..max_x.
        x_vals = np.arange(1, max_x + 1, dtype=np.float64)
        M_vals = alpha * (T[1:max_x+1] - x_vals * S)
        
        max_M = np.max(M_vals)
        
        return max_M, S

    # Try increasing K
    K_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    
    for K in K_values:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        mu = compute_mu(K)
        
        # Check with alpha=1
        max_M, S = check_M(mu, K, 1.0)
        
        # Determine max alpha
        # Constraint 1: alpha * max_M <= 1.0001
        # Constraint 2: |alpha * S| <= 10
        
        alpha_max_M = 1.0001 / max_M if max_M > 0 else 1e9
        alpha_max_S = 10.0 / abs(S) if S != 0 else 1e9
        
        alpha = min(alpha_max_M, alpha_max_S, 10.0) # f(k) clipped to [-10,10], so alpha*mu(k) <= 10 => alpha <= 10
        
        if alpha <= 0:
            continue
            
        # Compute score
        # a = -sum f(k) log(k)/k
        # f(k) = alpha * mu(k) for k>1. f(1) term is 0.
        # a = -alpha * sum_{k=2}^K mu(k) log(k)/k
        
        log_k_over_k = np.log(np.arange(2, K+1, dtype=np.float64)) / np.arange(2, K+1, dtype=np.float64)
        sum_mu_log = np.sum(mu[2:K+1] * log_k_over_k)
        a = -alpha * sum_mu_log
        
        # Construct domain and values
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[1:] = alpha * mu[2:K+1]
        values[0] = -alpha * S
        
        # Clip values to [-10, 10]
        values = np.clip(values, -10, 10)
        
        # Verify feasibility with the actual values (in case clipping changed things)
        # Re-compute M(x) with clipped values
        # This is expensive, so only do it if score is promising
        if a > best_score:
            # Verify
            vals_clipped = np.clip(values, -10, 10)
            S_check = np.sum(vals_clipped / domain.astype(float))
            f_adj = vals_clipped.copy()
            f_adj[0] = f_adj[0] - S_check
            
            # Check M(x)
            max_x = 10 * K
            M_sum = np.zeros(max_x + 1)
            for k in range(1, K + 1):
                if f_adj[k-1] != 0:
                    # M_sum[x] += f_adj[k-1] * floor(x/k)
                    # Vectorized update
                    # x from 1 to max_x
                    # floor(x/k)
                    # We can use the block method again
                    m_val = f_adj[k-1]
                    for j in range(1, max_x // k + 1):
                        start = j * k
                        end = min((j + 1) * k - 1, max_x)
                        M_sum[start:end+1] += m_val * j
            
            if np.max(M_sum) <= 1.0001:
                best_score = a
                best_domain = domain.copy()
                best_values = values.copy()
            else:
                # Infeasible, try smaller alpha?
                # For now, skip
                pass

    return best_domain, best_values

# EVOLVE-BLOCK-END
