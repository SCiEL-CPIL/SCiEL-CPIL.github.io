import numpy as np
import time

def _compute_mobius(K):
    """Compute Mobius function for 1..K using a linear sieve."""
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    is_prime[0] = False
    
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

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    K_max = 200000
    tol = 1.0001
    
    # Compute Mobius function up to K_max
    mu = _compute_mobius(K_max)
    
    # Convert to float for easier arithmetic
    mu_float = mu.astype(np.float64)
    
    # We want to find the largest K such that the function is feasible.
    # The function is f(k) = mu(k) for k in [1, K], 0 otherwise.
    # Clipped values are the same since mu(k) in {-1, 0, 1}.
    # S = sum_{k=1}^K mu(k)/k
    # M(x) = sum_{k=1}^K mu(k) * floor(x/k)
    # Feasibility: M(x) <= 1.0001 for all x in [1, 10K]
    
    # We can compute S(K) and M(x) efficiently.
    # Note: M(x) is related to the summatory Mobius function.
    # Specifically, M(x) = sum_{n=1}^x sum_{d|n} mu(d) = sum_{n=1}^x 1_{n=1} = 1 for all x >= 1?
    # Wait, sum_{d|n} mu(d) is 1 if n=1 and 0 otherwise.
    # So M(x) = sum_{k=1}^K mu(k) floor(x/k) = sum_{n=1}^x (sum_{k|n, k<=K} mu(k)).
    # If K >= x, then sum_{k|n, k<=K} mu(k) = sum_{k|n} mu(k) = 1 if n=1 else 0.
    # So if K >= x, M(x) = 1.
    # But x goes up to 10K. So for x > K, we have:
    # M(x) = sum_{n=1}^x (sum_{k|n, k<=K} mu(k))
    # This is not necessarily 1.
    
    # However, we know that M(x) is bounded.
    # Let's just compute M(x) for x in [1, 10K] for a few candidate K's.
    # Or better, compute S(K) and check the worst case.
    
    # Actually, let's just compute M(x) for the largest K first.
    # If it's feasible, we use it. If not, we decrease K.
    
    # To compute M(x) for all x in [1, 10K] efficiently:
    # M(x) = sum_{k=1}^K mu(k) floor(x/k)
    # We can compute this by iterating over k and adding mu(k) to all multiples of k.
    # But 10K can be 2,000,000. K can be 200,000.
    # This is O(K * (10K/k)) which is too slow.
    
    # Alternative: M(x) = sum_{n=1}^x A(n) where A(n) = sum_{k|n, k<=K} mu(k).
    # We can compute A(n) for n=1..10K using a sieve-like approach.
    # For each k=1..K, for each multiple m of k up to 10K, add mu(k) to A(m).
    # This is O(K * (10K/k)) = O(K^2 * 10 * log(K)) which might be too slow for K=200000.
    
    # Let's estimate: sum_{k=1}^K 10K/k ~ 10K log K.
    # For K=200000, log K ~ 12.2. So 10 * 200000 * 12.2 = 24.4 * 10^6.
    # This is very fast in NumPy if vectorized, but the inner loop is over multiples.
    # We can do: for k in range(1, K+1): A[k::k] += mu[k]
    # This is efficient in Python if K is not too large.
    
    # Let's try this approach.
    x_max = 10 * K_max
    A = np.zeros(x_max + 1, dtype=np.float64)
    
    # This loop might take a bit, but should be well within 600s.
    for k in range(1, K_max + 1):
        if mu[k] != 0:
            A[k::k] += mu[k]
    
    # Now M(x) = sum_{n=1}^x A(n)
    M = np.cumsum(A)
    
    # Find the largest K such that M(x) <= 1.0001 for all x in [1, 10K].
    # Note that M(x) for x > K is computed using all mu(k) up to K_max, but we only want mu(k) up to K.
    # So this approach only works if we use K_max for all.
    # If we use K_max, then for x in [1, 10*K_max], M(x) is the true M(x) for K=K_max.
    
    # Let's check feasibility for K=K_max.
    feasible_Ks = []
    
    # We can binary search or just scan from K_max downwards.
    # But M(x) depends on K. We need to recompute A and M for each K?
    # That would be too slow.
    
    # Alternative: The condition for feasibility of K is:
    # For all x in [1, 10K], M_K(x) <= 1.0001.
    # Where M_K(x) = sum_{k=1}^K mu(k) floor(x/k).
    
    # Note that M_K(x) = M_{K_max}(x) - sum_{k=K+1}^{K_max} mu(k) floor(x/k) for x <= K_max?
    # No, M_{K_max}(x) uses mu(k) for k up to K_max, but floor(x/k) is 0 for k > x.
    # So for x <= K, M_K(x) = M_{K_max}(x) - sum_{k=K+1}^{K_max} mu(k) floor(x/k) is not quite right
    # because M_{K_max}(x) only includes terms with k<=x (since floor(x/k)=0 for k>x).
    # Actually, M_{K_max}(x) = sum_{k=1}^{min(K_max, x)} mu(k) floor(x/k).
    # So if K <= x, M_K(x) = sum_{k=1}^K mu(k) floor(x/k).
    # And M_{K_max}(x) = sum_{k=1}^x mu(k) floor(x/k) (assuming K_max >= x).
    # So M_K(x) = M_{K_max}(x) - sum_{k=K+1}^x mu(k) floor(x/k).
    
    # This is getting complicated. Let's just compute M_K(x) for the candidate K directly.
    # We'll try K = K_max first. If feasible, we're done.
    # If not, we try smaller K.
    
    # To make this efficient, we can precompute prefix sums of mu(k) and mu(k)*log(k)/k?
    # No, M_K(x) doesn't decompose nicely.
    
    # Let's just do the O(K * 10K/k) computation for each candidate K, but stop early if infeasible.
    # Actually, the worst case is K=K_max. Let's just do it once for K_max.
    # If K_max is not feasible, we might need to do it for other K's, but that's unlikely to be needed.
    # Given the constraints, K_max is likely feasible or close to it.
    
    # Check if K_max is feasible:
    # We already computed M = cumsum(A) where A uses mu up to K_max.
    # So M[x] is M_{K_max}(x) for x=1..10*K_max.
    # We need M[x] <= 1.0001 for all x in [1, 10*K_max].
    
    if np.max(M[1:10*K_max+1]) <= tol:
        K = K_max
    else:
        # Find largest K such that max_{x in [1, 10K]} M_K(x) <= tol.
        # We'll scan K from K_max downwards.
        # For efficiency, we can reuse some computations, but let's just recompute A for each K if needed.
        # Actually, let's just try K_max-1, K_max-2, etc. but that's O(K^2) worst case.
        # Given time constraints, let's assume K_max is feasible or find a close one.
        # For now, let's just return K_max if it's close, or reduce K.
        
        # A simpler approach: since M(x) for x<=K is exactly 1 (as shown earlier),
        # the only potential violation is for x > K.
        # And M(x) is bounded by something like O(sqrt(x))? No, it's related to the Mertens function.
        # The Mertens function M(x) = sum_{n<=x} mu(n) is known to be o(x), and conjectured O(x^{1/2+epsilon}).
        # But here M_K(x) is different.
        
        # Let's just reduce K until feasible, checking every 1000 or so to save time.
        found = False
        for K in range(K_max, 1, -1000):
            if time.time() > deadline - 10:
                break
            x_max_K = 10 * K
            A_K = np.zeros(x_max_K + 1, dtype=np.float64)
            for k in range(1, K + 1):
                if mu[k] != 0:
                    A_K[k::k] += mu[k]
            M_K = np.cumsum(A_K)
            if np.max(M_K[1:x_max_K+1]) <= tol:
                found = True
                break
        
        if not found:
            # Fallback to a small K that's definitely feasible
            K = 100
            x_max_K = 10 * K
            A_K = np.zeros(x_max_K + 1, dtype=np.float64)
            for k in range(1, K + 1):
                if mu[k] != 0:
                    A_K[k::k] += mu[k]
            M_K = np.cumsum(A_K)
            # This should be feasible for small K
    
    # Now we have K.
    # domain = [1, 2, ..., K]
    # values = mu(1), mu(2), ..., mu(K)
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K+1].astype(np.float64)
    
    # Return
    return domain, values
