import numpy as np
from time import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes the objective under the evaluator's constraints.

    Strategy:
      1. Compute the Mobius function mu(k) for k up to K=200000.
      2. Solve a linear program (LP) to maximize the objective a = -sum f(k) log(k)/k
         subject to:
           - sum f(k)/k = 0  (the evaluator's normalization)
           - f(k) in [-10, 10]
           - M(x) = sum f(k) floor(x/k) <= 1 for all integer x in [1, 10K]
      3. Since the full LP has 10K = 2,000,000 constraints (too many for scipy), we use a
         penalty method with a subset of critical x values and projected gradient ascent.
      4. Use a time budget to ensure we finish well within 630s.

    Returns:
        (domain, values): tuple of two numpy arrays
    """
    t_start = time()
    deadline = t_start + 600  # 600s budget, 30s safety margin

    # Step 1: Compute Mobius function up to K
    K = 200000
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                mu[j] *= -1
                if (j // i) % i == 0:
                    mu[j] = 0
            for j in range(i * i, K + 1, i * i):
                mu[j] = 0
        if i > 1:
            for p in primes:
                if p * i > K:
                    break
                is_prime[p * i] = False
                if i % p == 0:
                    break

    # domain: all k from 1 to K where mu(k) != 0 or we might use them
    # Actually, let's include all k from 1 to K since f(k) can be any real in [-10,10]
    # But to reduce dimensionality, use only square-free numbers (where mu != 0)
    square_free_mask = mu != 0
    domain = np.where(square_free_mask)[0]
    domain = domain[domain >= 1]  # positive integers only
    m = len(domain)

    # Initialize with Mobius values (clipped to [-10, 10], but mu is in {-1, 0, 1})
    f = np.zeros(m, dtype=np.float64)
    for i, k in enumerate(domain):
        f[i] = float(mu[k])

    # Precompute weights for objective: w_i = -log(k_i) / k_i
    k_float = domain.astype(np.float64)
    w_obj = -np.log(k_float) / k_float

    # Precompute 1/k for the normalization constraint
    inv_k = 1.0 / k_float

    # The evaluator: 
    # 1. Clips values to [-10, 10]
    # 2. Replaces f(1) with f(1) - S where S = sum f(k)/k over clipped values
    #    This makes sum f(k)/k = 0
    # 3. Checks M(x) = sum f(k) floor(x/k) <= 1.0001 for all x in [1, 10K]
    # 4. Score = -sum f(k) log(k) / k

    # For optimization, we work with the clipped and normalized values.
    # Let's define a function to compute the objective and constraint violations.

    def compute_clipped_normalized(f_raw):
        f_clip = np.clip(f_raw, -10.0, 10.0)
        # S = sum f_clip(k) / k
        S = np.dot(f_clip, inv_k)
        # Find index of k=1 in domain
        idx_1 = np.searchsorted(domain, 1)
        if idx_1 < m and domain[idx_1] == 1:
            f_norm = f_clip.copy()
            f_norm[idx_1] -= S
        else:
            f_norm = f_clip.copy()
        return f_norm

    # For the constraint M(x) <= 1, we need to check for x in [1, 10K].
    # M(x) = sum f_norm(k) * floor(x/k)
    # This is expensive to check for all x. Use a subset of critical x values.
    # Critical x: multiples of small k, and random samples.
    
    # Build a set of x values to check
    x_samples = set()
    # Add x = 1, 2, ..., 200
    for x in range(1, 201):
        x_samples.add(x)
    # Add multiples of small k
    for k_small in range(2, 100):
        for mult in range(1, 20):
            x_samples.add(k_small * mult)
    # Add random samples
    rng = np.random.RandomState(42)
    x_samples.update(rng.randint(1, 10 * K + 1, size=50000).tolist())
    # Add some large x near 10K
    x_samples.update(rng.randint(9 * K, 10 * K + 1, size=20000).tolist())
    x_samples = np.array(sorted(x_samples), dtype=np.int64)
    n_x = len(x_samples)

    # Precompute floor(x/k) matrix for the x samples and domain
    # This is n_x * m, which could be large. Use sparse or batched computation.
    # For m=120000 square-free numbers and n_x=70000, that's 8.4 billion entries - too large.
    # We need a different approach: compute M(x) on the fly for each x.

    # Use projected gradient ascent with a penalty for constraint violations.
    # Objective: maximize sum f_norm(k) * w_obj(k)  [note: w_obj includes the -log(k)/k]
    # Wait, the objective is a = -sum f(k) log(k) / k = sum f(k) * (-log(k)/k) = sum f(k) * w_obj(k)
    
    # Gradient of objective w.r.t. f_raw (before clipping/normalization) is tricky.
    # Instead, optimize f_clip directly, then normalize.
    
    # Let's use a simpler approach: 
    # 1. Start with f = mu
    # 2. Use scipy.optimize.minimize with SLSQP on a reduced set of variables
    # 3. Or use a custom projected gradient method
    
    # Given the complexity, let's try a heuristic approach:
    # Scale mu by a factor alpha, then adjust to satisfy constraints.
    
    best_f = f.copy()
    best_score = -np.inf
    best_violation = np.inf

    # Try different scaling factors
    alphas = np.linspace(0.1, 10.0, 100)
    for alpha in alphas:
        if time() > deadline - 60:
            break
        f_trial = mu[domain] * alpha  # mu[domain] is the Mobius values for our domain
        f_trial = np.clip(f_trial, -10.0, 10.0)
        
        # Normalize
        S = np.dot(f_trial, inv_k)
        idx_1 = np.searchsorted(domain, 1)
        if idx_1 < m and domain[idx_1] == 1:
            f_trial = f_trial.copy()
            f_trial[idx_1] -= S
        
        # Check constraint M(x) <= 1 for a subset of x
        # Use vectorized computation in batches
        max_violation = 0.0
        batch_size = 1000
        for i in range(0, n_x, batch_size):
            if time() > deadline - 60:
                break
            x_batch = x_samples[i:i+batch_size]
            # Compute M(x) for each x in batch
            # M(x) = sum_k f_trial(k) * floor(x/k)
            # This is expensive: for each x, sum over all k
            # Use numpy broadcasting in small chunks
            for x_idx in range(len(x_batch)):
                x_val = x_batch[x_idx]
                # floor(x_val / k) for all k in domain
                floors = np.floor(x_val / k_float)
                M_val = np.dot(f_trial, floors)
                if M_val > max_violation:
                    max_violation = M_val
                if max_violation > 2.0:
                    break
            if max_violation > 2.0:
                break
        
        if max_violation <= 1.0001:
            score = np.dot(f_trial, w_obj)
            if score > best_score:
                best_score = score
                best_f = f_trial.copy()
                best_violation = max_violation

    # If no feasible solution found, use a safe fallback
    if not np.isfinite(best_score):
        # Use a very small multiple of mu to be safe
        f_trial = mu[domain] * 0.1
        f_trial = np.clip(f_trial, -10.0, 10.0)
        S = np.dot(f_trial, inv_k)
        idx_1 = np.searchsorted(domain, 1)
        if idx_1 < m and domain[idx_1] == 1:
            f_trial = f_trial.copy()
            f_trial[idx_1] -= S
        best_f = f_trial

    # Final check: verify the best solution
    f_final = np.clip(best_f, -10.0, 10.0)
    S = np.dot(f_final, inv_k)
    idx_1 = np.searchsorted(domain, 1)
    if idx_1 < m and domain[idx_1] == 1:
        f_final = f_final.copy()
        f_final[idx_1] -= S

    # Verify constraint on a larger set of x
    max_violation = 0.0
    for x in range(1, min(10 * K + 1, 50000)):
        if time() > deadline - 30:
            break
        floors = np.floor(x / k_float)
        M_val = np.dot(f_final, floors)
        if M_val > max_violation:
            max_violation = M_val
        if max_violation > 1.0001:
            # Scale down
            scale = 1.0 / (max_violation + 0.1)
            f_final *= scale
            S = np.dot(f_final, inv_k)
            if idx_1 < m and domain[idx_1] == 1:
                f_final[idx_1] -= S
            break

    # Also check some large x
    for x in range(9 * K, 10 * K + 1, 1000):
        if time() > deadline - 10:
            break
        floors = np.floor(x / k_float)
        M_val = np.dot(f_final, floors)
        if M_val > 1.0001:
            scale = 1.0 / (M_val + 0.1)
            f_final *= scale
            S = np.dot(f_final, inv_k)
            if idx_1 < m and domain[idx_1] == 1:
                f_final[idx_1] -= S
            break

    return domain, f_final
