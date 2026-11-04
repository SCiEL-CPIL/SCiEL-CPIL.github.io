"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    # We want to maximize R = L2^2 / (L1 * Sup_AC).
    # Since L1 is invariant under scaling, we can fix L1 = 1 (or any constant) 
    # and maximize L2^2 / Sup_AC.
    # Note: Sup_AC >= L2^2 is not generally true for non-negative h? 
    # Wait, Sup_AC is max of conv(h,h). conv(h,h)_k = sum_i h_i h_{k-i}.
    # L2^2 = sum h_i^2. This is conv(h,h)_0.
    # So Sup_AC >= conv(h,h)_0 = L2^2.
    # Thus L2^2 / Sup_AC <= 1.
    # The maximum possible value is 1, achieved if the peak of the autocorrelation is at lag 0.
    # For non-negative heights, the autocorrelation at lag 0 is the sum of squares.
    # Autocorrelation at lag k != 0 is sum h_i h_{i+k}.
    # If we can make all other lags zero, Sup_AC = L2^2, so Ratio = 1.
    # This happens if the support of h is a Sidon set (sum-free in a strong sense, or specifically B_2 set)
    # such that no two pairs of indices have the same difference, OR simply if the support is such that
    # h_i h_j = 0 for all i != j with i-j = k != 0? No.
    # If h is supported on a set S, conv(h,h)_k is sum_{i in S, i+k in S} h_i h_{i+k}.
    # If S is a Sidon set (B_2 sequence), every integer is the sum of at most one pair of elements of S.
    # This implies that for any k != 0, the equation i - j = k has at most one solution in S x S.
    # So conv(h,h)_k has at most one term.
    # To make conv(h,h)_k = 0 for all k != 0, we need the supports to be disjoint shifts?
    # No, that's impossible for a single sequence unless N=1.
    # However, we just want to minimize Sup_AC relative to L2^2.
    # If we use a Sidon set, the peaks at non-zero lags are "small" because they involve only one product h_i h_j.
    # The peak at lag 0 is sum h_i^2.
    # If we choose amplitudes such that the single products at non-zero lags are small compared to sum of squares, 
    # we can get close to 1.
    # Actually, if we make most h_i small and a few large, but ensure the large ones don't align with other large ones
    # at any non-zero lag, we can do well.
    
    # Strategy:
    # 1. Select a set of K indices from 0..N-1 that form a Sidon set (or at least have distinct differences).
    # 2. Assign non-zero heights to these K indices.
    # 3. Optimize the K heights to maximize L2^2 / Sup_AC.
    
    # Let's generate some candidate Sidon sets or random sparse sets with distinct differences.
    # A random sparse set is likely to have distinct differences if K is small.
    
    best_score = 0.0
    best_h = [1.0] / N  # Fallback: uniform distribution
    
    # Helper to calculate score
    def score(h):
        h = np.asarray(h)
        l1 = np.sum(h)
        if l1 == 0: return 0.0
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        ac = np.convolve(h_norm, h_norm, mode='full')
        sup_ac = np.max(ac)
        if sup_ac == 0: return 0.0
        return l2_sq / sup_ac
    
    # Generate candidate supports
    # We will try K = 5, 10, 15, 20
    # For each K, we will try to find a "good" set of indices.
    # A good set has many distinct differences, and ideally the differences don't repeat often.
    # Actually, we just want to optimize the amplitudes given the support.
    
    # Let's use a simple heuristic for supports:
    # Try powers of 2: 1, 2, 4, 8, 16, 32 (indices)
    # Try random sets and filter for "Sidon-like" property (distinct differences)
    
    import random
    random.seed(42)
    
    candidate_supports = []
    
    # Power of 2 indices
    pows = [1, 2, 4, 8, 16, 32]
    if len(pows) < N:
        candidate_supports.append(pows)
        
    # Random sparse sets with distinct differences
    # Check if a set has distinct differences
    def has_distinct_diffs(indices):
        diffs = set()
        for i in range(len(indices)):
            for j in range(len(indices)):
                if i != j:
                    d = indices[i] - indices[j]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    # Generate a few random Sidon sets
    attempts = 0
    while len(candidate_supports) < 20 and attempts < 1000:
        attempts += 1
        if time.time() > deadline: break
        K = random.randint(3, 15)
        indices = sorted(random.sample(range(N), K))
        if has_distinct_diffs(indices):
            candidate_supports.append(indices)
            
    # Add a few random sets even if not Sidon, to see if optimization finds a good config
    for _ in range(10):
        if time.time() > deadline: break
        K = random.randint(5, 15)
        indices = sorted(random.sample(range(N), K))
        candidate_supports.append(indices)
        
    # For each support, optimize the amplitudes
    # We fix L1 = 1. Let x be the log-amplitudes.
    # h_i = exp(x_i) / sum(exp(x_j))
    # We want to maximize L2^2 / Sup_AC.
    
    from scipy.optimize import minimize
    
    def objective(log_x, support):
        # log_x corresponds to the non-zero indices in support
        x = log_x
        # Softmax to get probabilities
        # Note: we only have K variables. The other N-K are 0.
        # Let's construct the full vector for convenience in conv
        e = np.exp(x - np.max(x))
        h_sparse = e / np.sum(e)
        
        # Create full vector
        h = np.zeros(N)
        for i, idx in enumerate(support):
            h[idx] = h_sparse[i]
            
        # Calculate score components
        l2_sq = np.sum(h**2)
        ac = np.convolve(h, h, mode='full')
        sup_ac = np.max(ac)
        
        if sup_ac < 1e-15:
            return 0.0
        return - (l2_sq / sup_ac)
        
    def score_from_log_x(log_x, support):
        e = np.exp(log_x - np.max(log_x))
        h_sparse = e / np.sum(e)
        h = np.zeros(N)
        for i, idx in enumerate(support):
            h[idx] = h_sparse[i]
        l2_sq = np.sum(h**2)
        ac = np.convolve(h, h, mode='full')
        sup_ac = np.max(ac)
        if sup_ac < 1e-15:
            return 0.0
        return l2_sq / sup_ac

    # Optimize for each support
    for support in candidate_supports:
        if time.time() > deadline: break
        K = len(support)
        if K == 0: continue
        
        # Initial guess: uniform amplitudes
        x0 = np.zeros(K)
        
        # Try a few different initializations
        inits = [
            np.zeros(K),
            np.ones(K) * 0.1,
            np.random.rand(K) * 2
        ]
        
        for init in inits:
            if time.time() > deadline: break
            try:
                res = minimize(objective, init, args=(support,), method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
                # Calculate the score of the result
                s = score_from_log_x(res.x, support)
                if s > best_score:
                    best_score = s
                    # Construct the full vector
                    e = np.exp(res.x - np.max(res.x))
                    h_sparse = e / np.sum(e)
                    h = np.zeros(N)
                    for i, idx in enumerate(support):
                        h[idx] = h_sparse[i]
                    best_h = list(h)
            except Exception:
                pass
                
    # If we haven't found anything better than the seed (which scores 1.0? No, seed scores 1.0 in search score, which is value/seed_value).
    # The seed program returns [1.0]*50.
    # Let's check the score of the seed program.
    # h = [1.0]*50. L1=50. h_norm = [0.02]*50.
    # L2^2 = 50 * 0.02^2 = 50 * 0.0004 = 0.02.
    # AC: conv(h_norm, h_norm).
    # h_norm is constant c.
    # ac_k = sum c*c = (50-k)*c^2 for k in -49..49.
    # Max is at k=0: 50 * c^2 = 50 * 0.0004 = 0.02.
    # Ratio = 0.02 / 0.02 = 1.0.
    # So the seed value is 1.0.
    # My current best_score is likely <= 1.0 if I started from 0.
    # If best_score is still 0, return seed.
    
    if best_score < 1.0:
        return [1.0] * N
        
    return best_h
# EVOLVE-BLOCK-END
