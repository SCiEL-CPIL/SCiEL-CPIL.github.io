import numpy as np
import time
import itertools

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    # h*h has length 2N-1
    # Zero pad to power of 2 for FFT efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = -np.inf
    
    # Strategy:
    # 1. Generate a pool of candidate supports.
    #    - Random supports of varying sizes.
    #    - "Sidon-like" supports: sets of indices where differences are unique (minimizing collisions in autoconvolution).
    #    - Structured supports: e.g., geometric progressions, primes, etc.
    # 2. For each support, optimize the weights using a fast projected gradient ascent or coordinate ascent.
    #    Since the ratio is scale invariant, we fix sum(w) = 1.
    #    The objective is to maximize sum(w^2) / max_k (sum_{i+j=k} w_i w_j).
    #    Note: max_k (sum_{i+j=k} w_i w_j) is the max coefficient of the autoconvolution.
    #    For sparse supports, the max coefficient is often just the max w_i^2 (if no collisions) or a sum of products.
    #    If the support is a Sidon set (distinct differences), the autoconvolution coefficients are either 0 or w_i^2 (for i=i) or w_i w_j (for i!=j).
    #    Wait, autoconvolution (h*h)_k = sum_{i+j=k} h_i h_j.
    #    If h is supported on S, (h*h)_k = sum_{i in S, j in S, i+j=k} h_i h_j.
    #    For a Sidon set (B2 set), each sum i+j is represented uniquely (up to order).
    #    So (h*h)_k is either 0, h_i^2 (if k=2i), or h_i h_j (if k=i+j, i!=j).
    #    The max of these is what we want to minimize (relative to sum w^2).
    #    Actually, we want to MAXIMIZE sum(w^2) / max_k (h*h)_k.
    #    This is equivalent to MINIMIZING max_k (h*h)_k / sum(w^2).
    #    
    #    Let's use a simple iterative refinement for weights given a fixed support.
    #    We can use a few steps of "power iteration" style or simple hill climbing.
    
    def optimize_weights_for_support(support, max_iter=200):
        """
        Optimize weights w for a fixed support to maximize ratio.
        Uses a simple projected gradient ascent on the log-weights or direct weights.
        We normalize sum(w)=1.
        """
        K = len(support)
        if K == 0:
            return None, -np.inf
        
        # Initialize with uniform weights
        w = np.ones(K) / K
        
        # Precompute the structure of the autoconvolution for this support
        # We need to know, for each possible sum index k, which pairs (i,j) in support sum to k.
        # support is a list of indices in [0, N-1].
        # sums range from 0 to 2*(N-1).
        # Map sum -> list of (i, j) indices in w such that support[i] + support[j] = sum.
        
        sum_map = {}
        for i in range(K):
            for j in range(K):
                s = support[i] + support[j]
                if s not in sum_map:
                    sum_map[s] = []
                sum_map[s].append((i, j))
        
        # To be efficient, we can just compute the max coefficient directly in each step.
        # But K is small (<= 20 usually), so O(K^2) is fine.
        
        best_w = w.copy()
        best_score = -np.inf
        
        # Simple hill climbing with random perturbations and projection
        T = 0.05
        for it in range(max_iter):
            # Perturb
            w_new = w.copy()
            # Add small noise
            w_new += rng.normal(0, 0.05, size=K)
            w_new = np.maximum(w_new, 0)
            s = np.sum(w_new)
            if s < 1e-10:
                continue
            w_new /= s
            
            # Compute ratio for w_new
            h_temp = np.zeros(N)
            h_temp[support] = w_new
            score = compute_ratio(h_temp)
            
            if score > best_score:
                best_score = score
                best_w = w_new.copy()
            
            # Accept/Reject (Simple Simulated Annealing)
            if score >= compute_ratio(np.concatenate([np.zeros(0)])) or True: # Just check improvement or random accept
                pass
            
            # Actually, let's just do a simple gradient-free ascent:
            # Try coordinate ascent?
            # For now, let's just keep the best found.
            # To escape local minima, we occasionally reset to uniform or random.
            if it % 50 == 0 and it > 0:
                if rng.random() < 0.1:
                    w = np.ones(K) / K
                    w = w + rng.normal(0, 0.1, size=K)
                    w = np.maximum(w, 0)
                    w /= np.sum(w)
                else:
                    w = best_w.copy()
            else:
                # Move towards best
                w = best_w.copy()
                # Add noise
                w += rng.normal(0, 0.02, size=K)
                w = np.maximum(w, 0)
                s = np.sum(w)
                if s > 0:
                    w /= s
        
        return best_w, best_score

    # Generate candidate supports
    
    # 1. Random supports of size K
    for K in range(3, 15):
        if time.time() > deadline:
            break
        for _ in range(100):
            if time.time() > deadline:
                break
            support = list(rng.choice(N, size=K, replace=False))
            w, score = optimize_weights_for_support(support, max_iter=100)
            if score > best_score:
                best_score = score
                best_h = np.zeros(N)
                best_h[support] = w
    
    # 2. Sidon-like supports
    # A Sidon set (B2 set) in [0, N-1] has distinct sums.
    # We can greedily construct them or use known constructions.
    # Let's try a few greedy constructions.
    
    def is_sidon(support):
        sums = set()
        for i in range(len(support)):
            for j in range(i, len(support)):
                s = support[i] + support[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    for _ in range(200):
        if time.time() > deadline:
            break
        # Greedy construction
        support = []
        used_sums = set()
        available = list(range(N))
        rng.shuffle(available)
        
        for idx in available:
            # Check if adding idx creates a duplicate sum
            new_sums = set()
            valid = True
            for existing in support:
                s = idx + existing
                if s in used_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                support.append(idx)
                used_sums.update(new_sums)
                # Also add 2*idx
                s_2 = 2 * idx
                if s_2 in used_sums:
                    valid = False
                    support.pop()
                    break
                used_sums.add(s_2)
        
        if len(support) >= 3:
            w, score = optimize_weights_for_support(support, max_iter=150)
            if score > best_score:
                best_score = score
                best_h = np.zeros(N)
                best_h[support] = w

    # 3. Structured: Powers of 2 (or similar sparse sets)
    # Indices: 0, 1, 2, 4, 8, 16, 32
    for base in [2, 3]:
        support = []
        val = 0
        while val < N:
            support.append(val)
            val = val * base + 1 # Slightly perturbed
            if val < N:
                val += 1
        if len(support) >= 3:
            w, score = optimize_weights_for_support(support, max_iter=150)
            if score > best_score:
                best_score = score
                best_h = np.zeros(N)
                best_h[support] = w

    # Final Polish: If we have a best_h, try to refine it with a longer run
    if best_h is not None and time.time() < deadline - 10:
        support = np.where(best_h > 1e-8)[0]
        if len(support) > 0:
            w, score = optimize_weights_for_support(list(support), max_iter=1000)
            if score > best_score:
                best_score = score
                best_h = np.zeros(N)
                best_h[support] = w

    if best_h is None:
        # Fallback: uniform
        best_h = np.ones(N) / N
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
