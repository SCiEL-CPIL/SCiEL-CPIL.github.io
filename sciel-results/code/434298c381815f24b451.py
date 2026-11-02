import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
DEADLINE = 145  # seconds, leave margin under 150s

def _score(h):
    """Compute R = ||h||_2^2 / (||h||_1 * sup|h*h|)."""
    h = np.asarray(h, dtype=np.float64)
    if h.sum() == 0:
        return 0.0
    # Autoconvolution (full)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 <= 0 or sup_conv <= 0:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _optimize_for_support(indices, max_time=5.0):
    """
    Given a fixed set of non-zero indices, optimize the amplitudes to maximize the ratio.
    We maximize log(R) to ensure stability and positivity.
    R = sum(h_i^2) / (sum(h_i) * max_j conv_j)
    log(R) = log(sum(h_i^2)) - log(sum(h_i)) - log(max_j conv_j)
    """
    k = len(indices)
    if k == 0:
        return None, 0.0
    
    def neg_log_ratio(log_amps):
        # Recover amplitudes
        amps = np.exp(log_amps)
        h = np.zeros(N)
        h[indices] = amps
        
        l2_sq = np.sum(amps**2)
        l1 = np.sum(amps)
        
        if l1 <= 0 or l2_sq <= 0:
            return 0.0
            
        conv = np.convolve(h, h, mode='full')
        sup_conv = np.max(conv)
        
        if sup_conv <= 0:
            return 0.0
            
        # We want to maximize R, so minimize -log(R)
        # R = l2_sq / (l1 * sup_conv)
        # log(R) = log(l2_sq) - log(l1) - log(sup_conv)
        # -log(R) = -log(l2_sq) + log(l1) + log(sup_conv)
        
        return -np.log(l2_sq) + np.log(l1) + np.log(sup_conv)

    # Initial guess: equal amplitudes
    x0 = np.zeros(k)
    
    # Try a few random initializations in log-space
    best_x = None
    best_val = float('inf')
    start_time = time.time()
    
    for _ in range(3):
        if time.time() - start_time > max_time:
            break
        # Random log-amplitudes in [-2, 2]
        x_init = np.random.uniform(-2, 2, k)
        try:
            res = minimize(neg_log_ratio, x_init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    if best_x is None:
        return None, 0.0
        
    # Compute final score
    amps = np.exp(best_x)
    h = np.zeros(N)
    h[indices] = amps
    s = _score(h)
    return h, s

def construct():
    start_time = time.time()
    best_h = np.ones(N) * 0.01
    best_score = _score(best_h)
    
    # Phase 1: Search for good sparse supports
    # We will test supports of size K from 1 to 25
    # For each K, we try random subsets and some structured subsets (Sidon-like)
    
    # Helper to generate random subset
    def random_subset(k):
        return np.random.choice(N, k, replace=False)
    
    # Helper to generate Sidon-like subset (greedy)
    def sidon_subset(k):
        # Greedy Sidon set construction
        S = [0]
        while len(S) < k and len(S) < N:
            best_cand = None
            best_diffs = None
            for c in range(1, N):
                if c in S:
                    continue
                # Check if adding c creates duplicate differences
                # Differences are |c - s| for s in S
                # We want to minimize collision with existing differences
                # Existing diffs: |s1 - s2| for s1, s2 in S
                # This is O(K^2) per candidate, total O(K^3) roughly.
                # For K=25, this is fine.
                
                new_diffs = [abs(c - s) for s in S]
                # Check against existing diffs
                existing_diffs = set()
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        existing_diffs.add(abs(S[i] - S[j]))
                
                # Count collisions
                collisions = sum(1 for d in new_diffs if d in existing_diffs)
                
                if best_cand is None or collisions < best_diffs:
                    best_cand = c
                    best_diffs = collisions
                    
            if best_cand is not None:
                S.append(best_cand)
            else:
                break
        return np.array(S[:k])

    # We will iterate over K and try a mix of random and structured supports
    # Time budget allocation:
    # K=1..5: many trials
    # K=6..15: medium trials
    # K=16..25: few trials
    
    # Pre-compute some structured supports
    structured_supports = {}
    for k in range(1, 26):
        if time.time() - start_time > DEADLINE - 10:
            break
        structured_supports[k] = sidon_subset(k)

    # Main search loop
    for k in range(1, 26):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Determine number of trials for this k
        # More trials for small k, fewer for large k
        if k <= 5:
            num_trials = 50
        elif k <= 10:
            num_trials = 20
        elif k <= 15:
            num_trials = 10
        else:
            num_trials = 5
            
        # Add structured support to the list of trials
        trial_indices_list = [structured_supports[k]]
        
        for _ in range(num_trials):
            if time.time() - start_time > DEADLINE - 5:
                break
            trial_indices_list.append(random_subset(k))
            
        for indices in trial_indices_list:
            if time.time() - start_time > DEADLINE - 2:
                break
                
            h, s = _optimize_for_support(indices, max_time=2.0)
            if h is not None and s > best_score:
                best_score = s
                best_h = h.copy()
                
    # Phase 2: Refine the best found support with more iterations
    # Extract support of best_h
    support = np.where(best_h > 1e-10)[0]
    if len(support) > 0:
        # Re-optimize with more time
        h, s = _optimize_for_support(support, max_time=10.0)
        if h is not None and s > best_score:
            best_score = s
            best_h = h.copy()

    # Final safety
    best_h = np.asarray(best_h, dtype=np.float64)
    best_h[best_h < 0] = 0.0
    if best_h.sum() == 0:
        best_h = [0.01] * N
    else:
        best_h = list(best_h)
        
    return best_h

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    print(f"Time: {time.time() - t0:.2f}s, N={len(result)}")
    print(f"Score: {_score(result):.6f}")
