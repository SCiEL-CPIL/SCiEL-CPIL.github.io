"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_overlap(h, shifts=None):
    """
    Compute the maximum overlap of h with shifted 1-h.
    Overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum over all shifts k of this quantity.
    
    Actually, let's re-read the task: "minimize the maximum overlap of h with shifted 1-h".
    The overlap of h with shifted 1-h at shift k is:
        sum_{i=0}^{N-1} h[i] * (1 - h[(i+k) % N])
    
    Since sum h = N/2, we have:
        sum h[i]*(1-h[(i+k)%N]) = sum h[i] - sum h[i]*h[(i+k)%N]
                                 = N/2 - autocorr(k)
    
    where autocorr(k) = sum_i h[i] * h[(i+k)%N].
    
    So minimizing the maximum overlap is equivalent to:
        min_h max_k [N/2 - autocorr(k)]
    which is equivalent to:
        max_h min_k autocorr(k)
    
    Wait, that's not quite right. Let's think again.
    
    Overlap_k = sum_i h[i] * (1 - h[(i+k)%N]) = N/2 - sum_i h[i]*h[(i+k)%N]
    
    We want to minimize max_k Overlap_k = N/2 - min_k autocorr(k).
    
    So minimizing max_k Overlap_k is equivalent to maximizing min_k autocorr(k).
    
    This is a max-min problem. The optimal value should be related to the structure of h.
    
    For h[i] = 0.5 for all i:
        autocorr(k) = sum_i 0.5 * 0.5 = N * 0.25 = N/4
        Overlap_k = N/2 - N/4 = N/4
        Max overlap = N/4 = 25 for N=100.
    
    Can we do better? We want to maximize the minimum autocorrelation.
    
    Note that autocorr(0) = sum h[i]^2. By Cauchy-Schwarz, sum h[i]^2 >= (sum h[i])^2 / N = (N/2)^2 / N = N/4.
    So autocorr(0) >= N/4.
    
    For the minimum over k, we want all autocorr(k) to be as large as possible.
    
    If we could make autocorr(k) = N/4 for all k, then min autocorr = N/4, and max overlap = N/4 - N/4 = 0? No wait.
    
    Overlap_k = N/2 - autocorr(k). If autocorr(k) = N/4, then Overlap_k = N/2 - N/4 = N/4.
    
    So the constant h=0.5 gives overlap N/4 for all shifts.
    
    Is it possible to get a lower maximum overlap? That would require some autocorr(k) > N/4 for all k? No, we need min_k autocorr(k) to be larger.
    
    Actually, let's reconsider. The sum of all autocorrelations:
        sum_k autocorr(k) = sum_k sum_i h[i] h[(i+k)%N] = sum_i sum_k h[i] h[(i+k)%N] = sum_i h[i] * sum_j h[j] = (sum h)^2 = (N/2)^2 = N^2/4.
    
    So the average autocorr(k) over all k is (N^2/4) / N = N/4.
    
    Therefore, min_k autocorr(k) <= N/4 (since the average is N/4).
    
    So max_k Overlap_k = N/2 - min_k autocorr(k) >= N/2 - N/4 = N/4.
    
    And the constant h=0.5 achieves exactly N/4. So the optimal value is N/4 = 25 for N=100.
    
    Wait, but the task says "score is the upper bound on C5" and "lower is better". The seed returns 0.5 for all, which gives overlap N/4 = 25. But the score is 0.0000, which suggests the evaluation might be different.
    
    Let me re-read: "score is the upper bound on C5". Maybe the overlap is normalized? Or maybe I'm misunderstanding the overlap definition.
    
    Let me just implement the objective correctly and try to minimize it. The constant 0.5 might be optimal, but let's verify with a simple computation.
    """
    if shifts is None:
        shifts = range(N)
    
    overlaps = []
    for k in shifts:
        shifted = np.roll(h, -k)
        overlap = np.sum(h * (1 - shifted))
        overlaps.append(overlap)
    return max(overlaps)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Start with the known optimal candidate: constant 0.5
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _compute_overlap(h)
    
    # Try a few simple structured candidates
    candidates = []
    
    # Constant
    candidates.append(np.full(N, 0.5))
    
    # Alternating 0 and 1 (if N is even)
    if N % 2 == 0:
        alt = np.zeros(N)
        alt[::2] = 1.0
        candidates.append(alt)
    
    # Periodic patterns
    for period in range(2, N+1):
        if N % period != 0:
            continue
        # Try different phase patterns within each period
        # Simple: half 1s, half 0s in each period
        if period % 2 == 0:
            pattern = np.zeros(period)
            pattern[:period//2] = 1.0
            h_cand = np.tile(pattern, N // period)
            candidates.append(h_cand)
    
    # Evaluate all candidates
    for h_cand in candidates:
        if np.sum(h_cand) != N / 2:
            continue
        score = _compute_overlap(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_cand.copy()
    
    # Try local optimization using a simple gradient-free method
    # Since the problem is discrete-ish (step heights), let's try random restarts with local search
    
    np.random.seed(42)
    
    def _local_search(h_init, max_iters=1000):
        h = h_init.copy()
        best_local = _compute_overlap(h)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random index to flip or adjust
            idx = np.random.randint(N)
            
            # Try flipping: if h[idx] is close to 0, try 1; if close to 1, try 0
            # Or just try small perturbations
            # Since we need sum = N/2, let's try swapping two elements
            jdx = np.random.randint(N)
            if idx == jdx:
                continue
            
            # Try swapping values
            h_new = h.copy()
            h_new[idx], h_new[jdx] = h[jdx], h[idx]
            
            if np.sum(h_new) != N / 2:
                continue
            
            score_new = _compute_overlap(h_new)
            if score_new < best_local:
                h = h_new
                best_local = score_new
        
        return h, best_local
    
    # Do a few random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random binary vector with sum N/2
        if N % 2 == 0:
            h_rand = np.zeros(N)
            indices = np.random.choice(N, N//2, replace=False)
            h_rand[indices] = 1.0
        else:
            continue
        
        h_opt, score_opt = _local_search(h_rand, max_iters=500)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
    
    return best_h
# EVOLVE-BLOCK-END
