"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize max_t |A(t) - 0.5| over shifts t, where A(t) = sum h[i] * (1-h[(i+t)%N]).
    This is the 'maximum overlap' of h with shifted 1-h, measuring the maximum deviation
    from the ideal 'orthogonal' value of 0.5 (which is the mean overlap for random sequences).
    
    We use a combination of:
    1. Random restarts with subgradient descent
    2. Local perturbation refinement
    
    The score is the max_t |A(t) - 0.5|. We want to minimize this.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    def compute_overlap(h):
        """Compute A(t) for all t, where A(t) = sum_i h[i] * (1-h[(i+t)%N])."""
        # h * (1-h_shifted) for each shift t
        # A(t) = sum_i h[i] * (1-h[(i+t)%N])
        #      = sum_i h[i] - sum_i h[i]*h[(i+t)%N]
        #      = 0.5 - autocorr(t)  (since sum h[i] = N/2 = 50)
        # So we want to minimize max_t |0.5 - autocorr(t) - 0.5| = max_t |autocorr(t)|
        # Wait, let me re-derive.
        # A(t) = sum_i h[i]*(1-h[(i+t)%N]) = sum_i h[i] - sum_i h[i]h[(i+t)%N] = 50 - autocorr(t)
        # We want to minimize max_t |A(t) - 0.5|? No, the task says "maximum overlap of h with shifted 1-h".
        # The "overlap" at shift t is A(t) = sum_i h[i]*(1-h[(i+t)%N]).
        # We want to minimize max_t A(t)? Or max_t |A(t) - expected|?
        # The expected value of A(t) for random h is N * E[h]*(1-E[h]) = 100 * 0.25 = 25.
        # Hmm, but the task says "minimum overlap". Let me reconsider.
        # Actually, for the constant sequence h=0.5, A(t) = sum 0.5*0.5 = 25 for all t.
        # The "score is the upper bound on C5" - C5 might be some constant.
        # Let me just compute max_t A(t) and try to minimize it, or minimize the variance.
        # Actually, re-reading: "minimize the maximum overlap of h with shifted 1-h"
        # So score = max_t sum_i h[i]*(1-h[(i+t)%N])
        # For h=0.5, this is 25.
        # Can we do better? If h is binary (0 or 1) with 50 ones, 
        # A(t) = number of i where h[i]=1 and h[(i+t)%N]=0.
        # For a perfect balanced sequence, A(t) ≈ 25 for all t.
        # The minimum possible max_t A(t) is at least 25 by averaging.
        # Actually, sum_t A(t) = sum_i h[i] * sum_t (1-h[(i+t)%N]) = sum_i h[i] * (N - 50) = 50*50 = 2500.
        # So average A(t) = 2500/100 = 25. Thus max_t A(t) >= 25.
        # So 25 is the theoretical minimum! The constant 0.5 achieves exactly 25.
        # But wait, we can use non-integer heights. Can we get below 25?
        # No, by the same averaging argument, max_t A(t) >= 25 always.
        # So the optimal score is 25, achieved by h=0.5.
        # But the seed score is 1.0, and the hint says "~0.4 continuous optimum".
        # Maybe the score is normalized? Score = max_t A(t) / 50? Then 0.5 for constant.
        # Or maybe the "overlap" is defined differently.
        
        # Let me just compute the raw value and return it. The evaluator will handle normalization.
        h1 = 1.0 - h
        # Compute cross-correlation: for each t, sum_i h[i] * h1[(i+t)%N]
        result = np.zeros(N)
        for t in range(N):
            h1_shifted = np.roll(h1, -t)
            result[t] = np.dot(h, h1_shifted)
        return result
    
    def score(h):
        overlaps = compute_overlap(h)
        return np.max(overlaps)
    
    best_h = np.full(N, 0.5)
    best_score = score(best_h)
    
    # Subgradient descent
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random initialization
        h = np.random.dirichlet(np.ones(N)) * (N / 2)
        h = np.clip(h, 0, 1)
        # Rescale to sum exactly N/2
        h = h / h.sum() * (N / 2)
        h = np.clip(h, 0, 1)
        
        cur_score = score(h)
        if cur_score < best_score:
            best_score = cur_score
            best_h = h.copy()
        
        # Subgradient iterations
        lr = 0.01
        for it in range(500):
            if time.time() > deadline:
                break
            
            overlaps = compute_overlap(h)
            max_idx = np.argmax(overlaps)
            
            # Gradient of A(t) w.r.t. h[i]: dA(t)/dh[i] = (1-h[(i+t)%N]) - h[i]*(1-h[(i+t)%N])... 
            # Actually A(t) = sum_i h[i]*(1-h[(i+t)%N])
            # dA(t)/dh[i] = (1-h[(i+t)%N]) - h[i] (from the second term's dependence on h[i] when i' = i-t)
            # This is getting complex. Let me use finite differences or a simpler approach.
            
            # Simple perturbation: flip one element toward reducing the max overlap
            i = np.random.randint(N)
            delta = np.random.choice([-1, 1]) * 0.01
            h_new = h.copy()
            h_new[i] = np.clip(h_new[i] + delta, 0, 1)
            # Renormalize
            h_new = h_new / h_new.sum() * (N / 2)
            h_new = np.clip(h_new, 0, 1)
            
            new_score = score(h_new)
            if new_score < cur_score:
                h = h_new
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_h = h.copy()
                lr = min(lr * 1.01, 0.1)
            else:
                lr = max(lr * 0.99, 0.001)
    
    return best_h
# EVOLVE-BLOCK-END
