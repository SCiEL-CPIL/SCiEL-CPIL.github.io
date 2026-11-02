"""
Program for the first autocorrelation inequality.
Strategy: Use a two-block disjoint rectangle structure, optimize parameters via scipy,
then refine with coordinate descent.
"""
import time
import numpy as np
from scipy import optimize

N = 600
# Interval is [-1/4, 1/4] with 600 equal intervals.
# Width of each interval is 0.5 / 600 = 1/1200.
dx = 0.5 / N

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Strategy 1: Two disjoint blocks
    # h(x) = A if x in [c1, c1+w1] U [c2, c2+w2], 0 otherwise
    # Due to symmetry and the nature of the problem, we can assume the blocks are symmetric
    # around 0 or just use two general blocks.
    # Let's parameterize by center and width of the two blocks.
    # To keep it simple, let's try blocks that are symmetric: [-d-w, -d+w] and [d-w, d+w]
    # Or just [a, b] and [c, d].
    
    def param_two_blocks(params):
        # params: [c1, w1, c2, w2]
        c1, w1, c2, w2 = params
        h = np.zeros(N)
        # Discretize: indices i correspond to x_i = -0.25 + (i+0.5)*dx (centers of intervals)
        # Or just x_i = -0.25 + i*dx
        x = np.linspace(-0.25, 0.25, N, endpoint=False) # Left edges
        # Actually, step function height on interval [x_i, x_{i+1}]
        # It's easier to check if the interval is inside the block.
        # For simplicity, let's say h[i] = A if the interval center is in the block.
        # Or better, assign 1 if the entire interval is inside? 
        # Standard step function representation: h is constant on [x_i, x_{i+1}].
        # Convolution is exact if we treat h as values on intervals.
        # np.convolve treats them as samples. 
        # Let's stick to the sample approximation for speed and simplicity, 
        # which is what compute_score does.
        
        # Mask for block 1
        mask1 = (x >= c1 - w1) & (x <= c1 + w1)
        # Mask for block 2
        mask2 = (x >= c2 - w2) & (x <= c2 + w2)
        
        h[mask1] = 1.0
        h[mask2] = 1.0
        
        # Ensure non-negative (already is)
        return h

    # Objective for optimization
    def objective(params):
        h = param_two_blocks(params)
        return compute_score(h)
        
    # Initial guess for two blocks
    # Try blocks near the edges
    initial_guesses = [
        [0.2, 0.1, -0.2, 0.1], # Symmetric near edges
        [0.15, 0.1, -0.15, 0.1],
        [0.1, 0.1, -0.1, 0.1],
        [0.22, 0.05, -0.22, 0.05],
        [0.18, 0.08, -0.18, 0.08],
        [0.0, 0.2, 0.0, 0.0] # Degenerate to one block (center)
    ]
    
    bounds = [(-0.25, 0.25), (0.0, 0.25), (-0.25, 0.25), (0.0, 0.25)]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        try:
            res = optimize.minimize(objective, guess, method='Nelder-Mead', 
                                     options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500})
            if res.fun < best_score:
                best_score = res.fun
                best_h = param_two_blocks(res.x)
        except Exception:
            pass
    
    # Strategy 2: Coordinate Descent Refinement
    # Now refine best_h by adjusting individual heights.
    # This is a local search.
    # We want to reduce the max convolution.
    # conv[k] = sum_i h[i]h[i+k]
    # Changing h[i] affects conv[k] for k in [-i, N-1-i] (roughly).
    # Specifically, conv[k] involves h[i] if k=i-j or j=i-k...
    # conv[k] = sum_j h[j] h[j+k].
    # If we change h[i], which terms in conv[k] are affected?
    # Term h[i]h[i-k] in conv[k] (where j=i) and h[i-k]h[i] (where j+k=i -> j=i-k).
    # So conv[k] is affected for all k such that i is in the range of indices.
    # Actually, changing h[i] adds delta*h[i-k] to conv[k] and delta*h[i+k] to conv[k]?
    # No.
    # Let h' = h + delta * e_i.
    # (h'*h')[k] = (h*h)[k] + delta*(h[k-i] + h[k+i]) + delta^2 * (e_i * e_i)[k].
    # (e_i * e_i)[k] is 1 if k=0 and i=i, i.e., if k=0 and the index difference is 0.
    # Wait, (e_i * e_i)[k] = sum_j e_i[j] e_i[j+k].
    # e_i[j] is 1 if j=i.
    # So sum is 1 if i=i and i+k=i -> k=0.
    # So delta^2 term only affects conv[0].
    
    # Linear terms: delta * (h[k-i] + h[k+i])?
    # Let's check indices.
    # (h + d e_i) * (h + d e_i) = h*h + d(h*e_i + e_i*h) + d^2 e_i*e_i.
    # (h*e_i)[k] = sum_j h[j] e_i[j+k] = h[i-k] (if i-k is valid index).
    # (e_i*h)[k] = sum_j e_i[j] h[j+k] = h[i+k] (if i+k is valid index).
    # So conv_new[k] = conv_old[k] + d*(h[k-i] + h[k+i]) + d^2 * delta_{k,0}.
    # Note: h is defined on 0..N-1. We treat out-of-bounds as 0.
    
    # We want to minimize max_k conv_new[k].
    # Let M = max conv_old.
    # Let k* be the index(es) where max is achieved.
    # We want to reduce conv[k*].
    # If we decrease h[i], d is negative.
    # To reduce conv[k*], we need d*(h[k*-i] + h[k*+i]) < 0.
    # Since d < 0, we need h[k*-i] + h[k*+i] > 0.
    # This means we should decrease h[i] if h[i] contributes to the peak at k*.
    # However, this might increase conv at other indices.
    
    # Simple heuristic:
    # Identify the peak index k_peak.
    # For each i, calculate the change in the peak if we reduce h[i] by a small amount.
    # Or just try setting h[i] to 0 if it's small, or scaling it down.
    
    # Let's do a few passes of "peak reduction".
    for _ in range(5):
        if time.time() > deadline:
            break
        
        conv = np.convolve(best_h, best_h, mode='full')
        k_peak = np.argmax(conv)
        peak_val = conv[k_peak]
        
        # Try to reduce h[i] for all i
        # We want to minimize the max convolution.
        # Greedy: reduce the h[i] that causes the most reduction in the peak, 
        # with minimal increase in other conv values.
        
        # Approximation: Just scale down the whole function? No, that doesn't change normalized score.
        # We need to change the shape.
        
        # Let's try adjusting h[i] based on its contribution to the peak.
        # Contribution of h[i] to conv[k_peak] is h[k_peak - i] + h[k_peak + i].
        # Let's call this c_i.
        # If we reduce h[i] by epsilon, peak reduces by epsilon * c_i.
        # Other conv[k] change by epsilon * (h[k-i] + h[k+i]).
        
        # We want to find an i and a direction (increase/decrease) that improves the max.
        # Since h >= 0, we can only decrease or increase (if 0).
        
        # Let's just do a random perturbation search on the current best_h
        # but guided by the peak.
        
        for iter in range(200):
            if time.time() > deadline:
                break
            
            h_new = best_h.copy()
            # Pick a few random indices to perturb
            num_pert = 5
            idxs = np.random.choice(N, num_pert, replace=False)
            
            for idx in idxs:
                # Decide to increase or decrease
                # If h[idx] is large, try decreasing. If small, try increasing.
                # Or just random multiplicative factor.
                factor = np.random.uniform(0.8, 1.2)
                h_new[idx] *= factor
                if h_new[idx] < 0:
                    h_new[idx] = 0
                    
            score = compute_score(h_new)
            if score < best_score:
                best_score = score
                best_h = h_new
                
    return best_h.tolist()
