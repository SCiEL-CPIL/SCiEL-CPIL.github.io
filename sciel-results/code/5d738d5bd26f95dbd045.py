import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 600

def _score_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    The problem asks to minimize max(autoconv) / (area * max_val).
    Since h is a step function on equal intervals, area = sum(h) * dx.
    max_val = max(h).
    autoconv is the convolution of h with itself.
    """
    h = np.asarray(h)
    if np.any(h < 0):
        return float('inf')
    
    area = np.sum(h)
    max_val = np.max(h)
    
    if area == 0 or max_val == 0:
        return float('inf')
        
    # Compute autoconvolution
    # The autoconvolution of a sequence h is (h * h).
    # For discrete sequences, np.convolve(h, h, 'full') gives the full convolution.
    # The "normalized maximum" usually refers to the peak of the autoconvolution 
    # relative to the scale of the function.
    # The task says "minimize the normalized maximum of its autoconvolution".
    # Given the context of "upper bound on C1" and the seed program returning uniform,
    # let's look at the score definition implied by the inspirations.
    # Inspiration 1 says: "minimize max(autoconv) / (area * max_val)"
    
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    
    return max_conv / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Strategy: "Spike on Uniform"
    # Let h be uniform with value 'u' everywhere, except at one index k where it is 'v'.
    # By symmetry, the position of the spike doesn't matter for the max autoconvolution,
    # only the values u and v.
    # We fix the scale. Let sum(h) = 1.
    # Then (N-1)u + v = 1  => u = (1-v)/(N-1).
    # We need u >= 0 and v >= 0. So 0 <= v <= 1.
    # Also, for it to be a "spike", we usually expect v > u, but let's search broadly.
    
    def objective(v):
        if v < 0 or v > 1:
            return float('inf')
        u = (1.0 - v) / (N - 1)
        h = np.full(N, u)
        h[0] = v # Place spike at index 0
        
        # Quick check for time
        if time.time() > deadline:
            return float('inf')
            
        return _score_from_h(h)
    
    # Search for optimal v
    # We expect v to be somewhat large.
    # Let's do a coarse grid search first to find a good starting point, then refine.
    
    best_v = 1.0 / N # Uniform
    best_score = _score_from_h(np.ones(N) / N)
    
    # Grid search
    vs = np.linspace(0, 1, 1000)
    for v in vs:
        if time.time() > deadline:
            break
        s = objective(v)
        if s < best_score:
            best_score = s
            best_v = v
            
    # Refine with scipy
    if time.time() < deadline - 5:
        # Use the best v from grid as bracket center or just use bounds
        # minimize_scalar requires a function of one variable
        res = minimize_scalar(objective, bounds=(0, 1), method='bounded', 
                              options={'xatol': 1e-10, 'maxiter': 100})
        if res.success and time.time() < deadline:
            v_opt = res.x
            s_opt = res.fun
            if s_opt < best_score:
                best_score = s_opt
                best_v = v_opt
    
    # Construct final h
    u = (1.0 - best_v) / (N - 1)
    h = np.full(N, u)
    h[0] = best_v
    
    return [float(x) for x in h]

if __name__ == "__main__":
    # Self-test
    h = construct()
    print(f"Score: {_score_from_h(h)}")
    print(f"Max height: {max(h)}")
    print(f"Min height: {min(h)}")
    print(f"Sum: {sum(h)}")
