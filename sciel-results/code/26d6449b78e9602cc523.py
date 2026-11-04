import numpy as np
import time
import scipy.optimize as opt

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def construct():
    deadline = time.time() + 115  # Safety margin
    
    def score(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        # np.convolve(h, h) is linear convolution
        z = np.convolve(h, h)
        # The metric takes max of signed entries
        return abs(2 * N * np.max(z) / (s * s))

    # Strategy: Use a function with negative lobes to cancel the central peak of linear convolution.
    # Model: h(x) = A_center * exp(-((x-c)/w_center)^2) - A_side * exp(-((x-c-d)/w_side)^2) - A_side * exp(-((x-c+d)/w_side)^2)
    # Symmetry around center c=0.5 (index 199.5)
    
    x = np.linspace(0, 1, N)
    center = 0.5
    
    # Parameters: [A_center, A_side, w_center, w_side, d]
    # A_center, A_side > 0
    # w_center, w_side > 0
    # d > 0 (distance from center)
    
    def build_h(params):
        A_c, A_s, w_c, w_s, d = params
        # Ensure positive widths and amplitudes
        A_c = max(A_c, 0.1)
        A_s = max(A_s, 0.1)
        w_c = max(w_c, 0.01)
        w_s = max(w_s, 0.01)
        d = max(d, 0.05)
        
        # Center lobe (positive)
        h_c = A_c * np.exp(-((x - center) / w_c) ** 2)
        
        # Side lobes (negative)
        h_s1 = A_s * np.exp(-((x - (center - d)) / w_s) ** 2)
        h_s2 = A_s * np.exp(-((x - (center + d)) / w_s) ** 2)
        
        h = h_c - h_s1 - h_s2
        return h

    def objective(params):
        h = build_h(params)
        s = np.sum(h)
        if s <= 0:
            # We need sum > 0 to be valid and to keep the ratio sign consistent for optimization
            # If sum is negative, the squared term is positive, but the "lower is better" logic
            # for the raw value still applies. However, if sum is close to 0, value blows up.
            # If sum is negative, we can flip the sign of h? 
            # value = abs(... / sum^2). Flipping h doesn't change sum^2 or max(conv) sign?
            # conv(-h, -h) = conv(h, h). So sign flip of h doesn't change value.
            # But if sum is 0, it's invalid.
            # We want to avoid sum=0. 
            # Let's just return a large penalty if sum is too small or negative?
            # Actually, if sum is negative, the value is still positive. 
            # But typically we want the "main" lobe to dominate sum.
            # Let's penalize if sum < 0.1 * max_abs_h to avoid instability.
            return 1e6
        
        return score(h)

    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Initial guesses for [A_c, A_s, w_c, w_s, d]
    # We want the negative lobes to be strong enough to dip the convolution but not kill the sum.
    guesses = [
        [1.0, 0.5, 0.1, 0.1, 0.2],
        [1.0, 0.8, 0.1, 0.1, 0.15],
        [1.0, 0.3, 0.2, 0.1, 0.3],
        [1.0, 0.6, 0.15, 0.05, 0.2],
        [1.0, 1.0, 0.1, 0.1, 0.1],
        [2.0, 1.0, 0.05, 0.05, 0.1],
        [1.0, 0.4, 0.2, 0.2, 0.25],
    ]
    
    for guess in guesses:
        if time.time() > deadline - 10:
            break
        
        try:
            res = opt.minimize(objective, guess, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-8})
            
            if res.fun < best_score and np.isfinite(res.fun):
                h_opt = build_h(res.x)
                s_opt = np.sum(h_opt)
                if s_opt > 0: # Ensure valid
                    val = score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt
        except Exception:
            pass

    # If the 5-param model didn't beat seed, try a more flexible local search
    # around the best found or around a specific "optimal" structure for linear conv cancellation.
    # The minimum of linear convolution of a function with itself is related to the L2 norm.
    # For a function with zero mean, the central peak is reduced.
    
    # Let's try a direct random search with local refinement on a slightly different parametric form
    # to ensure we don't get stuck in a local min of the Gaussian mixture.
    # Form: h = A * (Gaussian_center - B * Gaussian_side)
    
    if time.time() < deadline - 20:
        # Random restarts
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random params
            A_c = np.random.uniform(0.5, 2.0)
            A_s = np.random.uniform(0.2, 1.5)
            w_c = np.random.uniform(0.05, 0.2)
            w_s = np.random.uniform(0.05, 0.2)
            d = np.random.uniform(0.1, 0.4)
            
            guess = [A_c, A_s, w_c, w_s, d]
            
            try:
                res = opt.minimize(objective, guess, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-8})
                
                if res.fun < best_score and np.isfinite(res.fun):
                    h_opt = build_h(res.x)
                    s_opt = np.sum(h_opt)
                    if s_opt > 0:
                        val = score(h_opt)
                        if val < best_score:
                            best_score = val
                            best_h = h_opt
            except Exception:
                pass

    return best_h
# EVOLVE-BLOCK-END
