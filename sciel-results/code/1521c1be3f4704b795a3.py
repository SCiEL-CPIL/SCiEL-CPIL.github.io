import numpy as np
from scipy.optimize import minimize

def construct():
    n = 50
    # Grid points in [-1/4, 1/4]
    x_grid = np.linspace(-0.25, 0.25, n)
    
    def calc_score(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        # Autoconvolution
        c = np.convolve(h, h, mode='full')
        # Supremum norm of c
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return (l2_sq / (l1 * c_max))

    best_score = 0.0
    best_h = np.zeros(n)
    
    # Number of atoms
    K = 8
    
    # Define the mapping from parameters to height vector
    # params: [c1, w1, a1, c2, w2, a2, ...]
    # c: center, w: width (sigma), a: amplitude
    def params_to_h(params):
        h = np.zeros(n)
        for k in range(K):
            c = params[3*k]
            w = params[3*k+1]
            a = params[3*k+2]
            # Ensure width is positive and amplitude is positive
            # Use softplus or just abs/exp to maintain differentiability and positivity
            # For simplicity and robustness in scipy, we can constrain bounds or use transforms
            # Here we use a transform: w = exp(w_raw), a = exp(a_raw)
            # But scipy minimize with bounds is easier.
            # Let's just use the raw values and clip in the objective? No, that breaks gradients.
            # Let's use exp for w and a in the objective evaluation? 
            # Actually, let's just use bounds in minimize.
            sigma = w
            amp = a
            h += amp * np.exp(-0.5 * ((x_grid - c) / sigma)**2)
        return h

    def objective(params):
        h = params_to_h(params)
        # Ensure non-negative (should be if a>0, w>0)
        h = np.maximum(h, 0)
        if np.sum(h) < 1e-9:
            return -1e6
        score = calc_score(h)
        # We want to maximize score, so minimize -score
        return -score

    # We will run several restarts
    for restart in range(50):
        # Initialize parameters randomly
        params = np.zeros(3*K)
        for k in range(K):
            # Centers spread out
            params[3*k] = np.random.uniform(-0.25, 0.25)
            # Widths small to start (localized)
            params[3*k+1] = np.random.uniform(0.01, 0.05)
            # Amplitudes positive
            params[3*k+2] = np.random.uniform(0.5, 2.0)
            
        # Bounds
        # Centers in [-0.25, 0.25]
        # Widths in [0.005, 0.2]
        # Amplitudes in [0.0, 5.0]
        bounds = []
        for k in range(K):
            bounds.append((-0.25, 0.25))
            bounds.append((0.005, 0.2))
            bounds.append((0.0, 5.0))
            
        # Use Nelder-Mead as it doesn't require gradients and is robust for small dim
        # Method='Powell' might also be good.
        result = minimize(objective, params, method='Nelder-Mead', bounds=bounds, 
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        h_opt = params_to_h(result.x)
        h_opt = np.maximum(h_opt, 0)
        
        # Normalize to ensure stability? The score is scale invariant, so no need.
        # But let's check score
        score = calc_score(h_opt)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
            
        # Early exit if we hit a very high score?
        # Theoretical max for Sidon-like sets is around 2-4? 
        # If we get > 3.0, it's likely near optimal.
        if best_score > 3.0:
            break

    # Fallback
    if best_score == 0.0:
        best_h[0] = 1.0

    return list(best_h)
