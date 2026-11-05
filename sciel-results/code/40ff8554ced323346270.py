import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # Use DCT-II basis for parameterization
    # h = DCT^{-1}(coeffs)
    # We optimize over a subset of low-frequency coefficients to constrain the search space
    # and ensure smoothness, which often helps in minimizing the peak of the autoconvolution.
    
    # Let's use the first K coefficients of the DCT.
    # h[k] = sum_{j=0}^{K-1} a_j * cos(pi * (k+0.5) * j / N)
    # Actually, scipy.fftpack.dct and idct handle this.
    # Let's define a function that takes frequency coefficients and returns h.
    
    K = 20  # Number of frequency components to optimize
    
    def coeffs_to_h(coeffs):
        # coeffs is a vector of length K
        # We extend it to length N with zeros for higher frequencies
        full_coeffs = np.zeros(N)
        full_coeffs[:K] = coeffs
        # Use idct (inverse DCT-II)
        h = np.fft.idct(full_coeffs, type=2, norm='ortho') * np.sqrt(N)
        # Ensure sum is not zero by adding a small constant if needed? 
        # Actually, the DC term (j=0) in DCT corresponds to the mean.
        # If we optimize the DC term, the sum changes.
        # The objective is scale-invariant, so we can fix the sum to 1 by normalizing h.
        # But it's better to let the optimizer handle the relative scales.
        # However, to avoid division by zero issues, we can normalize h to have sum 1 
        # inside the objective evaluation, or just ensure the DC component is large enough.
        # Let's normalize h to have unit sum before computing the value to make the landscape smoother?
        # No, the value is scale invariant, so f(lambda h) = f(h).
        # This means the objective function is constant along rays from the origin.
        # This creates a flat direction. Nelder-Mead might struggle with flat directions.
        # Solution: Fix the sum of h to 1 by adjusting the DC component or projecting.
        # Let's project h to have sum 1.
        s = np.sum(h)
        if abs(s) < 1e-9:
            return h # Invalid, but return something
        return h / s

    def objective(coeffs):
        h = coeffs_to_h(coeffs)
        return compute_value(h)

    # Initial guesses for coefficients
    # 1. All ones corresponds to only DC term being non-zero.
    # DCT of ones:
    h_ones = np.ones(N)
    coeffs_ones = np.fft.dct(h_ones, type=2, norm='ortho') / np.sqrt(N)
    # coeffs_ones[0] is non-zero, others are 0.
    
    # 2. Random low-frequency perturbations
    rng = np.random.default_rng(42)
    
    best_val = compute_value(np.ones(N))
    best_h = np.ones(N)
    
    # Generate diverse starting points in coefficient space
    starts = []
    
    # Start 1: Pure DC (ones)
    starts.append(coeffs_ones.copy())
    
    # Start 2: DC + small random low freq
    for _ in range(5):
        c = coeffs_ones.copy()
        # Add small random noise to low freq components
        noise = rng.normal(0, 0.1, K)
        c[:K] += noise
        starts.append(c)
        
    # Start 3: Specific wave patterns
    for k in range(1, 5):
        c = coeffs_ones.copy()
        c[k] = 0.5 # Add a specific harmonic
        starts.append(c)
        
    # Optimize from each start using Nelder-Mead
    for i, start_c in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is derivative free and good for non-smooth functions
            res = minimize(
                objective, 
                start_c, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.success or res.fun < best_val:
                h_candidate = coeffs_to_h(res.x)
                val_candidate = compute_value(h_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate.copy()
                    
        except Exception:
            pass
            
    # If we have time, do a local refinement in spatial domain from the best DCT solution
    if time.time() < deadline - 5:
        # Convert best DCT result back to spatial for local search? 
        # Or just do a few random local steps in spatial domain
        h_cur = best_h.copy()
        val_cur = best_val
        
        # Simple local search with decreasing step size
        step_size = 0.1
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            improved = False
            for _ in range(10):
                idx = rng.integers(0, N)
                h_new = h_cur.copy()
                h_new[idx] += rng.normal(0, step_size)
                val_new = compute_value(h_new)
                if val_new < val_cur:
                    h_cur = h_new
                    val_cur = val_new
                    improved = True
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
                    
        if val_cur < best_val:
            best_val = val_cur
            best_h = h_cur

    return best_h
