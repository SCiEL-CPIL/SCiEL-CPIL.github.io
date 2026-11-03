"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs_ac(h, temperature=1.0):
    """
    Smooth surrogate for max|AC|. 
    Uses log-sum-exp on squared values to approximate max.
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    # We want to minimize max(|ac|). 
    # max(x) ~ log(sum(exp(x^2/T))) * sqrt(T) ? No.
    # Let's use the L2 norm as a convex surrogate for L-infinity in the optimization step,
    # then verify with L-infinity.
    # Actually, minimizing L2 norm of AC is a good proxy for reducing peaks.
    return np.sqrt(np.sum(ac**2))

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin

    best_h = None
    best_val = float('inf')
    
    # Strategy: 
    # 1. Generate candidate sign patterns.
    # 2. For each, optimize positive magnitudes to minimize L2 norm of autoconvolution (convex-ish proxy).
    # 3. Evaluate true score (max-abs).
    # 4. Local search on signs.

    # Candidate Sign Patterns
    sign_patterns = []
    
    # 1. All Positive (Baseline)
    sign_patterns.append(np.ones(N))
    
    # 2. Alternating
    sign_patterns.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 3. Random Signs
    rng = random.Random(42)
    for _ in range(5):
        sign_patterns.append(np.array([1.0 if rng.random() > 0.5 else -1.0 for _ in range(N)]))
        
    # 4. Block Signs (e.g., half positive, half negative)
    sign_patterns.append(np.array([1.0]* (N//2) + [-1.0]*(N - N//2)))
    
    # 5. Legendre-like (simple pseudo-random deterministic)
    # Use a simple LFSR or just random with different seed
    for seed in [1, 2, 3]:
        rng_local = random.Random(seed)
        sign_patterns.append(np.array([1.0 if rng_local.random() > 0.5 else -1.0 for _ in range(N)]))

    # Optimization Function for Magnitudes
    # Variables: m (positive magnitudes)
    # h = signs * m
    # Objective: ||h*h||_2 (L2 norm of autoconvolution)
    
    def objective_m(m, signs):
        h = signs * m
        # Ensure h is not zero sum if possible? The score divides by sum^2.
        # If sum is 0, score is inf. We should avoid sum=0.
        # The L2 norm of AC is 0 if h is 0. 
        # We need to scale m to have a fixed sum to make the optimization well-posed?
        # The score is scale invariant: score(c*h) = score(h).
        # So we can constrain sum(m) = N (or 1) to fix the scale.
        # Let's constrain sum(m) = N.
        # If sum(h) is close to 0, the score is bad. 
        # Signs with sum 0 are bad for the denominator.
        # We will add a penalty if sum(h) is small.
        
        s = np.sum(h)
        if abs(s) < 1e-3 * N:
            return 1e10 + _soft_max_abs_ac(h)
        
        # Minimize L2 norm of AC, normalized by sum^2 to be scale invariant?
        # Since we fix scale, just minimize L2 norm.
        return _soft_max_abs_ac(h)

    # We will use SLSQP or BFGS with bounds m >= 0.
    # N=400 is large for BFGS. 
    # Instead, let's use a simpler iterative method: 
    # Gradient Descent on m with projection to simplex (sum(m)=N, m>=0).
    
    for signs in sign_patterns:
        if time.time() > deadline - 10:
            break
            
        # Initial guess: m = 1
        m = np.ones(N)
        
        # Simple Gradient Descent / Projected Gradient
        # Gradient of ||h*h||_2^2 w.r.t m?
        # This is complex. 
        # Alternative: Use scipy.optimize.minimize with method='L-BFGS-B' but limit iterations.
        # L-BFGS-B is efficient for large N with bounds.
        
        # To make it faster, we can optimize a smaller subset or use a coarse grid?
        # No, N=400 is small enough for L-BFGS-B if we limit iterations.
        
        bounds = [(0.0, None)] * N
        
        # We need to handle the scale invariance. 
        # Let's fix the sum of m to N.
        # This reduces dimensionality by 1, but L-BFGS-B doesn't handle equality constraints well.
        # We can just normalize m after each step? 
        # Or use 'SLSQP' with equality constraint.
        # SLSQP is slower.
        
        # Let's try a different approach: 
        # The optimal magnitude profile for a fixed sign sequence to minimize autoconvolution peaks 
        # often looks like a window function (tapered ends).
        # We can parametrize m using a few parameters (e.g., Gaussian width, Hanning alpha) 
        # and optimize those, then do a local search on m.
        
        # Parametric Magnitudes:
        # m_i = exp(-((i - center)^2) / (2*sigma^2)) * (1 + alpha * cos(2*pi*i/(N-1)))
        
        def make_m_param(params, N):
            sigma, alpha = params
            i = np.arange(N)
            center = (N - 1) / 2.0
            base = np.exp(-((i - center) ** 2) / (2 * sigma * sigma))
            # Ensure positive
            base = np.maximum(base, 0.01)
            # Taper
            if N > 1:
                phase = 2 * np.pi * i / (N - 1)
                taper = 1.0 + alpha * np.cos(phase)
            else:
                taper = 1.0
            m = base * taper
            # Normalize to sum N
            m = m / np.sum(m) * N
            return m

        # Optimize sigma, alpha
        def score_param(params):
            m = make_m_param(params, N)
            h = signs * m
            return _score(h)

        # Initial guess
        best_params = None
        best_p_val = float('inf')
        
        # Try a few grid points for sigma, alpha
        for sigma in [10, 20, 50, 100, 200]:
            for alpha in [-0.5, 0.0, 0.5, 1.0]:
                if time.time() > deadline - 5:
                    break
                val = score_param([sigma, alpha])
                if val < best_p_val:
                    best_p_val = val
                    best_params = [sigma, alpha]
        
        if best_params:
            # Refine with minimize_scalar? No, 2D.
            # Use minimize
            res = minimize(score_param, best_params, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-5})
            if res.success:
                m_opt = make_m_param(res.x, N)
                h_candidate = signs * m_opt
                val_candidate = _score(h_candidate)
                
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate.copy()
                    
        # Also try the unparametrized local search on the best parametric result?
        # Or just a simple random restart for m?
        
    # If we found a good candidate, do a local search on signs
    if best_h is not None:
        h_cur = best_h.copy()
        val_cur = best_val
        
        # Local Search: Flip signs of individual elements
        # This is combinatorial. N=400. 
        # Each flip changes AC.
        # We can use a greedy approach: 
        # For each i, if flipping sign reduces score, flip it.
        # Repeat until no improvement.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Random order to avoid bias
            indices = list(range(N))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline - 2:
                    break
                old_sign = np.sign(h_cur[i])
                if old_sign == 0:
                    continue
                h_cur[i] *= -1
                new_val = _score(h_cur)
                if new_val < val_cur:
                    val_cur = new_val
                    best_h = h_cur.copy()
                    improved = True
                else:
                    h_cur[i] *= -1
                    # If no improvement after a few, break? 
                    # Greedy might get stuck.
        
        # Also try small magnitude perturbations?
        # The current h_cur has signs and magnitudes.
        # We can try to optimize magnitudes again with the new signs.
        # But we already did that for the initial signs.
        # Let's do one more pass of magnitude optimization for the best sign pattern found.
        
        # Extract signs from best_h
        final_signs = np.sign(best_h)
        # Replace zeros with 1
        final_signs[final_signs == 0] = 1.0
        
        # Optimize magnitudes for these signs
        # Use the parametric approach again?
        # Or a simple gradient step on m?
        
        # Let's try a few random perturbations of the magnitudes
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            h_pert = best_h * (1 + 0.1 * np.random.randn(N))
            # Ensure signs are preserved? No, let it be.
            # But we want to keep the structure.
            # Let's just re-optimize the parametric form for these signs.
            def score_param_final(params):
                m = make_m_param(params, N)
                h = final_signs * m
                return _score(h)
            
            # Start from current magnitudes? 
            # We don't have the params that generated best_h easily.
            # Just do a quick grid search on sigma, alpha again.
            for sigma in [5, 10, 20, 50, 100]:
                for alpha in [-0.5, 0.0, 0.5]:
                    val = score_param_final([sigma, alpha])
                    if val < best_val:
                        best_val = val
                        m_new = make_m_param([sigma, alpha], N)
                        best_h = final_signs * m_new

    if best_h is None:
        # Fallback: All ones
        best_h = np.ones(N)
        best_val = _score(best_h)
        
    return best_h

# EVOLVE-BLOCK-END
