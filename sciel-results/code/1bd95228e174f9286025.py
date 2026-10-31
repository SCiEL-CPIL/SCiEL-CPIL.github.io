"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time
import scipy.optimize

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_x = None
    best_score = np.inf
    
    # Pre-compute FFT size for speed in scoring
    size = 2 * N - 1
    size2 = 1
    while size2 < size:
        size2 *= 2

    def fast_score(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return np.inf
        F = fft(x, size2)
        A = ifft(F * np.conj(F)).real
        m = np.max(np.abs(A[:size]))
        return m / (s * s)

    def try_candidate(x):
        nonlocal best_x, best_score
        sc = fast_score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Phase 1: Generate diverse smooth initial candidates
    # The strategy suggests smooth low-frequency sinusoidal profiles.
    # These distribute autoconvolution energy more evenly than binary steps.
    candidates = []
    
    # 1. Pure sine waves with various frequencies and phases
    # We scale them to fit in [-1, 1] and add a small DC offset to ensure sum is not zero
    # Actually, we can just use the raw sine, the optimizer will adjust amplitudes/phases.
    # But for initialization, let's try specific shapes.
    
    for freq in [1, 2, 3, 5, 8, 13]:
        for phase in [0, np.pi/2, np.pi]:
            x = np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            # Ensure values are within [-1, 1] (they are)
            # Add a small constant to avoid zero sum? 
            # The score is scale invariant in numerator/denominator? 
            # No, max|conv| scales with c^2, sum^2 scales with c^2. So ratio is scale invariant.
            # But if sum is 0, it's inf. 
            # Sine waves have sum ~ 0. 
            # We need to add a DC component or use rectified sines?
            # Let's use a mix: a + b*sin(...)
            # For now, just try the raw sine + small epsilon DC
            x_dc = x + 0.1
            candidates.append(x_dc)
            
    # 2. Rectified sines (half-wave)
    for freq in [1, 2, 4]:
        x = np.sin(2 * np.pi * freq * np.arange(N) / N)
        x = np.maximum(x, 0) # Half wave rectified
        # Normalize to [-1, 1]? No, just use as is. Sum is positive.
        candidates.append(x)
        
    # 3. Cosine bell shapes (Gaussian-like)
    for center in [N//4, N//2, 3*N//4]:
        for width in [N//4, N//8]:
            x = np.exp(-((np.arange(N) - center)**2) / (2 * width**2))
            # Center it around 0? 
            # x = x - np.mean(x)
            # Let's just use positive bump. Sum is positive.
            candidates.append(x)
            
    # 4. Random smooth noise (low pass filtered random)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x_raw = rng.normal(0, 1, N)
        # Low pass filter via FFT
        F = fft(x_raw, size2)
        # Zero out high frequencies
        cutoff = N // 4
        F[cutoff:-cutoff] = 0
        x_smooth = ifft(F).real
        x_smooth = x_smooth / np.max(np.abs(x_smooth)) # Normalize to [-1, 1]
        candidates.append(x_smooth)

    # Evaluate all initial candidates
    for x in candidates:
        if time.time() > deadline - 20:
            break
        try_candidate(x)
        
    if best_x is None:
        # Fallback to random
        best_x = rng.uniform(-1, 1, N)
        best_score = fast_score(best_x)

    # Phase 2: Nelder-Mead Optimization
    # We optimize the continuous vector x.
    # To handle the constraints x in [-1, 1], we can use a penalty or just let it run
    # and clip at the end? 
    # Nelder-Mead doesn't handle bounds natively well. 
    # Strategy: Use a transformation x = tanh(u) * 1.0 ? 
    # Or just run NM on the raw x and hope it stays in bounds? 
    # Better: Use scipy.optimize.minimize with method='Nelder-Mead' and options.
    # However, NM can drift out of bounds. 
    # Let's try optimizing the raw values but with a strong penalty if out of bounds.
    
    def objective(x):
        # Penalty for out of bounds
        if np.any(x > 1.0) or np.any(x < -1.0):
            penalty = np.sum((np.maximum(0, x - 1.0) + np.maximum(0, -1.0 - x))**2)
            return fast_score(x) + 100.0 * penalty
        return fast_score(x)

    # Run Nelder-Mead from the best initial candidate
    x0 = best_x.copy()
    
    # Adjust initial simplex size
    # Default is 5% of x0 + 0.00025. 
    # Since x0 is in [-1, 1], a small step is good.
    options = {
        'maxiter': 500, 
        'xatol': 1e-4, 
        'fatol': 1e-5
    }
    
    try:
        res = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', options=options)
        if res.x is not None:
            # Clip to [-1, 1] just in case
            res_x = np.clip(res.x, -1.0, 1.0)
            try_candidate(res_x)
    except Exception:
        pass

    # Phase 3: Local Refinement on the best continuous solution found
    # Use a finer grid search or steepest descent on the continuous domain?
    # Let's do a simple coordinate descent with line search on the best x.
    
    x_best = best_x.copy()
    sc_best = best_score
    
    # Coordinate Descent
    # For each dimension, try small perturbations
    step_sizes = [0.1, 0.01, 0.001]
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for dim in range(N):
            if time.time() > deadline - 10:
                break
            for step in step_sizes:
                # Try +step
                x_new = x_best.copy()
                x_new[dim] += step
                if -1.0 <= x_new[dim] <= 1.0:
                    sc_new = fast_score(x_new)
                    if sc_new < sc_best - 1e-8:
                        x_best = x_new
                        sc_best = sc_new
                        improved = True
                        break # Restart steps for this dim? Or just continue?
                
                # Try -step
                x_new = x_best.copy()
                x_new[dim] -= step
                if -1.0 <= x_new[dim] <= 1.0:
                    sc_new = fast_score(x_new)
                    if sc_new < sc_best - 1e-8:
                        x_best = x_new
                        sc_best = sc_new
                        improved = True
                        break

    if sc_best < best_score:
        best_x = x_best
        best_score = sc_best

    return best_x
# EVOLVE-BLOCK-END
