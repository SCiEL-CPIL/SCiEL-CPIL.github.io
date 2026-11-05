import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill

def _evaluate(h):
    """
    Evaluate the objective: abs(2 * N * max(autoconv) / sum(h)^2)
    Uses FFT for speed.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    # FFT based auto-convolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    
    H = np.fft.rfft(h, n=n_fft)
    S = H * np.conj(H)
    conv = np.fft.irfft(S, n=n_fft)
    
    # The linear convolution is in the first 2N-1 elements
    max_conv = np.max(conv[:2*N-1])
    
    return abs(2.0 * N * max_conv / (s * s))

def _build_h_from_spectrum(coeffs, dc_mode='auto'):
    """
    Reconstruct h from DCT-II coefficients.
    coeffs: array of length N.
    coeffs[0] is the DC term (scaled).
    """
    # scipy.fft.idct2 by default uses norm='ortho'
    # DCT-II: X[k] = sum_{n=0}^{N-1} x[n] cos(pi * (2n+1) * k / (2N))
    # IDCT-II (ortho): x[n] = (1/N) * X[0] + (2/N) * sum_{k=1}^{N-1} X[k] cos(...)
    # Wait, scipy.fft.idct2 with norm='ortho' is symmetric.
    # Let's just use scipy.fft.idct2
    try:
        from scipy.fft import idct2
        h = idct2(coeffs, type=2, norm='ortho')
    except ImportError:
        # Fallback to numpy
        n = np.arange(N)
        k = np.arange(N)
        # Construct basis matrix? Too slow.
        # Just assume scipy.fft is available as per task description "SciPy are available"
        raise ImportError("scipy.fft required")
        
    return h

def construct():
    global START_TIME, TIME_LIMIT
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h) # 2.0
    
    # Strategy:
    # 1. Parameterize h via DCT coefficients.
    #    We want to minimize peak auto-conv.
    #    Auto-conv in frequency domain is |H|^2.
    #    We want |H|^2 to be flat? No, that maximizes entropy but might not minimize peak.
    #    Actually, if h is a Dirac delta, auto-conv is a delta. Peak is huge.
    #    If h is constant, auto-conv is triangular. Peak is N/2 (relative to sum^2/N^2?).
    #    Let's look at the inspiration: "shaping low-frequency dominance with random spectral profiles".
    #    The key is that by manipulating the spectrum, we can potentially create destructive interference
    #    in the time-domain auto-convolution peaks, or simply shape the auto-convolution to be flatter.
    
    # Let's try a hybrid approach:
    # 1. Global search in a reduced dimensional space of DCT coefficients (e.g., first 20 coeffs).
    # 2. Local refinement in full time domain using L-BFGS-B on a smoothed objective.
    
    # Phase 1: Spectral Search
    # We fix the DC component to ensure sum(h) is non-zero and controlled.
    # Let h = h_dc + h_ac.
    # h_dc is constant. h_ac is zero-mean.
    # In DCT, coeffs[0] corresponds to DC.
    
    num_coeffs = 30 # Search in first 30 DCT coefficients (low frequencies)
    
    def spectral_objective(x):
        """
        x: array of length num_coeffs
        x[0] is the DC amplitude.
        x[1:] are AC amplitudes.
        """
        coeffs = np.zeros(N)
        # Map x to coeffs
        # We need to scale properly for idct2
        # For norm='ortho', the DC term in x[n] is x[0] / sqrt(N).
        # So if we want h_dc = c, we need coeffs[0] = c * sqrt(N).
        # For AC terms, h_ac[n] = 2/sqrt(N) * sum coeffs[k] cos(...).
        # So coeffs[k] = A_k * sqrt(N) / 2 for amplitude A_k.
        
        # Let's just treat x as the coefficients directly, but scaled.
        # Simpler: define h directly from x.
        # h[n] = x[0] + sum_{k=1}^{K-1} x[k] * cos(pi * (2n+1) * k / (2N))
        # This is a valid basis.
        
        n = np.arange(N)
        h = np.full(N, x[0])
        
        for k in range(1, num_coeffs):
            if x[k] != 0:
                # DCT-II basis function
                # cos(pi * (2n + 1) * k / (2N))
                h += x[k] * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
        
        if not np.all(np.isfinite(h)):
            return 1e10
            
        val = _evaluate(h)
        return val

    # Define bounds for spectral search
    # DC term: let's try a range of DC offsets.
    # AC terms: let's try a range of amplitudes.
    
    # We know that for constant h (x[0]=1, others=0), val=2.0.
    # Let's search around this.
    
    lb = np.zeros(num_coeffs)
    ub = np.zeros(num_coeffs)
    
    # DC term bounds
    lb[0] = -5.0
    ub[0] = 5.0
    
    # AC term bounds
    # Start with small amplitudes, expand if needed.
    lb[1:] = -1.0
    ub[1:] = 1.0
    
    # Use Differential Evolution for global search in this low-dim space
    # DE is robust and handles non-differentiable objectives (max) well.
    
    if time.time() - START_TIME < TIME_LIMIT - 60:
        try:
            # Use a limited number of iterations to save time
            # popsize=15, maxiter=100 -> ~1500 evaluations.
            # Each eval is FFT-based, so fast.
            res = differential_evolution(
                spectral_objective, 
                [lb, ub], 
                seed=42, 
                maxiter=50, 
                popsize=20,
                tol=1e-6,
                polish=False # We will polish later
            )
            
            if res.fun < best_val:
                best_val = res.fun
                # Reconstruct best_h from res.x
                x_opt = res.x
                n = np.arange(N)
                best_h = np.full(N, x_opt[0])
                for k in range(1, num_coeffs):
                    if x_opt[k] != 0:
                        best_h += x_opt[k] * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
                
        except Exception as e:
            pass

    # Phase 2: Local Refinement in Time Domain
    # Use the best_h from Phase 1 (or seed) as starting point.
    # Use L-BFGS-B with a smoothed max function to make it differentiable.
    
    def soft_max(x, beta=100.0):
        m = np.max(x)
        # Avoid overflow
        x_shifted = x - m
        return m + np.log(np.sum(np.exp(x_shifted)) / len(x)) / beta

    def objective_smooth(h_flat):
        h = h_flat.reshape(N)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
        
        H = np.fft.rfft(h, n=n_fft)
        S = H * np.conj(H)
        conv = np.fft.irfft(S, n=n_fft)
        
        max_conv_smooth = soft_max(conv[:2*N-1], beta=50.0)
        
        return abs(2.0 * N * max_conv_smooth / (s * s))

    # Multiple restarts for local optimization
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        if i == 0:
            h0 = best_h.copy()
        else:
            # Perturb best_h
            noise_scale = 0.1 * np.std(best_h) * (1 + i/num_restarts)
            h0 = best_h + rng.standard_normal(N) * noise_scale
        
        try:
            result = minimize(
                objective_smooth,
                h0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = result.x.reshape(N)
            val_opt = _evaluate(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Phase 3: Coordinate Descent / Greedy Refinement
    # Fine-tune individual elements to shave off remaining error.
    # This is effective for non-differentiable peaks.
    
    h_curr = best_h.copy()
    lr = 0.01 * np.std(h_curr)
    
    for coord_iter in range(50):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        improved = False
        for _ in range(N):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            j = rng.integers(0, N)
            old_val = h_curr[j]
            
            # Try a few steps
            for step in [-lr, -lr/2, lr/2, lr]:
                h_curr[j] = old_val + step
                val_new = _evaluate(h_curr)
                if val_new < best_val:
                    best_val = val_new
                    best_h = h_curr.copy()
                    improved = True
                    break
                else:
                    h_curr[j] = old_val
            
            if not improved:
                lr *= 0.9
                if lr < 1e-8:
                    break

    # Final check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    s = np.sum(h)
    conv = np.convolve(h, h)
    val = abs(2 * N * np.max(conv) / s**2)
    print(f"Score: {val}")
