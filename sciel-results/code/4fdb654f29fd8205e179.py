import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total < 1e-15:
            return float('inf')
        # Autoconvolution via FFT
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def spectral_to_heights(params):
        """
        Convert spectral parameters to step heights.
        
        The spectrum is modeled as a smooth, band-limited function:
        |H(k)| = exp(-alpha * (k/K_max)^2) * (1 + sum_j c_j * cos(2*pi*j*k/N))
        
        where k ranges over frequency bins, K_max is the bandwidth,
        alpha controls the spectral shape, and c_j are small window coefficients.
        
        Phase is set to zero (real, even spectrum) for simplicity.
        
        params = [K_max, alpha, c_1, c_2, ..., c_M]
        """
        M = 4  # number of cosine window coefficients
        K_max = params[0]  # bandwidth in units of N
        alpha = params[1]  # spectral decay
        coeffs = params[2:2+M]  # window coefficients
        
        # Frequency bins for rfft: 0 to N (inclusive), total N+1 bins
        k = np.arange(N + 1, dtype=float)
        
        # Base spectrum: Gaussian in frequency
        base = np.exp(-alpha * (k / (K_max * N)) ** 2)
        
        # Add cosine window terms to shape the spectrum
        for j in range(M):
            if abs(coeffs[j]) > 1e-12:
                base += coeffs[j] * np.cos(2.0 * np.pi * j * k / N)
        
        # Ensure nonnegative spectrum
        base = np.maximum(base, 0.0)
        
        # H is real and even, so H[k] = sqrt(base[k]) with zero phase
        H = np.sqrt(base)
        
        # Inverse FFT to get heights
        h = np.fft.irfft(H, n=N)
        
        # Project to nonnegative
        h = np.maximum(h, 0.0)
        
        # Normalize
        total = np.sum(h)
        if total > 1e-15:
            h = h / total
        else:
            h = np.ones(N) / N
        
        return h

    def objective(params):
        """Objective function for optimization."""
        if time_left() < 5:
            return 1e10
        h = spectral_to_heights(params)
        return compute_metric(h)

    # Differential evolution in spectral parameter space
    M = 4
    n_params = 2 + M  # K_max, alpha, c_1..c_M
    
    # Bounds: K_max in [0.05, 0.5], alpha in [0.1, 5.0], coeffs in [-0.3, 0.3]
    bounds = [(0.05, 0.5), (0.1, 5.0)]
    for _ in range(M):
        bounds.append((-0.3, 0.3))
    
    # Run differential evolution
    best_h = None
    best_m = float('inf')
    
    try:
        if time_left() > 60:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.7,
                seed=42,
                polish=True,
                init='sobol'
            )
            if de_result.fun < best_m:
                best_m = de_result.fun
                best_h = spectral_to_heights(de_result.x)
    except Exception:
        pass
    
    # Local refinement with L-BFGS-B from the best DE result
    if best_h is not None and time_left() > 30:
        # Use the DE result as initial guess
        if 'de_result' in dir() and de_result is not None:
            x0 = de_result.x
        else:
            x0 = np.array([0.3, 1.0] + [0.0]*M)
        
        try:
            lbfgs_result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if lbfgs_result.fun < best_m:
                best_m = lbfgs_result.fun
                best_h = spectral_to_heights(lbfgs_result.x)
        except Exception:
            pass
    
    # If DE didn't run or failed, try a few manual initializations
    if best_h is None or time_left() > 40:
        manual_inits = [
            [0.3, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.1, 0.0, 0.0, 0.0],
            [0.2, 2.0, 0.0, 0.1, 0.0, 0.0],
            [0.4, 1.5, -0.1, 0.05, 0.0, 0.0],
            [0.25, 1.0, 0.05, -0.05, 0.02, 0.0],
        ]
        
        for x0 in manual_inits:
            if time_left() < 15:
                break
            try:
                res = minimize(
                    objective,
                    np.array(x0),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_m:
                    best_m = res.fun
                    best_h = spectral_to_heights(res.x)
            except Exception:
                pass
    
    # Final direct refinement: use the best spectral result as a starting point
    # and do a few rounds of direct height-space optimization
    if best_h is not None and time_left() > 20:
        h = best_h.copy()
        
        # Simple local refinement: perturb small groups of heights
        np.random.seed(123)
        for it in range(2000):
            if time_left() < 10:
                break
            
            # Pick a random contiguous block
            block_len = np.random.randint(3, 15)
            start_idx = np.random.randint(0, N - block_len + 1)
            
            # Perturb the block with a small random offset
            delta = np.random.normal(0, 0.01, block_len)
            
            new_h = h.copy()
            new_h[start_idx:start_idx+block_len] += delta
            new_h = np.maximum(new_h, 0)
            
            # Renormalize
            total = np.sum(new_h)
            if total > 1e-15:
                new_h = new_h / total
            
            m_new = compute_metric(new_h)
            if m_new < best_m:
                best_m = m_new
                best_h = new_h
                h = new_h
    
    # Ensure we always return a valid result
    if best_h is None:
        best_h = np.ones(N) / N
        best_m = compute_metric(best_h)
    
    return list(best_h)

# EVOLVE-BLOCK-END
