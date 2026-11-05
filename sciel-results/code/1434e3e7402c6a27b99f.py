import numpy as np
import time

N = 400
_DEADLINE = 145.0  # safety margin under 150s limit

def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))


def construct():
    """Construct step heights minimizing the objective value."""
    start = time.time()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    def _update(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Strategy: Use a structured approach based on minimizing peak autocorrelation.
    # The key insight is that we want h such that its autocorrelation is as flat as possible
    # (except at lag 0), which minimizes the peak relative to the DC component.
    
    # Approach 1: Start with a known good structure and refine
    # A "flat spectrum" sequence has autocorrelation close to a delta function.
    # For real sequences, we can use m-sequences or related constructions.
    
    # Let's try a gradient-free optimization with a better parameterization.
    # We'll use a low-dimensional Fourier representation and optimize the coefficients.
    
    n = np.arange(N)
    
    # Precompute Fourier basis
    def _fourier_h(coeffs, M):
        """Construct h from Fourier coefficients."""
        h = np.zeros(N)
        # DC component
        h += coeffs[0]
        for j in range(1, M + 1):
            a = coeffs[2 * (j - 1)]
            b = coeffs[2 * (j - 1) + 1]
            h += a * np.cos(2 * np.pi * j * n / N) + b * np.sin(2 * np.pi * j * n / N)
        return h
    
    def _fourier_obj(coeffs, M):
        h = _fourier_h(coeffs, M)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e12
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))
    
    # Try different numbers of Fourier modes
    for M in [1, 2, 3, 4, 5, 6, 8, 10]:
        if time.time() - start > _DEADLINE - 5:
            break
        
        dim = 2 * M + 1  # DC + M cosine/sine pairs
        
        # Initialize with a random but structured guess
        rng = np.random.RandomState(42 + M)
        init = np.zeros(dim)
        init[0] = 1.0  # DC component
        for j in range(1, M + 1):
            init[2 * (j - 1)] = rng.uniform(-0.5, 0.5)
            init[2 * (j - 1) + 1] = rng.uniform(-0.5, 0.5)
        
        # Use Nelder-Mead for local optimization
        from scipy.optimize import minimize
        
        try:
            res = minimize(_fourier_obj, init, args=(M,), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.x is not None:
                h = _fourier_h(res.x, M)
                _update(h)
        except Exception:
            pass
        
        # Also try differential evolution for global search
        if time.time() - start > _DEADLINE - 10:
            break
        
        from scipy.optimize import differential_evolution
        
        bounds = [(-2.0, 2.0)] * dim
        # Fix DC component to be positive and significant
        bounds[0] = (0.1, 5.0)
        
        try:
            res_de = differential_evolution(
                _fourier_obj, bounds, args=(M,),
                maxiter=50, popsize=20,
                tol=0.01, seed=123 + M, polish=False
            )
            if res_de.x is not None:
                h_de = _fourier_h(res_de.x, M)
                _update(h_de)
                
                # Polish with Nelder-Mead
                if time.time() - start < _DEADLINE - 5:
                    try:
                        r2 = minimize(_fourier_obj, res_de.x, args=(M,), method='Nelder-Mead',
                                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                        if r2.x is not None:
                            h2 = _fourier_h(r2.x, M)
                            _update(h2)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Strategy 2: Direct optimization in the frequency domain using FFT
    # The autocorrelation in time domain corresponds to power spectrum in frequency domain.
    # We want to minimize the peak of the autocorrelation, which means making the power spectrum flat.
    
    if time.time() - start < _DEADLINE - 5:
        # Try a "chirp" or "sawtooth" inspired structure
        for freq_mult in [1, 2, 3, 4, 5]:
            if time.time() - start > _DEADLINE - 2:
                break
            
            # Create a band-limited signal with specific spectral properties
            h_test = np.zeros(N)
            for k in range(1, N // 2):
                # Emphasize frequencies that spread out the autocorrelation
                if k % freq_mult == 0:
                    h_test[k] = 1.0
                    h_test[N - k] = 1.0
            
            # Add DC component
            h_test += 1.0
            
            _update(h_test)
    
    # Strategy 3: Use a known optimal or near-optimal structure
    # For minimizing peak autocorrelation, "Golay" or "m-sequence" like structures are good.
    # Let's try a pseudo-random sequence with good autocorrelation properties.
    
    if time.time() - start < _DEADLINE - 5:
        rng = np.random.RandomState(12345)
        for trial in range(10):
            if time.time() - start > _DEADLINE - 2:
                break
            
            # Generate a random sequence and then "flatten" its spectrum
            h_rand = rng.randn(N)
            
            # Project onto a low-dimensional subspace to reduce peak autocorrelation
            # Use the first few Fourier modes only
            h_fft = np.fft.fft(h_rand)
            # Keep only low frequencies
            mask = np.zeros(N, dtype=bool)
            mask[:5] = True
            mask[-5:] = True
            mask[N//2 - 2:N//2 + 3] = True
            h_fft[~mask] = 0
            h_proj = np.fft.ifft(h_fft).real
            
            # Normalize
            h_proj = h_proj / np.sum(h_proj) * N
            
            _update(h_proj)
    
    # Strategy 4: Gradient descent with a good initialization
    if time.time() - start < _DEADLINE - 10:
        from scipy.optimize import minimize
        
        # Start from the best found so far and do a direct optimization
        # Using a low-dimensional parameterization: h = base + sum of a few sinusoids
        
        base = best_h.copy()
        
        def _direct_obj(params):
            """Optimize a few sinusoidal perturbations added to the base."""
            h = base.copy()
            for j in range(1, 4):  # 3 sinusoidal modes
                a = params[2 * (j - 1)]
                b = params[2 * (j - 1) + 1]
                h += a * np.cos(2 * np.pi * j * n / N) + b * np.sin(2 * np.pi * j * n / N)
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e12
            c = np.convolve(h, h)
            return abs(2.0 * N * np.max(c) / (s * s))
        
        init_params = np.zeros(6)
        try:
            res_direct = minimize(_direct_obj, init_params, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_direct.x is not None:
                h_direct = base.copy()
                for j in range(1, 4):
                    a = res_direct.x[2 * (j - 1)]
                    b = res_direct.x[2 * (j - 1) + 1]
                    h_direct += a * np.cos(2 * np.pi * j * n / N) + b * np.sin(2 * np.pi * j * n / N)
                _update(h_direct)
        except Exception:
            pass
    
    return best_h
