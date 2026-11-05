import numpy as np
import time
from scipy.optimize import minimize_scalar, differential_evolution
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Strategy: 
    # 1. Generate candidate step functions h from the logistic map.
    #    The logistic map x_{n+1} = r x_n (1 - x_n) generates chaotic sequences for r in (3.57, 4.0).
    #    We map the chaotic orbit to step heights.
    #    We normalize h such that sum(h) = 1. This removes the scale invariance and fixes the denominator.
    #    The objective becomes minimizing max(autoconv(h)).
    #    Since h is normalized, max(autoconv(h)) is bounded.
    #    Note: The problem allows negative heights. The logistic map is in [0,1].
    #    We can map it to [-1, 1] or use a signed version.
    #    Let's map x in [0,1] to h in [-1, 1] via h = 2x - 1.
    #    Then sum(h) = 2 sum(x) - N.
    #    If we normalize h such that sum(h)=1, we are effectively scaling the raw heights.
    
    def generate_h(r, x0):
        # Generate N steps of logistic map
        x = np.empty(N)
        x[0] = x0
        for i in range(1, N):
            x[i] = r * x[i-1] * (1 - x[i-1])
        
        # Map to [-1, 1]
        h_raw = 2 * x - 1
        
        # Normalize sum to 1
        s = np.sum(h_raw)
        if np.isclose(s, 0):
            # Perturb if sum is zero
            h_raw[0] += 1e-6
            s = np.sum(h_raw)
        
        h = h_raw / s
        return h

    def evaluate_params(params):
        r, x0 = params
        if not (0.0 < r < 4.0) or not (0.0 < x0 < 1.0):
            return float('inf')
        
        h = generate_h(r, x0)
        
        # Check finiteness
        if not np.all(np.isfinite(h)):
            return float('inf')
        
        s = np.sum(h)
        if np.isclose(s, 0):
            return float('inf')
        
        # Compute autocorrelation peak
        # Use FFT for speed
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n_fft)
        P = H * np.conj(H)
        acf = np.fft.irfft(P, n_fft)[:2*N-1]
        
        max_acf = np.max(acf)
        
        # The value is abs(2 * N * max_acf / s^2)
        # Since s=1, value = abs(2 * N * max_acf)
        return abs(2 * N * max_acf)

    # Phase 1: Scan r to find good regions
    # The logistic map is sensitive to r.
    # We expect good performance (low autocorrelation peak) when the orbit is well-distributed.
    
    best_r = 3.99
    best_x0 = 0.5
    best_val = float('inf')
    
    # Quick scan of r
    r_values = np.linspace(3.5, 4.0, 50)
    for r in r_values:
        # Try a few x0 values
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            val = evaluate_params((r, x0))
            if val < best_val:
                best_val = val
                best_r = r
                best_x0 = x0
        
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Local refinement around best_r, best_x0
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda p: evaluate_params(p),
                x0=[best_r, best_x0],
                bounds=[(3.0, 4.0), (0.01, 0.99)],
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_r, best_x0 = res.x
        except Exception:
            pass
    
    # Phase 3: Try Differential Evolution if time permits, to avoid local minima
    # DE on 2D is fast.
    if time.time() < deadline - 10:
        try:
            bounds = [(3.5, 4.0), (0.01, 0.99)]
            res_de = differential_evolution(
                evaluate_params,
                bounds,
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-6,
                polish=True
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_r, best_x0 = res_de.x
        except Exception:
            pass
    
    # Construct final h
    h = generate_h(best_r, best_x0)
    
    # Sanity check
    if not np.all(np.isfinite(h)) or np.isclose(np.sum(h), 0):
        return np.ones(N)
        
    return h
