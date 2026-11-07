"""Improved program using parametric optimization for smooth waveforms."""
import numpy as np
from scipy.optimize import minimize

N = 100

def _parametric_h(params):
    """
    Generate h from parameters.
    params: [A, P, phi]
    A: Amplitude
    P: Period
    phi: Phase
    """
    A = params[0]
    P = params[1]
    phi = params[2]
    
    # Ensure P is not zero
    P = max(P, 1.0)
    
    i = np.arange(N)
    # We want sum(h) = N/2.
    # If we use a pure sine with integer number of periods, sum is 0.
    # If P is not integer, sum is not 0.
    # We can correct the sum by subtracting the mean deviation.
    # h_raw = 0.5 + A * sin(2*pi*i/P + phi)
    # h = h_raw - mean(h_raw) + 0.5
    
    h_raw = 0.5 + A * np.sin(2.0 * np.pi * i / P + phi)
    h = h_raw - np.mean(h_raw) + 0.5
    
    # Clip to [0, 1]
    # Clipping might break the sum constraint slightly, but A is small enough usually.
    # If A is close to 0.5, clipping will occur.
    # We should ensure A is such that h stays in [0,1] without clipping if possible,
    # or accept clipping and re-normalize?
    # Re-normalizing after clipping is complex.
    # Better to constrain A <= 0.5 * (1 - epsilon) to avoid clipping?
    # Or just clip and check sum.
    
    h = np.clip(h, 0.0, 1.0)
    
    # Adjust sum exactly
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = current_sum - target_sum
    if abs(diff) > 1e-12:
        h -= diff / N
        h = np.clip(h, 0.0, 1.0)
        # If clipping again, do a second pass
        current_sum = np.sum(h)
        diff = current_sum - target_sum
        if abs(diff) > 1e-12:
             # Distribute to elements not at bounds
             mask = (h > 0.0) & (h < 1.0)
             if np.sum(mask) > 0:
                 h[mask] -= diff / np.sum(mask)
             else:
                 h -= diff / N
             h = np.clip(h, 0.0, 1.0)

    return h

def _objective(params):
    h = _parametric_h(params)
    corr = np.correlate(h, 1.0 - h, mode="full")
    val = np.max(corr) / N * 2.0
    # Penalty if h goes out of bounds (should be 0 due to clip)
    out_of_bounds = np.sum((h < 0) | (h > 1))
    return val + out_of_bounds * 100.0

def construct():
    # Try a few initial guesses for (A, P, phi)
    # A in [0, 0.5], P in [1, 50], phi in [0, 2pi]
    
    best_h = np.full(N, 0.5)
    best_val = 0.5 # Seed value
    
    # Define bounds
    bounds = [(0.0, 0.5), (1.0, 50.0), (0.0, 2.0 * np.pi)]
    
    # Multi-start optimization
    np.random.seed(42)
    for _ in range(10):
        # Random start
        A0 = np.random.uniform(0.1, 0.45)
        P0 = np.random.uniform(5, 40)
        phi0 = np.random.uniform(0, 2 * np.pi)
        
        x0 = [A0, P0, phi0]
        
        try:
            res = minimize(
                _objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if res.success:
                h_opt = _parametric_h(res.x)
                val_opt = np.max(np.correlate(h_opt, 1.0 - h_opt, mode="full")) / N * 2.0
                
                # Verify validity
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt) - 50.0) < 1e-9 and np.all(h_opt >= 0) and np.all(h_opt <= 1):
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt
        except Exception:
            pass
            
    # Also try a specific smooth guess: A=0.4, P=20
    x0 = [0.4, 20.0, 0.0]
    try:
        res = minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res.success:
            h_opt = _parametric_h(res.x)
            val_opt = np.max(np.correlate(h_opt, 1.0 - h_opt, mode="full")) / N * 2.0
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt) - 50.0) < 1e-9 and np.all(h_opt >= 0) and np.all(h_opt <= 1):
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt
    except Exception:
        pass

    return best_h
