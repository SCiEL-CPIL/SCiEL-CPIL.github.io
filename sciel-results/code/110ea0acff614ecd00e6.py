"""Program for Erdos minimum overlap task using low-dimensional parametric projection."""
import numpy as np
from itertools import product
import scipy.optimize as optimize

N = 100


def _evaluate_h(h):
    """Compute the objective value for a given height sequence."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return float('inf')
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return float('inf')
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _project_and_clip(h):
    """Project h onto sum=N/2 and clip to [0,1], then re-project."""
    h = h.copy()
    # First clip
    h = np.clip(h, 0.0, 1.0)
    # Project onto sum = N/2
    h = h + (N / 2.0 - np.sum(h)) / N
    # Clip again
    h = np.clip(h, 0.0, 1.0)
    # Final projection
    h = h + (N / 2.0 - np.sum(h)) / N
    return h


def construct():
    """
    Use a sum of three sinusoids with variable frequencies, amplitudes, and phases.
    Grid search over 9 parameters (3 freqs, 3 amps, 3 phases) to minimize
    max(np.correlate(h, 1-h, mode="full")) / N * 2.
    Then refine using scipy.optimize.minimize.
    """
    i = np.arange(N)
    
    # Define search ranges
    # We want frequencies that create non-trivial patterns.
    # Lower frequencies create broad waves, higher frequencies create fine ripples.
    # We include a range of frequencies.
    freq_options = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    amp_options = [0.1, 0.2, 0.3, 0.4, 0.45]
    phase_options = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
    
    best_value = float('inf')
    best_h = np.full(N, 0.5)
    best_params = None
    
    # Coarse grid search
    # To keep it fast, we limit the combinations.
    # 9^3 * 5^3 * 8^3 is too big.
    # Let's pick a smaller subset for the coarse grid.
    coarse_freqs = [0.1, 0.2, 0.3, 0.5, 0.8]
    coarse_amps = [0.2, 0.3, 0.4]
    coarse_phases = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    
    # Precompute sine waves for coarse grid to speed up? 
    # Or just calculate on the fly since N=100 is small.
    
    for f1 in coarse_freqs:
        for f2 in coarse_freqs:
            for f3 in coarse_freqs:
                for a1 in coarse_amps:
                    for a2 in coarse_amps:
                        for a3 in coarse_amps:
                            for p1 in coarse_phases:
                                for p2 in coarse_phases:
                                    for p3 in coarse_phases:
                                        h_raw = 0.5 + a1 * np.sin(2 * np.pi * f1 * i + p1) \
                                                 + a2 * np.sin(2 * np.pi * f2 * i + p2) \
                                                 + a3 * np.sin(2 * np.pi * f3 * i + p3)
                                        h = _project_and_clip(h_raw)
                                        val = _evaluate_h(h)
                                        if val < best_value:
                                            best_value = val
                                            best_h = h.copy()
                                            best_params = (f1, f2, f3, p1, p2, p3, a1, a2, a3)
    
    # Refine using scipy.optimize.minimize
    if best_params is not None:
        f1, f2, f3, p1, p2, p3, a1, a2, a3 = best_params
        
        def objective(params):
            f1, f2, f3, p1, p2, p3, a1, a2, a3 = params
            h_raw = 0.5 + a1 * np.sin(2 * np.pi * f1 * i + p1) \
                     + a2 * np.sin(2 * np.pi * f2 * i + p2) \
                     + a3 * np.sin(2 * np.pi * f3 * i + p3)
            h = _project_and_clip(h_raw)
            return _evaluate_h(h)
        
        x0 = np.array(best_params)
        
        # Bounds for parameters
        # Frequencies: [0.01, 2.0]
        # Phases: [0, 2*pi]
        # Amplitudes: [0, 0.5] (max amplitude to keep within [0,1] roughly before projection)
        bounds = [
            (0.01, 2.0), (0.01, 2.0), (0.01, 2.0), # freqs
            (0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi), # phases
            (0.0, 0.5), (0.0, 0.5), (0.0, 0.5) # amplitudes
        ]
        
        # Use Nelder-Mead as it doesn't require gradients and is robust
        result = optimize.minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                                   options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 1000, 'maxfev': 1000})
        
        if result.fun < best_value:
            best_value = result.fun
            best_params = result.x
            f1, f2, f3, p1, p2, p3, a1, a2, a3 = best_params
            h_raw = 0.5 + a1 * np.sin(2 * np.pi * f1 * i + p1) \
                     + a2 * np.sin(2 * np.pi * f2 * i + p2) \
                     + a3 * np.sin(2 * np.pi * f3 * i + p3)
            best_h = _project_and_clip(h_raw)
    
    # Final verification
    assert np.all(np.isfinite(best_h)), "Heights must be finite"
    assert np.all(best_h >= -1e-9) and np.all(best_h <= 1 + 1e-9), "Heights out of bounds"
    assert abs(np.sum(best_h) - N / 2) <= 1e-9 * N, "Sum constraint violated"
    
    return best_h
