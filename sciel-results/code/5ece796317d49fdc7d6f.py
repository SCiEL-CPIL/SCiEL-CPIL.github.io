"""Program for Erdos minimum overlap task using low-dimensional parametric projection."""
import numpy as np
from scipy.optimize import differential_evolution

N = 100


def construct():
    # Define the objective function to minimize
    def objective(params):
        # params: [a1, a2, a3, p1, p2, p3]
        a1, a2, a3, p1, p2, p3 = params
        
        i = np.arange(N)
        
        # Construct sum of 3 sinusoids
        # Frequencies chosen to be incommensurate or distinct to avoid cancellation
        # Periods: N/2, N/3, N/5 ? Or just distinct frequencies.
        # Let's use frequencies f1, f2, f3.
        # To minimize correlation with 1-h, we want h to be "spread out".
        # Low frequencies are good.
        f1 = 1.0 / (2 * N)  # Period 2N
        f2 = 1.0 / (3 * N)  # Period 3N
        f3 = 1.0 / (5 * N)  # Period 5N
        
        h_raw = (a1 * np.sin(2 * np.pi * f1 * i + p1) +
                 a2 * np.sin(2 * np.pi * f2 * i + p2) +
                 a3 * np.sin(2 * np.pi * f3 * i + p3))
        
        # Project onto mean 0.5
        h = h_raw + (0.5 - np.mean(h_raw))
        
        # Clip to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        
        # Re-project to ensure sum is exactly N/2 (approximately)
        # Note: Clipping can change the sum. We adjust by adding a constant.
        # If the adjustment pushes values out of [0,1], we might have issues.
        # But with small amplitudes, the mean shift is small.
        mean_h = np.mean(h)
        h = h + (0.5 - mean_h)
        
        # Final clip for safety
        h = np.clip(h, 0.0, 1.0)
        
        # Check validity
        if not np.all(np.isfinite(h)):
            return 1e6
        if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
            return 1e6
            
        # Calculate value
        value = np.max(np.correlate(h, 1 - h, mode="full")) / N * 2
        return value

    # Bounds for parameters
    # Amplitudes: [0, 0.5] to stay within [0,1] mostly
    # Phases: [0, 2*pi]
    bounds = [(0.0, 0.5), (0.0, 0.5), (0.0, 0.5),
              (0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)]
    
    # Use differential evolution for global optimization
    # maxiter and popsize control convergence
    result = differential_evolution(objective, bounds, maxiter=100, popsize=15, tol=1e-8, seed=42)
    
    # Reconstruct h with the best parameters
    best_params = result.x
    a1, a2, a3, p1, p2, p3 = best_params
    
    i = np.arange(N)
    f1 = 1.0 / (2 * N)
    f2 = 1.0 / (3 * N)
    f3 = 1.0 / (5 * N)
    
    h_raw = (a1 * np.sin(2 * np.pi * f1 * i + p1) +
             a2 * np.sin(2 * np.pi * f2 * i + p2) +
             a3 * np.sin(2 * np.pi * f3 * i + p3))
             
    h = h_raw + (0.5 - np.mean(h_raw))
    h = np.clip(h, 0.0, 1.0)
    mean_h = np.mean(h)
    h = h + (0.5 - mean_h)
    h = np.clip(h, 0.0, 1.0)
    
    # Verify constraints
    assert np.all(np.isfinite(h)), "Heights must be finite"
    assert np.all(h >= -1e-9) and np.all(h <= 1 + 1e-9), "Heights out of bounds"
    
    return h
