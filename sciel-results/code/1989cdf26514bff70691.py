"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
DEADLINE = time.time() + 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint

def _objective(x):
    """
    x: 10 control points
    Interpolate to 100 points, project to [0,1] and sum constraint,
    return the value to minimize.
    """
    # Linearly interpolate 10 control points to 100 samples
    # Control points are at positions 0, 1, ..., 9
    # We want 100 samples at positions 0, 1, ..., 99
    # So we need to map 100 sample positions to the control point scale
    # Control point i is at position i * (99/9) = i * 11
    # Sample j is at position j
    # So sample j corresponds to control position j / 11
    
    cp_positions = np.arange(10) * 99.0 / 9.0  # 0, 11, 22, ..., 99
    sample_positions = np.arange(100)
    
    # Interpolate
    h = np.interp(sample_positions, cp_positions, x)
    
    # Project to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Enforce sum constraint: sum(h) should be N/2 = 50
    # Simple projection: shift to meet sum, then clip again, iterate
    for _ in range(50):
        s = np.sum(h)
        if abs(s - N / 2.0) < 1e-12:
            break
        h = h + (N / 2.0 - s) / N
        h = np.clip(h, 0.0, 1.0)
    
    # Final check
    if not np.all(np.isfinite(h)):
        return 1e10
    
    # Compute the value
    c = np.correlate(h, 1 - h, mode="full")
    value = np.max(c) / N * 2
    
    return value

def _objective_and_grad(x):
    """
    Objective with gradient approximation.
    """
    # Use finite differences for gradient since the projection makes analytical grad complex
    eps = 1e-6
    grad = np.zeros_like(x)
    f0 = _objective(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        xp = np.clip(xp, -1, 2)  # allow some range for interpolation
        grad[i] = (_objective(xp) - f0) / eps
    return f0, grad

def construct():
    # Start with seed: all 0.5
    # Control points for all 0.5
    x0 = np.full(10, 0.5)
    
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Try multiple initializations to find good global basin
    initializations = [x0]
    
    # Bimodal: low in middle, high at edges
    initializations.append(np.array([0.9, 0.8, 0.6, 0.3, 0.1, 0.1, 0.3, 0.6, 0.8, 0.9]))
    # Anti-bimodal: high in middle, low at edges
    initializations.append(np.array([0.1, 0.3, 0.6, 0.8, 0.9, 0.9, 0.8, 0.6, 0.3, 0.1]))
    # Alternating
    initializations.append(np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]))
    # Ramp up
    initializations.append(np.linspace(0.0, 1.0, 10))
    # Ramp down
    initializations.append(np.linspace(1.0, 0.0, 10))
    # U-shape
    initializations.append(np.array([0.8, 0.6, 0.4, 0.3, 0.25, 0.25, 0.3, 0.4, 0.6, 0.8]))
    
    for x_init in initializations:
        if time.time() > DEADLINE - 10:
            break
        
        x_init = np.clip(x_init, 0.0, 1.0)
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * 10,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Polish: do a fine search around the best found
    if time.time() < DEADLINE - 15:
        # Try small perturbations
        for _ in range(100):
            if time.time() > DEADLINE - 10:
                break
            perturbation = np.random.normal(0, 0.05, 10)
            x_try = np.clip(best_x + perturbation, 0.0, 1.0)
            
            try:
                result = minimize(
                    _objective,
                    x_try,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * 10,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    # Reconstruct final h from best control points
    cp_positions = np.arange(10) * 99.0 / 9.0
    sample_positions = np.arange(100)
    h = np.interp(sample_positions, cp_positions, best_x)
    h = np.clip(h, 0.0, 1.0)
    
    # Final projection to ensure sum constraint
    for _ in range(100):
        s = np.sum(h)
        if abs(s - N / 2.0) < 1e-12:
            break
        h = h + (N / 2.0 - s) / N
        h = np.clip(h, 0.0, 1.0)
    
    # Ensure finite
    h = np.where(np.isfinite(h), h, 0.5)
    
    return h
# EVOLVE-BLOCK-END
