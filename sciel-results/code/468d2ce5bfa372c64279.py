"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import math

N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
    n = len(pts)
    total = 0.0
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e10  # invalid, too close
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    # Start with a golden spiral configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use L-BFGS-B optimization with gradient
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Try multiple random restarts with different initial configurations
    for restart in range(5):
        if restart == 0:
            current_pts = pts.copy()
        else:
            # Add random perturbation to create diversity
            current_pts = pts.copy()
            random.seed(42 + restart)
            noise = np.random.randn(N, 3) * 0.1
            current_pts = _project_to_sphere(current_pts + noise)
        
        # Define objective function
        def objective(x):
            p = _project_to_sphere(x.reshape(-1, 3))
            # Check for very close points
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist < 1e-12:
                return 1e10
            return _log_energy(p)
        
        # Initial point
        x0 = current_pts.flatten()
        
        # Run L-BFGS-B optimization
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < best_energy:
            opt_pts = _project_to_sphere(result.x.reshape(-1, 3))
            # Verify no points are too close
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if np.linalg.norm(opt_pts[i] - opt_pts[j]) < 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if valid and np.all(np.isfinite(opt_pts)):
                e = _log_energy(opt_pts)
                if e < best_energy:
                    best_energy = e
                    best_pts = opt_pts.copy()
    
    # Final projection to ensure unit norm
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
