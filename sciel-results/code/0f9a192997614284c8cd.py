import numpy as np
from scipy.optimize import minimize
import random

N = 12

# EVOLVE-BLOCK-START
def _hex_constraint(d, theta_outer):
    """Compute the maximum |d·u| + 1 over 6 hexagon directions for a point at distance d from center."""
    # 6 directions of the outer hexagon
    angles = theta_outer + np.arange(6) * (np.pi / 3.0)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # d is (x, y), compute dot products
    dots = np.abs(d[0] * cos_a + d[1] * sin_a)
    return np.max(dots) + 1.0

def _objective(params):
    """params: [x0,y0, x1,y1, ..., x11,y11, theta_outer] -> outer side length (smooth approx)"""
    n = N
    centers = params[:2*n].reshape(n, 2)
    theta_outer = params[2*n]
    
    # For each inner hexagon center, compute the required outer side
    constraints = []
    for i in range(n):
        d = centers[i]
        angles = theta_outer + np.arange(6) * (np.pi / 3.0)
        dots = np.abs(d[0] * np.cos(angles) + d[1] * np.sin(angles))
        constraints.append(np.max(dots) + 1.0)
    
    constraints = np.array(constraints)
    # Smooth max using log-sum-exp with large beta
    beta = 50.0
    m = np.max(constraints)
    smooth_max = m + np.log(np.mean(np.exp(beta * (constraints - m)))) / beta
    return smooth_max

def _gradient(params):
    """Numerical gradient of the objective."""
    eps = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (_objective(params_plus) - _objective(params_minus)) / (2 * eps)
    return grad

def construct():
    n = N
    
    # Initial guess: place hexagons in a ring or grid pattern
    # Use a hexagonal packing arrangement
    random.seed(42)
    np.random.seed(42)
    
    # Initial positions: spread them out in a circle
    angles_init = np.linspace(0, 2*np.pi, n, endpoint=False)
    r_init = 2.0
    centers = np.column_stack([r_init * np.cos(angles_init), r_init * np.sin(angles_init)])
    theta_outer_init = 0.0
    
    x0 = np.concatenate([centers.flatten(), [theta_outer_init]])
    
    # Try multiple initializations
    best_x = x0
    best_val = _objective(x0)
    
    for trial in range(5):
        # Different initial configurations
        if trial == 0:
            angles_init = np.linspace(0, 2*np.pi, n, endpoint=False)
            r_init = 2.0
        elif trial == 1:
            # Hexagonal lattice
            rows = []
            for i in range(4):
                for j in range(3):
                    x = j * 2.0 + (i % 2) * 1.0
                    y = i * 1.732
                    rows.append([x, y])
            centers = np.array(rows[:n])
        elif trial == 2:
            # Ring with two layers
            angles_init = np.linspace(0, 2*np.pi, 6, endpoint=False)
            centers1 = np.column_stack([1.5 * np.cos(angles_init), 1.5 * np.sin(angles_init)])
            angles_init2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
            centers2 = np.column_stack([3.0 * np.cos(angles_init2), 3.0 * np.sin(angles_init2)])
            centers = np.vstack([centers1, centers2])
        else:
            # Random
            angles_init = np.random.uniform(0, 2*np.pi, n)
            r_init = np.random.uniform(1.0, 3.0, n)
            centers = np.column_stack([r_init * np.cos(angles_init), r_init * np.sin(angles_init)])
        
        theta_outer_init = np.random.uniform(0, np.pi/3)
        x0 = np.concatenate([centers.flatten(), [theta_outer_init]])
        
        # Optimize
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    
    # Extract results
    centers = best_x[:2*n].reshape(n, 2)
    theta_outer = best_x[2*n]
    theta_outer_deg = np.degrees(theta_outer)
    
    # Convert to the required format
    inner = []
    for i in range(n):
        x, y = centers[i]
        inner.append([float(x), float(y), 0.0])  # Inner hexagons don't need rotation for the constraint
    
    outer_center = (0.0, 0.0)
    side_length = float(best_val)
    
    return inner, outer_center, side_length, float(theta_outer_deg)
# EVOLVE-BLOCK-END
