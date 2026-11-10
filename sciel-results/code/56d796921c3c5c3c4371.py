"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(pts):
    """Compute the packing radius (min of half-min-distance and min-distance-to-boundary)."""
    pts = np.asarray(pts)
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Distance to boundary
    radii = np.sqrt(np.sum(pts**2, axis=-1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)

def _energy_and_grad(pts, r_target):
    """
    Energy function: sum of violations.
    For each pair i<j: if dist < 2*r_target, add (2*r_target - dist)^2
    For each i: if radius > 1 - r_target, add (radius - (1 - r_target))^2
    """
    pts = np.asarray(pts)
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i+1, n):
            d = pts[i] - pts[j]
            dist = np.sqrt(np.dot(d, d))
            if dist < 1e-12:
                # Perturb slightly
                dist = 1e-12
                d = np.array([1e-12, 0.0])
            if dist < 2.0 * r_target:
                violation = 2.0 * r_target - dist
                energy += violation**2
                # Gradient: d/d(pts[i]) of (2r - dist)^2 = -2*(2r-dist) * d(dist)/d(pts[i])
                # d(dist)/d(pts[i]) = -d/dist (where d = pts[i]-pts[j])
                # So gradient contribution = -2*violation * (-d/dist) = 2*violation*d/dist
                factor = 2.0 * violation / dist
                grad[i] += factor * d
                grad[j] -= factor * d
    
    # Boundary constraints
    for i in range(n):
        radius = np.sqrt(np.dot(pts[i], pts[i]))
        if radius < 1e-12:
            continue
        if radius > 1.0 - r_target:
            violation = radius - (1.0 - r_target)
            energy += violation**2
            # d/d(pts[i]) of (radius - (1-r))^2 = -2*(radius-(1-r)) * d(radius)/d(pts[i])
            # d(radius)/d(pts[i]) = pts[i]/radius
            # So gradient = -2*violation * pts[i]/radius
            factor = -2.0 * violation / radius
            grad[i] += factor * pts[i]
    
    return energy, grad

def _hexagonal_initial(n):
    """Generate hexagonal lattice points inside unit circle."""
    points = []
    # Start with a reasonable spacing
    for spacing in np.linspace(0.2, 0.05, 30):
        k = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k, k+1):
            for i in range(-k, k+1):
                x = (i + 0.5*(j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x*x + y*y) <= 1.0:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= n:
            # Sort by radius and take the innermost n
            radii = np.sqrt(np.sum(pts**2, axis=1))
            idx = np.argsort(radii, kind='stable')[:n]
            return pts[idx]
    # Fallback: random
    return np.random.randn(n, 2) * 0.8

def _optimize_from_initial(pts_init, r_target, max_iter=5000):
    """Run optimization to find a valid configuration with radius >= r_target."""
    x0 = pts_init.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        e, g = _energy_and_grad(pts, r_target)
        return e, g.flatten()
    
    res = minimize(objective, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
    pts = res.x.reshape(-1, 2)
    return pts, res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with hexagonal initial
    pts_init = _hexagonal_initial(N)
    
    # Binary search on radius
    r_low = 0.1
    r_high = 0.14
    
    # First, find a valid radius
    best_pts = pts_init.copy()
    best_r = _score(best_pts)
    
    # Try to improve using optimization
    while time.time() < deadline:
        # Binary search for max feasible radius
        r_mid = (r_low + r_high) / 2.0
        
        if time.time() > deadline - 30:
            break
        
        # Try to find a configuration with radius >= r_mid
        pts_opt, energy = _optimize_from_initial(pts_init, r_mid, max_iter=2000)
        
        actual_r = _score(pts_opt)
        if actual_r >= r_mid - 1e-8:
            r_low = r_mid
            best_pts = pts_opt.copy()
            best_r = actual_r
        else:
            r_high = r_mid
        
        # If we've converged enough
        if r_high - r_low < 1e-6:
            break
    
    # Final polish: try to maximize radius directly
    if time.time() < deadline - 20:
        # Use the best points found and do a direct optimization
        def neg_radius(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        
        res = minimize(neg_radius, best_pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
        pts_nm = res.x.reshape(-1, 2)
        r_nm = _score(pts_nm)
        if r_nm > best_r:
            best_pts = pts_nm
            best_r = r_nm
    
    # Ensure validity
    best_pts = np.asarray(best_pts)
    # Clip to ensure inside unit circle
    radii = np.sqrt(np.sum(best_pts**2, axis=1))
    mask = radii > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / radii[mask, None] * 0.999
    
    return best_pts
# EVOLVE-BLOCK-END
