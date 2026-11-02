"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _covering_radius(points, grid_points):
    """Compute the covering radius in radians using a dense grid approximation."""
    # points: (N, 3), grid_points: (M, 3)
    # For each grid point, find the minimum angular distance to any of the N points.
    # Angular distance = arccos(clamp(dot, -1, 1))
    # To avoid arccos (expensive), we can work with cosines: min angle = arccos(max cos)
    # covering radius = max over grid of min angle = arccos(min over grid of max cos)
    # So we need: for each grid point g, compute max_i dot(g, p_i), then take min over g.
    
    # Efficient: compute all dot products at once
    # dots: (M, N)
    dots = grid_points @ points.T  # (M, N)
    max_dots = np.max(dots, axis=1)  # (M,)
    min_max_dot = np.min(max_dots)
    # Clamp for numerical safety
    min_max_dot = np.clip(min_max_dot, -1.0, 1.0)
    return np.arccos(min_max_dot)

def _generate_grid(n_theta=100, n_phi=200):
    """Generate a dense spherical grid for approximation."""
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate a fixed dense grid
    grid = _generate_grid(n_theta=80, n_phi=160)
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z*z, 0.0))
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def _params_to_points(params):
        """Convert parameters to points on the sphere.
        Use spherical coordinates: each point has (theta, phi).
        params: (2N,) -> (theta_0, phi_0, theta_1, phi_1, ...)
        """
        thetas = params[0::2]
        phis = params[1::2]
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        return np.column_stack([x, y, z])
    
    def _objective(params):
        pts = _params_to_points(params)
        return _covering_radius(pts, grid)
    
    def _objective_grad(params):
        """Compute gradient of the covering radius w.r.t. parameters.
        This is tricky due to the max-min structure.
        We'll use numerical differentiation for reliability, or implement analytically.
        For now, let's use a finite difference approach via scipy's minimize with BFGS
        which can handle it, but let's try to be smarter.
        
        Actually, let's just use L-BFGS-B with numerical gradients first to ensure correctness,
        then optimize if time permits.
        """
        # Use numerical gradient via central differences for robustness
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = _objective(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps
            grad[i] = (_objective(params_plus) - _objective(params_minus)) / (2*eps)
        return grad
    
    # Convert initial points to spherical coordinate parameters
    def _points_to_params(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        thetas = np.arccos(np.clip(z, -1.0, 1.0))
        phis = np.arctan2(y, x)
        # Ensure phis are in [0, 2*pi)
        phis = np.mod(phis, 2*np.pi)
        params = np.zeros(2*N)
        params[0::2] = thetas
        params[1::2] = phis
        return params
    
    params0 = _points_to_params(x0)
    
    best_params = params0.copy()
    best_score = _objective(params0)
    
    # Try multiple random restarts and local optimizations
    np.random.seed(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            params_init = params0.copy()
        else:
            # Random initialization
            thetas = np.random.uniform(0, np.pi, N)
            phis = np.random.uniform(0, 2*np.pi, N)
            params_init = np.zeros(2*N)
            params_init[0::2] = thetas
            params_init[1::2] = phis
        
        # Use L-BFGS-B with numerical gradients
        result = minimize(
            _objective,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if result.success or result.fun < best_score:
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        
        if time.time() > deadline - 20:
            break
    
    # Final polish with tighter tolerances if time permits
    if time.time() < deadline - 10:
        result = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    
    # Convert back to Cartesian points
    points = _params_to_points(best_params)
    
    # Normalize to ensure unit length
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
