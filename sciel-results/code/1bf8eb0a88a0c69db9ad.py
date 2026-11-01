"""Rewritten program: BFGS-style optimization of logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0  # seconds before the deadline to stop and return best


def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient with respect to the points (as a 150x3 array).
    
    For a pair (i, j), the contribution to energy is -log(d_ij) where d_ij = |x_i - x_j|.
    dE/dx_i = (x_i - x_j) / d_ij^2   (from -log(d_ij), d(-log d)/dx_i = -1/d * d(d)/dx_i = -1/d * (x_i-x_j)/d = -(x_i-x_j)/d^2)
    Wait: let me re-derive.
    
    E = -sum_{i<j} log(d_ij)
    d/dx_i [ -log(d_ij) ] = -(1/d_ij) * d(d_ij)/dx_i = -(1/d_ij) * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    
    So grad_i = sum_{j != i} -(x_i - x_j)/d_ij^2 = sum_{j != i} (x_j - x_i)/d_ij^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    dist = np.sqrt(np.maximum(dist2, 1e-30))  # (n, n)
    
    # Energy: sum over i<j of -log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
    # = -sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: (x_i - x_j)/d_ij^2 = diff[i,j] / dist2[i,j]
    # So grad_i = -sum_j diff[i,j]/dist2[i,j] for j != i
    
    # diff[i,j] = x_i - x_j
    # We want sum_{j != i} (x_j - x_i)/d_ij^2 = -sum_{j != i} (x_i - x_j)/d_ij^2 = -sum_{j != i} diff[i,j]/dist2[i,j]
    
    # Build matrix: M[i,j] = diff[i,j] / dist2[i,j] for i != j, 0 for i == j
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist2 = np.zeros_like(dist2)
        mask = dist2 > 1e-30
        inv_dist2[mask] = 1.0 / dist2[mask]
    
    # grad_i = -sum_j diff[i,j] * inv_dist2[i,j]
    # diff is (n, n, 3), inv_dist2 is (n, n)
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize each point to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _halton_init(n, base1=2, base2=3, base3=5):
    """Generate n points on the unit sphere using a 3D Halton sequence mapped to sphere."""
    def halton(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        u = halton(i, base1)
        v = halton(i, base2)
        w = halton(i, base3)
        
        # Map to sphere: use spherical coordinates with adjusted parameters
        # theta = arccos(1 - 2*v), phi = 2*pi*u
        theta = math.acos(1.0 - 2.0 * v)
        phi = 2.0 * math.pi * u
        
        points[i-1, 0] = math.sin(theta) * math.cos(phi)
        points[i-1, 1] = math.sin(theta) * math.sin(phi)
        points[i-1, 2] = math.cos(theta)
    
    return _project_to_sphere(points)


def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    best_pts = None
    best_energy = float('inf')
    
    # Try multiple initializations
    initializations = []
    
    # Fibonacci spiral
    pts_fib = _fibonacci_init(N)
    initializations.append(pts_fib)
    
    # Halton sequence
    pts_halton = _halton_init(N)
    initializations.append(pts_halton)
    
    # Random initialization
    np.random.seed(42)
    pts_rand = np.random.randn(N, 3)
    pts_rand = _project_to_sphere(pts_rand)
    initializations.append(pts_rand)
    
    for init_pts in initializations:
        if time.time() > deadline:
            break
        
        # Evaluate initial energy
        e0, g0 = _energy_and_grad(init_pts)
        
        if e0 < best_energy:
            best_energy = e0
            best_pts = init_pts.copy()
        
        # Run BFGS optimization
        def objective(x_flat):
            pts = x_flat.reshape(N, 3)
            pts = _project_to_sphere(pts)
            e, _ = _energy_and_grad(pts)
            return e
        
        def grad_func(x_flat):
            pts = x_flat.reshape(N, 3)
            pts = _project_to_sphere(pts)
            _, g = _energy_and_grad(pts)
            # Project gradient to tangent space: g_tangent = g - (g . x) * x
            g_flat = g.reshape(-1)
            # For each point i: g_i_tangent = g_i - (g_i . x_i) * x_i
            pts_flat = pts.reshape(-1)
            dots = np.sum(g_flat * pts_flat.reshape(N, 3), axis=1)  # (N,)
            g_tangent = g - dots[:, np.newaxis] * pts
            return g_tangent.reshape(-1)
        
        x0 = init_pts.reshape(-1)
        
        try:
            result = minimize(
                objective,
                x0,
                jac=grad_func,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-8, 'disp': False}
            )
            
            if result.success or result.fun < best_energy:
                pts_opt = _project_to_sphere(result.x.reshape(N, 3))
                e_opt, _ = _energy_and_grad(pts_opt)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = pts_opt.copy()
        except Exception:
            pass
        
        # If we have time, do a second round of optimization with tighter tolerance
        if time.time() < deadline and best_pts is not None:
            try:
                x0 = best_pts.reshape(-1)
                result2 = minimize(
                    objective,
                    x0,
                    jac=grad_func,
                    method='BFGS',
                    options={'maxiter': 1000, 'gtol': 1e-10, 'disp': False}
                )
                if result2.fun < best_energy:
                    pts_opt2 = _project_to_sphere(result2.x.reshape(N, 3))
                    e_opt2, _ = _energy_and_grad(pts_opt2)
                    if e_opt2 < best_energy:
                        best_energy = e_opt2
                        best_pts = pts_opt2.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_init(N)
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                best_pts[j] += 1e-8 * np.random.randn(3)
                best_pts[j] = _project_to_sphere(best_pts[j])
    
    return best_pts
