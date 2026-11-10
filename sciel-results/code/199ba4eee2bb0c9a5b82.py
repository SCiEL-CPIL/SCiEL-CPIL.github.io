"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(points)
    total = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return float('inf')
            total.append(math.log(1.0 / d))
    return math.fsum(total)

def _log_energy_vec(points):
    """Vectorized logarithmic energy for speed."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    # Avoid log(0)
    dist = np.maximum(dist, 1e-15)
    return np.sum(np.log(1.0 / dist))

def _log_energy_grad(points):
    """Gradient of logarithmic energy w.r.t. point positions."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/dx_i log(1/|x_i - x_j|) = -1/|x_i-x_j| * (x_i-x_j)/|x_i-x_j| = -(x_i-x_j)/|x_i-x_j|^2
            term = -diff / (d * d)
            grad[i] += term
            grad[j] -= term
    return grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_spiral(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize_from_start(points, deadline):
    """
    Optimize a set of points on the sphere using L-BFGS-B with 
    spherical parameterization.
    """
    start_time = time.time()
    
    # Use spherical coordinates: theta (polar), phi (azimuthal)
    # Point = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    
    def _cart_to_spherical(pts):
        z = pts[:, 2]
        x = pts[:, 0]
        y = pts[:, 1]
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x)
        return np.column_stack([theta, phi])
    
    def _spherical_to_cart(theta_phi):
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    def _energy_and_grad_spherical(theta_phi):
        pts = _spherical_to_cart(theta_phi)
        n = len(pts)
        energy = 0.0
        grad_sph = np.zeros_like(theta_phi)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-15:
                    continue
                # Gradient w.r.t. cartesian coords
                term = -diff / (d * d)
                
                # Chain rule: dE/dtheta_i, dE/dphi_i
                # x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
                # dx/dtheta = cos(theta)*cos(phi), dx/dphi = -sin(theta)*sin(phi)
                # dy/dtheta = cos(theta)*sin(phi), dy/dphi = sin(theta)*cos(phi)
                # dz/dtheta = -sin(theta), dz/dphi = 0
                
                for idx in [i, j]:
                    sign = 1 if idx == i else -1
                    t = theta_phi[idx, 0]
                    p = theta_phi[idx, 1]
                    st = np.sin(t)
                    ct = np.cos(t)
                    sp = np.sin(p)
                    cp = np.cos(p)
                    
                    dtheta = sign * (term[0] * ct * cp + term[1] * ct * sp + term[2] * (-st))
                    dphi = sign * (term[0] * (-st * sp) + term[1] * (st * cp))
                    
                    grad_sph[idx, 0] += dtheta
                    grad_sph[idx, 1] += dphi
        
        # Compute energy
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-15:
                    return float('inf'), grad_sph
                energy += math.log(1.0 / d)
        
        return energy, grad_sph
    
    # Initial spherical coordinates
    theta_phi = _cart_to_spherical(points)
    
    # Use L-BFGS-B
    result = minimize(
        fun=lambda x: _energy_and_grad_spherical(x)[0],
        x0=theta_phi,
        jac=lambda x: _energy_and_grad_spherical(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12}
    )
    
    if result.success or result.fun < float('inf'):
        best_sph = result.x
        best_energy = result.fun
    else:
        best_sph = theta_phi
        best_energy = _log_energy(points)
    
    # Convert back to cartesian and normalize
    best_pts = _spherical_to_cart(best_sph)
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts, best_energy

def _refine_with_repulsion(points, deadline):
    """
    Refine points using iterative repulsion method.
    Each point is pushed away from others proportional to 1/d^2.
    """
    start_time = time.time()
    n = len(points)
    pts = points.copy()
    
    for iteration in range(1000):
        if time.time() - start_time > deadline - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-15:
                    continue
                # Force = -diff / d^3 (repulsion)
                forces[i] += diff / (d * d * d)
        
        # Move points and project to sphere
        step_size = 0.01
        pts = pts + step_size * forces
        pts = _project_to_sphere(pts)
    
    return pts

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_points = None
    best_energy = float('inf')
    
    # Multiple starting points
    starts = []
    
    # Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() - start_time > deadline - 20:
            break
        starts.append(_random_points(N, seed=seed))
    
    # Optimize each start
    for idx, start_pts in enumerate(starts):
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break
        
        # First try repulsion refinement
        refined = _refine_with_repulsion(start_pts, deadline)
        energy = _log_energy(refined)
        
        if energy < best_energy:
            best_energy = energy
            best_points = refined.copy()
        
        # Then try L-BFGS-B optimization
        remaining = deadline - (time.time() - start_time)
        if remaining > 15:
            optimized, opt_energy = _optimize_from_start(refined, deadline)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_points = optimized.copy()
    
    # Final polish with more iterations if time allows
    if best_points is not None and (time.time() - start_time) < deadline - 10:
        final = _refine_with_repulsion(best_points, deadline)
        final_energy = _log_energy(final)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final
    
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Ensure all points are on unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Perturb
                best_points[j] += 1e-6 * np.random.randn(3)
                best_points = _project_to_sphere(best_points)
                break
    
    return best_points
# EVOLVE-BLOCK-END
