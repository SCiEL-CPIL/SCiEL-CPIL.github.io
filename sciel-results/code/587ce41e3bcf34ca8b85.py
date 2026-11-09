"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import math


def _normalize(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_dists = np.clip(pair_dists, 1e-15, None)
    return -np.sum(np.log(pair_dists))


def _log_energy_grad(pts):
    """Compute energy and gradient of log energy w.r.t. points in R^3."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    inv_d2 = 1.0 / (dists * dists)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (x_i - x_j) / d_ij^2
    grad_raw = (diff * inv_d2[:, :, np.newaxis]).sum(axis=1)
    
    # The actual gradient of E is -grad_raw
    grad_E = -grad_raw
    
    # Project gradient onto tangent space of sphere at each point
    dot = np.sum(grad_E * pts, axis=1, keepdims=True)
    grad_tangent = grad_E - dot * pts
    
    energy = _log_energy(pts)
    return energy, grad_tangent


def _generate_lattice_points(lattice_params):
    """
    Generate 150 points on the sphere using a symmetry-reduced lattice model.
    
    lattice_params: array of shape (9,) representing 3 lattice vectors (a, b, c).
    We generate a grid of points in the unit cell, map them to the sphere,
    and select the 150 best distributed points (or just use a fixed grid).
    
    For simplicity and speed, we use a cubic lattice scaled to fit the sphere,
    then project to sphere. We use a 5x5x6 grid = 150 points.
    """
    a, b, c = lattice_params[:3], lattice_params[3:6], lattice_params[6:9]
    
    # Generate grid points in [0,1]^3
    nx, ny, nz = 5, 5, 6
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Map grid to lattice space
    # point = (i/nx)*a + (j/ny)*b + (k/nz)*c
    pts = grid[:, 0:1] * a + grid[:, 1:2] * b + grid[:, 2:3] * c
    
    # Center the lattice
    center = np.mean(pts, axis=0)
    pts = pts - center
    
    # Normalize to unit sphere
    pts = _normalize(pts)
    
    return pts


def _lattice_energy_and_grad(lattice_params):
    """
    Compute energy and gradient of the lattice-based point set.
    """
    pts = _generate_lattice_points(lattice_params)
    
    # Check for very close points
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    min_dist = np.min(pair_dists)
    
    if min_dist < 1e-6:
        return 1e10, np.zeros_like(lattice_params)
    
    energy = _log_energy(pts)
    
    # Numerical gradient for lattice params
    eps = 1e-6
    grad = np.zeros_like(lattice_params)
    for i in range(9):
        lp_plus = lattice_params.copy()
        lp_plus[i] += eps
        pts_plus = _generate_lattice_points(lp_plus)
        # Check validity
        diff_plus = pts_plus[:, np.newaxis, :] - pts_plus[np.newaxis, :, :]
        dists_plus = np.linalg.norm(diff_plus, axis=2)
        np.fill_diagonal(dists_plus, 1.0)
        pair_dists_plus = dists_plus[iu]
        if np.min(pair_dists_plus) < 1e-6:
            grad[i] = 0
        else:
            energy_plus = _log_energy(pts_plus)
            grad[i] = (energy_plus - energy) / eps
    
    return energy, grad


def _fibonacci_points(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n):
    """Generate random points on the sphere."""
    pts = np.random.normal(0, 1, size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _spherical_lbfgs(pts0, max_time=5.0):
    """Optimize points on sphere using L-BFGS-B with spherical projection."""
    pts = pts0.copy()
    t_start = time.time()
    
    def objective_and_grad(x_flat):
        pts_local = x_flat.reshape(-1, 3)
        pts_local = _normalize(pts_local)
        energy, grad = _log_energy_grad(pts_local)
        grad_flat = grad.flatten()
        return energy, grad_flat
    
    try:
        res = minimize(
            objective_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            options={
                'maxiter': 100,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 300,
            }
        )
        
        if res.success or res.fun < _log_energy(pts):
            pts_opt = _normalize(res.x.reshape(-1, 3))
            energy_opt = _log_energy(pts_opt)
            return pts_opt, energy_opt
        else:
            return pts, _log_energy(pts)
    
    except Exception:
        return pts, _log_energy(pts)


def _optimize_lattice(lattice_params0, max_time=10.0):
    """
    Optimize the lattice parameters using L-BFGS-B.
    """
    t_start = time.time()
    
    def objective_and_grad(lp):
        if time.time() - t_start > max_time:
            return 1e10, np.zeros_like(lp)
        return _lattice_energy_and_grad(lp)
    
    try:
        res = minimize(
            objective_and_grad,
            lattice_params0,
            method='L-BFGS-B',
            options={
                'maxiter': 50,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'maxfun': 100,
            }
        )
        
        if res.success or res.fun < 1e9:
            pts_opt = _generate_lattice_points(res.x)
            # Validate
            diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            iu = np.triu_indices(N, k=1)
            pair_dists = dists[iu]
            if np.min(pair_dists) > 1e-9:
                energy_opt = _log_energy(pts_opt)
                return pts_opt, energy_opt
    
    except Exception:
        pass
    
    return _generate_lattice_points(lattice_params0), _log_energy(_generate_lattice_points(lattice_params0))


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Optimize lattice model
    # Initialize lattice with identity-like vectors
    np.random.seed(42)
    
    # Try multiple lattice initializations
    lattice_inits = []
    
    # 1. Identity lattice
    lattice_inits.append(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
    
    # 2. Slightly perturbed identity
    lattice_inits.append(np.array([1.0, 0.1, 0.0, -0.1, 1.0, 0.1, 0.0, -0.1, 1.0]))
    
    # 3. Random small perturbations
    for _ in range(5):
        base = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        perturbed = base + np.random.normal(0, 0.2, size=9)
        lattice_inits.append(perturbed)
    
    for i, lp in enumerate(lattice_inits):
        if time.time() > deadline - 30.0:
            break
        
        try:
            pts_opt, energy_opt = _optimize_lattice(lp, max_time=15.0)
            
            if energy_opt < best_energy:
                best_energy = energy_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Refine best lattice result with full L-BFGS
    if best_pts is not None and time.time() < deadline - 10.0:
        try:
            remaining = deadline - time.time() - 5.0
            pts_refined, energy_refined = _spherical_lbfgs(best_pts, max_time=remaining)
            if energy_refined < best_energy:
                best_energy = energy_refined
                best_pts = pts_refined
        except Exception:
            pass
    
    # Phase 3: If lattice failed or is worse than Fibonacci, try standard approach
    fib_pts = _fibonacci_points(N)
    fib_energy = _log_energy(fib_pts)
    
    if best_pts is None or best_energy > fib_energy:
        best_pts = fib_pts
        best_energy = fib_energy
        
        # Try random starts with L-BFGS
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            try:
                rand_pts = _random_points(N)
                pts_opt, energy_opt = _spherical_lbfgs(rand_pts, max_time=10.0)
                if energy_opt < best_energy:
                    best_energy = energy_opt
                    best_pts = pts_opt
            except Exception:
                pass
    
    # Final validation
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        return _fibonacci_points(N)
    
    # Ensure no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    if np.min(pair_dists) < 1e-9:
        return _fibonacci_points(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
