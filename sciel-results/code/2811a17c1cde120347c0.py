"""
Combined program: Simulated Annealing and L-BFGS-B optimization
for minimal logarithmic energy of 150 points on the unit sphere.

Strategy:
1. Initialize with a well-distributed deterministic sequence (Halton).
2. Use Simulated Annealing to escape local minima (structural change from pure gradient descent).
3. Polish the result with L-BFGS-B for high-precision convergence.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0  # seconds before the deadline to stop and return best


def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) and its gradient.
    Vectorized implementation.
    """
    n = pts.shape[0]
    
    # Compute pairwise differences and squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient calculation:
    # grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # = (inv_d2 @ pts)_i - x_i * (sum(inv_d2, axis=1))_i
    
    weighted_sum_pts = inv_d2 @ pts  # (n, 3)
    sum_inv_d2 = np.sum(inv_d2, axis=1)  # (n,)
    grad = weighted_sum_pts - pts * sum_inv_d2[:, np.newaxis]
    
    # Project gradient to tangent space: g_tangent = g - (g . x) * x
    dot_prod = np.sum(grad * pts, axis=1)  # (n,)
    grad = grad - dot_prod[:, np.newaxis] * pts
    
    # Energy calculation: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)  # Prevent log(0)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed (no gradient)."""
    n = pts.shape[0]
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        u3 = halton(i, base_start + 2)
        
        # Map to sphere using spherical coordinates
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _simulated_annealing(pts, time_limit=80.0, initial_temp=1.0, cooling_rate=0.9999, target_acceptance=0.5):
    """
    Simulated Annealing to escape local minima.
    Uses random perturbations and Metropolis criterion.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy, _ = _energy_and_grad(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    temperature = initial_temp
    step_size = 0.05
    rng = np.random.default_rng(42)
    
    # Track acceptance rate for adaptive step size
    recent_acceptances = []
    
    while time.time() - start_time < time_limit:
        # Generate a random perturbation
        # Add random noise in the ambient space, then project back
        # This effectively samples the sphere locally
        noise = rng.normal(0, step_size, size=current_pts.shape)
        proposed_pts = _normalize(current_pts + noise)
        
        proposed_energy, _ = _energy_and_grad(proposed_pts)
        
        delta_e = proposed_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
            # Accept move
            current_pts = proposed_pts
            current_energy = proposed_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
            
            recent_acceptances.append(1)
        else:
            recent_acceptances.append(0)
        
        # Cool down
        temperature *= cooling_rate
        
        # Adaptive step size adjustment every 100 iterations
        if len(recent_acceptances) >= 100:
            accept_rate = sum(recent_acceptances[-100:]) / 100.0
            if accept_rate > target_acceptance + 0.1:
                step_size *= 1.1
            elif accept_rate < target_acceptance - 0.1:
                step_size *= 0.9
            recent_acceptances = []
            
            # Ensure step size stays in reasonable bounds
            step_size = np.clip(step_size, 0.001, 0.5)
            
    return best_pts, best_energy


def _lbfgs_optimize(pts, time_limit=30.0):
    """
    Optimize using L-BFGS-B with projected gradient.
    This is a second-order method that is robust for initial convergence.
    """
    start = time.time()
    
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxfun': 10000, 'maxls': 50})
        
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        energy = _log_energy_fast(x_opt)
        
        return x_opt, energy
    except Exception:
        return pts, _log_energy_fast(pts)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    best_pts = None
    best_energy = float('inf')
    
    # Initializations
    # Halton is generally more uniform than Fibonacci for this purpose
    pts_init = _halton_sphere(N)
    
    # Phase 1: Simulated Annealing for global exploration
    # Allocate ~75% of the time to SA
    sa_time_limit = min(80.0, (deadline - start_time) * 0.75)
    
    if sa_time_limit > 5.0:
        try:
            pts_sa, e_sa = _simulated_annealing(pts_init, time_limit=sa_time_limit)
            if e_sa < best_energy:
                best_energy = e_sa
                best_pts = pts_sa.copy()
        except Exception:
            best_pts = pts_init.copy()
            best_energy = _log_energy_fast(best_pts)
    else:
        best_pts = pts_init.copy()
        best_energy = _log_energy_fast(best_pts)
        
    # Phase 2: L-BFGS-B for local refinement
    time_left = deadline - time.time()
    if time_left > 5.0 and best_pts is not None:
        try:
            pts_opt, e_opt = _lbfgs_optimize(best_pts, time_limit=min(time_left, 30.0))
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        best_pts = _halton_sphere(N)
    
    # Final projection and validation
    best_pts = _normalize(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                rng = np.random.default_rng(i+j)
                best_pts[j] += 1e-8 * rng.randn(3)
                best_pts[j] = _normalize(best_pts[j])
    
    return best_pts
