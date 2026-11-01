import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # seconds, safety margin for 150s limit

def _fibonacci_start():
    """Golden-angle Fibonacci spiral on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_start(dim=3, n=N):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, dim))
    bases = [2, 3, 5]
    for i in range(n):
        for j, base in enumerate(bases):
            pts[i, j] = halton(i + 1, base)
    
    # Map [0,1)^3 to sphere using spherical coordinates
    theta = 2 * np.pi * pts[:, 0]
    cos_phi = 1 - 2 * pts[:, 1]
    sin_phi = np.sqrt(np.maximum(0, 1 - cos_phi**2))
    
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = cos_phi
    
    return np.column_stack([x, y, z])

def _random_start(seed):
    """Random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _log_energy_fsum(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """
    Compute the energy and the Euclidean gradient.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    energy_terms = -np.log(dist)
    np.fill_diagonal(energy_terms, 0.0)
    energy = 0.5 * np.sum(energy_terms)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    forces = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, forces

def _riemannian_optimize(pts0, max_iter=500, tol=1e-10):
    """
    Riemannian Steepest Descent with backtracking line search.
    """
    pts = pts0.copy()
    energy, grad_euclid = _energy_and_grad(pts)
    
    dot = np.sum(grad_euclid * pts, axis=1)
    grad_riem = grad_euclid - dot[:, np.newaxis] * pts
    
    best_energy = energy
    best_pts = pts.copy()
    
    for k in range(max_iter):
        if np.linalg.norm(grad_riem) < tol:
            break
        
        d = -grad_riem
        
        alpha = 1.0
        for _ in range(20):
            v = alpha * d
            norm_d = np.linalg.norm(v, axis=1)
            safe_norm = np.where(norm_d < 1e-12, 1e-12, norm_d)
            cos_term = np.cos(safe_norm)
            sin_term = np.sin(safe_norm) / safe_norm
            
            new_pts = pts * cos_term[:, np.newaxis] + v * sin_term[:, np.newaxis]
            
            new_pts_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.maximum(new_pts_norms, 1e-15)
            
            new_energy, _ = _energy_and_grad(new_pts)
            
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                if energy < best_energy:
                    best_energy = energy
                    best_pts = pts.copy()
                break
            else:
                alpha *= 0.5
                if alpha < 1e-10:
                    break
        else:
            break
        
        _, grad_euclid = _energy_and_grad(pts)
        dot = np.sum(grad_euclid * pts, axis=1)
        grad_riem = grad_euclid - dot[:, np.newaxis] * pts
        
    return best_pts, best_energy

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Multiple diverse starting points
    starts = []
    
    # Fibonacci spiral
    pts_fib = _fibonacci_start()
    starts.append(pts_fib)
    
    # Halton sequence
    pts_halton = _halton_start()
    starts.append(pts_halton)
    
    # Random starts
    for seed in [1, 42, 123, 456]:
        pts_rand = _random_start(seed)
        starts.append(pts_rand)
    
    best_pts = starts[0].copy()
    best_energy = _log_energy_fsum(starts[0])
    
    # Phase 1: Optimize from each diverse start
    for i, pts_start in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 15:
            break
        
        # Give more iterations to first few starts
        if i < 2:
            max_iter = 500
        else:
            max_iter = 300
        
        pts_opt, e_opt = _riemannian_optimize(pts_start, max_iter=max_iter, tol=1e-10)
        
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            e_precise = _log_energy_fsum(pts_opt)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = pts_opt.copy()
    
    # Phase 2: Additional random restarts if time permits
    for i in range(10):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 15:
            break
        
        pts_rand = _random_start(1000 + i)
        pts_new, e_new = _riemannian_optimize(pts_rand, max_iter=200, tol=1e-9)
        
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            e_precise = _log_energy_fsum(pts_new)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = pts_new.copy()
    
    # Phase 3: Final high-precision polish on the best configuration
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        pts_polished, e_polished = _riemannian_optimize(best_pts, max_iter=1000, tol=1e-12)
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 5:
            e_final = _log_energy_fsum(pts_polished)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = pts_polished.copy()
    
    # Final verification
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        final_e = _log_energy_fsum(best_pts)
        if final_e < best_energy:
            best_energy = final_e
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or np.any(~np.isfinite(best_pts)):
        return _fibonacci_start()
    
    return _project_to_sphere(best_pts)

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms
