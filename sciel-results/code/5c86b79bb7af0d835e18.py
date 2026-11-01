import numpy as np
import time
import math
from scipy.spatial.distance import pdist

N = 150

def _halton_init(n, dims=3):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = np.zeros((n, dims))
    for d in range(dims):
        base = d + 2  # bases: 2, 3, 5
        for i in range(n):
            points[i, d] = halton(i + 1, base)
    
    # Map [0,1]^3 to sphere using inverse CDF of uniform on sphere
    u = points[:, 0]
    v = points[:, 1]
    z = 2.0 * u - 1.0
    theta = 2.0 * np.pi * v
    r = np.sqrt(1.0 - z * z)
    points_sphere = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return points_sphere

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = pdist(pts)
    d = np.clip(d, 1e-15, None)
    return math.fsum(math.log(1.0 / di) for di in d)

def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its gradient."""
    n = pts.shape[0]
    energy = 0.0
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            d = max(d, 1e-15)
            log_term = math.log(1.0 / d)
            energy += log_term
            # d/dx_i log(1/|x_i - x_j|) = - (x_i - x_j) / |x_i - x_j|^2
            coeff = 1.0 / d
            grad[i] += diff * coeff
            grad[j] -= diff * coeff
    
    return energy, grad

def _project_tangent(pts, grad):
    """Project gradient onto tangent space of the sphere at each point."""
    # For a point x on unit sphere, tangent projection: v - (v·x)x
    proj = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    return proj

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-15, None)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 5s safety margin before 140s kill
    
    # Initialize with Halton sequence
    pts = _halton_init(N)
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Riemannian L-BFGS-B style optimization with penalty for minimum separation
    # We use a simple gradient descent with line search on the sphere
    
    def _energy_with_penalty(pts, min_sep=0.01, penalty_coeff=100.0):
        """Energy with penalty for pairs closer than min_sep."""
        e, g = _log_energy_and_grad(pts)
        # Add penalty for close pairs
        d = pdist(pts)
        close_mask = d < min_sep
        if np.any(close_mask):
            # Penalty term: sum over close pairs of (min_sep - d)^2
            close_d = d[close_mask]
            penalty = penalty_coeff * np.sum((min_sep - close_d) ** 2)
            # Gradient of penalty: for each close pair (i,j), 
            # d/dx_i (min_sep - |x_i - x_j|)^2 = 2*(min_sep - d) * (x_i - x_j)/d
            # We need to map the flat distance indices back to pairs
            n_pairs = N * (N - 1) // 2
            pair_idx = 0
            penalty_grad = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    if close_mask[pair_idx]:
                        diff = pts[i] - pts[j]
                        d_ij = np.linalg.norm(diff)
                        d_ij = max(d_ij, 1e-15)
                        coeff = 2.0 * penalty_coeff * (min_sep - d_ij) / d_ij
                        penalty_grad[i] += diff * coeff
                        penalty_grad[j] -= diff * coeff
                    pair_idx += 1
            return e + penalty, g + penalty_grad
        return e, g
    
    # Simple gradient descent with adaptive step size
    lr = 0.01
    min_lr = 1e-8
    max_iterations = 5000
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        energy, grad = _energy_with_penalty(pts, min_sep=0.01, penalty_coeff=50.0)
        
        # Project gradient to tangent space
        tangent_grad = _project_tangent(pts, grad)
        
        # Update
        new_pts = pts - lr * tangent_grad
        new_pts = _normalize(new_pts)
        
        # Check for invalid (non-finite)
        if not np.all(np.isfinite(new_pts)):
            lr *= 0.5
            if lr < min_lr:
                break
            continue
        
        new_energy = _log_energy(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            # Increase learning rate slightly
            lr = min(lr * 1.01, 0.1)
        else:
            # Decrease learning rate
            lr *= 0.7
            if lr < min_lr:
                break
        
        # Update best
        current_energy = _log_energy(pts)
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = pts.copy()
    
    # Final polish with smaller penalty
    if time.time() < deadline - 5:
        pts = best_pts.copy()
        lr = 1e-4
        for it in range(2000):
            if time.time() > deadline:
                break
            energy, grad = _log_energy_and_grad(pts)
            tangent_grad = _project_tangent(pts, grad)
            new_pts = pts - lr * tangent_grad
            new_pts = _normalize(new_pts)
            if not np.all(np.isfinite(new_pts)):
                lr *= 0.5
                if lr < 1e-10:
                    break
                continue
            new_energy = _log_energy(new_pts)
            if new_energy < energy:
                pts = new_pts
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = pts.copy()
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
    
    # Ensure minimum separation
    best_pts = _normalize(best_pts)
    return best_pts
