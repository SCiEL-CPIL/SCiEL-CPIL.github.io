"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def _log_energy(pts):
    """Compute log energy sum_{i<j} log(1/d_ij) = -sum log(d_ij).
    Uses chordal distances. Returns (energy, valid_flag)."""
    # Compute all pairwise distances
    # pts shape (N, 3)
    # dist^2[i,j] = |x_i - x_j|^2
    # To save memory for N=150, this is fine: 150x150 = 22500 floats
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Only consider i < j
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    
    # Check for points too close
    if np.any(d_sq < 1e-18):  # distance < 1e-9 means d_sq < 1e-18
        return np.inf, False
    
    d = np.sqrt(d_sq)
    # energy = sum log(1/d) = -sum log(d)
    energy = -np.sum(np.log(d))
    return energy, True


def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. points on sphere.
    Gradient of -log(d_ij) w.r.t. x_i = (x_i - x_j) / d_ij^2
    But we need to project onto tangent space of sphere.
    
    Actually, for optimization on the sphere, we can use the gradient
    of the energy in R^3 and project to tangent space, or use a
    parameterization. Here we use the R^3 gradient projected to tangent.
    
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = -sum_{j!=i} d/dx_i log(|x_i - x_j|) = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    
    For the sphere constraint, the effective gradient is the projection of
    this onto the tangent space: grad_tangent = grad - (grad . x_i) x_i
    """
    N_pts = len(pts)
    grad = np.zeros_like(pts)
    
    for i in range(N_pts):
        for j in range(i + 1, N_pts):
            d = pts[i] - pts[j]
            dist_sq = np.dot(d, d)
            if dist_sq < 1e-30:
                return None  # invalid
            # Contribution to grad[i]: -d/dist_sq (from -log(dist) derivative)
            # Wait: E = -sum log(d_ij), dE/dx_i = -d/dx_i log(d_ij) = -(x_i - x_j)/d_ij^2
            # So grad[i] += -(x_i - x_j)/d_ij^2 = (x_j - x_i)/d_ij^2
            # grad[j] += (x_i - x_j)/d_ij^2
            factor = 1.0 / dist_sq
            grad[i] += -d * factor  # -(x_i - x_j)/d^2
            grad[j] += d * factor   # (x_i - x_j)/d^2
    
    # Project to tangent space of sphere
    for i in range(N_pts):
        r = np.dot(grad[i], pts[i])
        grad[i] -= r * pts[i]
    
    return grad


def _log_energy_grad_vec(pts):
    """Vectorized gradient computation."""
    N_pts = len(pts)
    grad = np.zeros_like(pts)
    
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    dist_sq[dist_sq < 1e-30] = 1e-30
    
    # factor[i,j] = 1/dist_sq[i,j] for i != j, 0 for i == j
    factor = np.zeros_like(dist_sq)
    mask = ~np.eye(N_pts, dtype=bool)
    factor[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = sum_j factor[i,j] * (-(x_i - x_j)) for j != i
    # grad[i] = -sum_j factor[i,j] * diff[i,j]
    grad = -np.sum(diff * factor[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    r = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= r * pts
    
    return grad


def _energy_and_grad(pts):
    """Compute energy and gradient. Returns (energy, grad) or (inf, None)."""
    N_pts = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    iu = np.triu_indices(N_pts, k=1)
    d_sq = dist_sq[iu]
    
    if np.any(d_sq < 1e-18):
        return np.inf, None
    
    d = np.sqrt(d_sq)
    energy = -np.sum(np.log(d))
    
    # Vectorized gradient
    dist_sq_safe = dist_sq.copy()
    dist_sq_safe[dist_sq_safe < 1e-30] = 1e-30
    factor = np.zeros_like(dist_sq_safe)
    mask = ~np.eye(N_pts, dtype=bool)
    factor[mask] = 1.0 / dist_sq_safe[mask]
    grad = -np.sum(diff * factor[:, :, np.newaxis], axis=1)
    
    # Project to tangent
    r = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= r * pts
    
    return energy, grad


def _fibonacci_spiral(N_pts):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N_pts) + 0.5
    z = 1.0 - 2.0 * i / N_pts
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Try multiple random restarts with optimization
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = best_pts.copy()
        else:
            # Generate random points on sphere
            pts = rng.normal(size=(N, 3))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts /= norms
        
        # Optimize using L-BFGS-B in R^3, then project to sphere
        # Actually, better to use a sphere-aware optimization
        
        # Use scipy minimize with the gradient projected to tangent
        # We'll optimize in R^3 and project back to sphere after each step
        # Or use the fact that the gradient on the sphere is the projected gradient
        
        def objective(x_flat):
            pts = x_flat.reshape(N, 3)
            # Project to sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms
            e, g = _energy_and_grad(pts)
            if g is None:
                return 1e10, None
            # The gradient w.r.t. the unconstrained x is related to the
            # tangent gradient. For optimization in R^3 with projection,
            # we just use the projected gradient as the gradient.
            return e, g.flatten()
        
        x0 = pts.flatten()
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': 200, 'gtol': 1e-8,
                                      'maxls': 50})
            if result.success or result.fun < best_energy:
                new_pts = result.x.reshape(N, 3)
                # Project to sphere
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                new_pts = new_pts / norms
                e, _ = _energy_and_grad(new_pts)
                if e < best_energy and e < np.inf:
                    best_energy = e
                    best_pts = new_pts
        except Exception:
            pass
        
        # Also try a few more iterations with tighter tolerance if time permits
        if time.time() < deadline - 10:
            try:
                def objective2(x_flat):
                    pts = x_flat.reshape(N, 3)
                    norms = np.linalg.norm(pts, axis=1, keepdims=True)
                    norms[norms < 1e-10] = 1.0
                    pts = pts / norms
                    e, g = _energy_and_grad(pts)
                    if g is None:
                        return 1e10, None
                    return e, g.flatten()
                
                result2 = minimize(objective2, best_pts.flatten(), method='L-BFGS-B',
                                  options={'maxiter': 300, 'gtol': 1e-10,
                                           'maxls': 50})
                if result2.fun < best_energy:
                    new_pts = result2.x.reshape(N, 3)
                    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                    norms[norms < 1e-10] = 1.0
                    new_pts = new_pts / norms
                    e, _ = _energy_and_grad(new_pts)
                    if e < best_energy and e < np.inf:
                        best_energy = e
                        best_pts = new_pts
            except Exception:
                pass
    
    # Final polish: use the best points found
    # Do one more optimization pass with very tight tolerance
    if time.time() < deadline - 5:
        try:
            def objective3(x_flat):
                pts = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                pts = pts / norms
                e, g = _energy_and_grad(pts)
                if g is None:
                    return 1e10, None
                return e, g.flatten()
            
            result3 = minimize(objective3, best_pts.flatten(), method='L-BFGS-B',
                              options={'maxiter': 500, 'gtol': 1e-12,
                                       'maxls': 50})
            if result3.fun < best_energy:
                new_pts = result3.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                new_pts = new_pts / norms
                e, _ = _energy_and_grad(new_pts)
                if e < best_energy and e < np.inf:
                    best_energy = e
                    best_pts = new_pts
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    best_pts = best_pts / norms
    
    # Verify validity
    e, valid = _log_energy(best_pts)
    if not valid or not np.isfinite(e):
        # Fall back to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
