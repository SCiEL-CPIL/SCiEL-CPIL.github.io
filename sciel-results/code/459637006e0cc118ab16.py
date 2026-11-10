"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _fibonacci_sphere(n, seed=0):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Add small random perturbation if seed != 0 to get different starting points
    if seed != 0:
        pts = pts + rng.normal(0, 0.01, size=pts.shape)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
    return pts

def _compute_energy(pts):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = pts.shape[0]
    total = 0.0
    terms = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return -1e18
            terms.append(np.log(1.0 / d))
    return sum(terms)

def _compute_energy_fast(pts):
    """Compute the logarithmic energy using vectorized operations."""
    n = pts.shape[0]
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Only use upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _gradient_F(pts):
    """
    Compute gradient of F = sum_{i<j} log(d_ij) w.r.t. each point.
    grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-30)
    # grad_i = sum_j (x_i - x_j) / dist_sq_ij
    inv_dist_sq = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_dist_sq, 0.0)
    np.divide(1.0, dist_sq, out=inv_dist_sq, where=dist_sq > 1e-20)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return grad

def _project_gradient(pts, grad):
    """Project gradient onto tangent space of the sphere."""
    # Project: g - (g . x) x
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts

def _optimize_sphere(pts, max_iters=5000, tol=1e-12, deadline=None):
    """
    Gradient descent on the sphere to maximize F = sum log(d_ij).
    Returns (pts, energy, converged).
    """
    pts = pts.copy().astype(np.float64)
    n = pts.shape[0]
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Initial energy
    E = _compute_energy_fast(pts)
    best_pts = pts.copy()
    best_E = E
    
    # Adaptive step size
    alpha = 0.01
    min_alpha = 1e-10
    max_alpha = 1.0
    
    for it in range(max_iters):
        # Check deadline
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient of F
        grad_F = _gradient_F(pts)
        
        # Project onto tangent space
        grad_proj = _project_gradient(pts, grad_F)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_proj)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        improved = False
        for backtrack in range(30):
            new_pts = pts + alpha * grad_proj
            # Renormalize
            new_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / new_norms
            
            new_E = _compute_energy_fast(new_pts)
            
            if new_E < E - 1e-15:  # Energy decreased (F increased)
                E = new_E
                pts = new_pts
                if E < best_E:
                    best_E = E
                    best_pts = pts.copy()
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            # Try a very small step
            alpha = min_alpha
            new_pts = pts + alpha * grad_proj
            new_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / new_norms
            new_E = _compute_energy_fast(new_pts)
            if new_E < E:
                E = new_E
                pts = new_pts
                if E < best_E:
                    best_E = E
                    best_pts = pts.copy()
            else:
                # No improvement possible; break
                break
        
        # Occasionally increase alpha if we've been doing many backtracks
        if improved and alpha < 0.5:
            alpha = min(alpha * 1.1, max_alpha)
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts, best_E, grad_norm < tol

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_E = -1e18
    
    # Multiple restarts with different initial configurations
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            # Fibonacci spiral
            pts = _fibonacci_sphere(N, seed=0)
        elif restart == 1:
            # Fibonacci spiral with perturbation
            pts = _fibonacci_sphere(N, seed=1)
        elif restart == 2:
            # Random points
            rng = np.random.RandomState(42 + restart)
            pts = rng.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            # More perturbed Fibonacci
            pts = _fibonacci_sphere(N, seed=restart)
        
        # Optimize
        pts_opt, E_opt, converged = _optimize_sphere(
            pts, 
            max_iters=3000,
            tol=1e-14,
            deadline=deadline - 3.0
        )
        
        if E_opt < best_E:
            best_E = E_opt
            best_pts = pts_opt.copy()
        
        # If we haven't converged and have time, do a polish pass
        if not converged and time.time() < deadline - 10.0:
            pts_polish, E_polish, _ = _optimize_sphere(
                pts_opt,
                max_iters=2000,
                tol=1e-16,
                deadline=deadline - 3.0
            )
            if E_polish < best_E:
                best_E = E_polish
                best_pts = pts_polish.copy()
    
    # Final polish if we have time
    if best_pts is not None and time.time() < deadline - 5.0:
        pts_final, E_final, _ = _optimize_sphere(
            best_pts,
            max_iters=5000,
            tol=1e-16,
            deadline=deadline - 1.0
        )
        if E_final < best_E:
            best_E = E_final
            best_pts = pts_final.copy()
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, seed=0)
    
    # Final normalization and validation
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N, seed=0)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms
    
    # Check for points too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them
                best_pts[j] = best_pts[i] + 1e-8 * np.array([1.0, 0.0, 0.0])
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
