"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

# Global deadline
_deadline = 0.0

def _log_energy_fast(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    # dist^2 = 2 - 2*dot
    dot = pts @ pts.T
    dist_sq = np.maximum(2.0 - 2.0 * dot, 1e-15)
    np.fill_diagonal(dist_sq, 1.0)
    # We need sum of log(1/sqrt(dist_sq)) = -0.5 * sum(log(dist_sq))
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    # Check for invalid
    if np.any(d_sq < 1e-18):
        return float('inf')
    terms = -0.5 * np.log(d_sq)
    return math.fsum(terms.tolist())

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_newton(pts, max_iters=200, time_limit=None):
    """
    Optimize points on sphere using Newton's method on the tangent space.
    Energy E = sum_{i<j} -log(|xi-xj|)
    Gradient g_i = sum_{j!=i} (xi - xj) / |xi - xj|^2
    Hessian H is complex, we approximate using the Hessian of the energy on R3 
    projected to tangent space.
    
    Hessian of E wrt xi:
    d^2 E / dxi dxj = 2 * ( (xi-xj)(xi-xj)^T / |xi-xj|^4 - I / |xi-xj|^2 ) for i!=j
    d^2 E / dxi dxi = sum_{j!=i} ( 3 (xi-xj)(xi-xj)^T / |xi-xj|^4 - I / |xi-xj|^2 )
    
    This is expensive to compute fully (N x 3 x N x 3). 
    Instead, we use a quasi-Newton or a simplified Newton step.
    Given N=150, N^2 = 22500 pairs. 
    Computing full Hessian is 150*3 x 150*3 = 450x450 matrix. 
    Filling it requires iterating over pairs. 
    Let's try to compute the gradient and a diagonal approximation or use L-BFGS from scipy if available, 
    but we must stick to standard libs + numpy + scipy. 
    The prompt says SciPy is available. 
    Let's use scipy.optimize.minimize with L-BFGS-B or BFGS on the sphere? 
    Constraint: xi . xi = 1. 
    Standard scipy minimize doesn't handle manifold constraints well without penalties.
    
    Better: Use the "Newton step on tangent space" approach but compute the Hessian-vector product efficiently?
    Or just use a very well-tuned Gradient Descent with adaptive step size and vectorized ops.
    The previous program's gradient computation was the bottleneck (python loop).
    Let's vectorize the gradient.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    # Precompute pairwise differences and distances
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    # This is (N, N, 3) which is 150*150*3 = 67500 floats. Small enough.
    
    best_energy = _log_energy_fast(pts)
    best_pts = pts.copy()
    
    # We will use a vectorized gradient descent with adaptive learning rate
    # and line search.
    
    for it in range(max_iters):
        if time_limit and time.time() > time_limit:
            break
            
        # 1. Compute Gradient
        # grad[i] = sum_{j != i} (xi - xj) / |xi - xj|^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
        dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # weights[i, j] = 1 / dist_sq[i, j]
        weights = 1.0 / dist_sq
        np.fill_diagonal(weights, 0.0)
        
        # grad[i] = sum_j weights[i, j] * diff[i, j]
        grad = np.einsum('ij,ijk->ik', weights, diff)
        
        # 2. Project Gradient to Tangent Space
        # g_t = g - (g . xi) xi
        proj_factor = np.sum(grad * pts, axis=1, keepdims=True)
        grad_t = grad - proj_factor * pts
        
        # 3. Adaptive Step Size
        # Estimate Lipschitz constant or use fixed decay
        # Let's try a simple adaptive scheme: 
        # Start with lr = 1e-3, decrease if energy increases.
        
        # For robustness, let's use a simple line search on the step size.
        # Try steps: lr * 0.5^k
        current_energy = best_energy # Use current best to decide if we accept
        
        # Actually, we should track the *current* state's energy, not just best.
        # But _log_energy_fast is somewhat expensive (N^2 log). 
        # Let's assume we compute it every iteration.
        
        curr_energy = _log_energy_fast(pts)
        
        # Newton-like step? 
        # Approximate Hessian eigenvalue for step size.
        # For now, let's use a robust GD with backtracking line search.
        
        lr = 1e-3
        accepted = False
        for k in range(20):
            step = lr * (0.5 ** k)
            new_pts = pts - step * grad_t
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / norms
            
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < curr_energy:
                pts = new_pts
                curr_energy = new_energy
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_pts = pts.copy()
                accepted = True
                break
            else:
                # If we can't even take a small step, we might be at a minimum or stuck
                if k == 19:
                    break
        
        if not accepted:
            # Stuck, try random perturbation?
            # Or break if we are very close to convergence
            if np.linalg.norm(grad_t) < 1e-8:
                break
            
    return best_pts, best_energy

def construct():
    global _deadline
    _deadline = time.time() + 115.0  # Safety margin
    
    # 1. Initialize with Fibonacci
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy_fast(best_pts)
    
    # 2. Optimize from this start
    # We have ~115s. 
    # Vectorized GD is fast. Let's allow many iterations.
    opt_pts, opt_energy = _optimize_newton(best_pts, max_iters=5000, time_limit=_deadline - 5.0)
    if opt_energy < best_energy:
        best_energy = opt_energy
        best_pts = opt_pts
    
    # 3. Try a few other starts if time permits
    # Random restarts are less effective for log energy than starting from a good symmetric distribution,
    # but let's try one random uniform restart to see if it beats Fibonacci.
    if time.time() < _deadline - 10.0:
        np.random.seed(123)
        # Simple random uniform
        theta = np.random.uniform(0, 2 * np.pi, N)
        phi = np.arccos(2 * np.random.uniform(0, 1, N) - 1)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        rand_pts = np.column_stack([x, y, z])
        
        opt_pts2, opt_energy2 = _optimize_newton(rand_pts, max_iters=2000, time_limit=_deadline - 5.0)
        if opt_energy2 < best_energy:
            best_energy = opt_energy2
            best_pts = opt_pts2
            
    # 4. Final polish on the best solution
    if time.time() < _deadline - 2.0:
        opt_pts3, opt_energy3 = _optimize_newton(best_pts, max_iters=2000, time_limit=_deadline - 1.0)
        if opt_energy3 < best_energy:
            best_energy = opt_energy3
            best_pts = opt_pts3
            
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
