import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for N points on sphere.
    Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient at point i: -sum_{j!=i} (p_i - p_j) / d_ij^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    # Energy: sum over i<j of -log(d_ij)
    iu = np.triu_indices(n, 1)
    energy = -np.sum(np.log(d[iu]))
    
    # Gradient: for each i, grad_i = -sum_{j!=i} (p_i - p_j) / d_ij^2
    # = sum_{j!=i} (p_j - p_i) / d_ij^2
    inv_d2 = np.zeros_like(d2)
    mask = d2 > 0
    inv_d2[mask] = 1.0 / d2[mask]
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j inv_d2[i,j] * (pts[j] - pts[i])
    # = sum_j inv_d2[i,j] * pts[j] - pts[i] * sum_j inv_d2[i,j]
    grad = np.einsum('ij,jk->ik', inv_d2, pts) - pts * np.sum(inv_d2, axis=1, keepdims=True)
    
    return energy, grad

def _hessian_tangent(pts):
    """
    Compute the Hessian of the energy restricted to the tangent space.
    H_ij = d^2 E / d(p_i) d(p_j)
    
    For the log energy E = -sum_{i<j} log(d_ij):
    dE/dp_i = -sum_{j!=i} (p_i - p_j)/d_ij^2
    
    The Hessian involves second derivatives. For a pair (i,j):
    d^2E/dp_i dp_j = (p_i - p_j)(p_i - p_j)^T / d^3 - delta_ij / d^2  (for i != j)
    Wait, let me be more careful.
    
    E = -sum_{i<j} log(|p_i - p_j|)
    
    dE/dp_i = -sum_{j!=i} (1/|p_i-p_j|) * d|p_i-p_j|/dp_i
             = -sum_{j!=i} (1/d_ij) * (p_i - p_j)/d_ij
             = -sum_{j!=i} (p_i - p_j)/d_ij^2
    
    Now d^2E/dp_i dp_j for i != j:
    d/dp_j [- (p_i - p_j)/d_ij^2]
    = -[ (-1) * d_ij^2 + (p_i - p_j) * 2*d_ij * d(d_ij)/dp_j ] / d_ij^4
    = (p_i - p_j)/d_ij^2 - 2(p_i - p_j)(p_i - p_j)^T (p_j - p_i) / d_ij^4
    Wait, this is getting complex. Let me use the formula:
    
    d/dx [f(x)/g(x)^2] where f = p_i - p_j, g = d_ij
    
    Actually, let me use a different approach. The Hessian of -log(d) with respect to p_i and p_j:
    
    For i != j:
    d^2(-log d_ij)/dp_i dp_j = (p_i - p_j)(p_i - p_j)^T / d_ij^3 - delta_ik * delta_jl / d_ij^2
    No wait, that's not right either.
    
    Let me just compute it numerically or use the known formula.
    
    Actually, for the purpose of Newton's method on the sphere, I'll compute the Hessian
    in ambient space and then project to tangent space.
    
    H_ii = sum_{j!=i} [ 3(p_i-p_j)(p_i-p_j)^T / d^5 - I / d^3 ]
    H_ij = -[ (p_i-p_j)(p_i-p_j)^T / d^3 - I / d^5 ]  for i != j
    
    Wait, let me derive this properly.
    
    E = -sum_{k<l} log(d_kl)
    
    dE/dp_i = -sum_{j!=i} (p_i - p_j)/d_ij^2
    
    d^2E/dp_i^2 = -sum_{j!=i} d/dp_i [(p_i - p_j)/d_ij^2]
    = -sum_{j!=i} [ I/d_ij^2 + (p_i-p_j) * (-2)(p_i-p_j)^T / d_ij^4 ]
    = -sum_{j!=i} [ I/d_ij^2 - 2(p_i-p_j)(p_i-p_j)^T / d_ij^4 ]
    = sum_{j!=i} [ 2(p_i-p_j)(p_i-p_j)^T / d_ij^4 - I/d_ij^2 ]
    
    d^2E/dp_i dp_j (i != j) = -d/dp_j [(p_i - p_j)/d_ij^2]
    = -[ (-I)/d_ij^2 + (p_i-p_j) * (-2)(p_i-p_j)^T (-1) / d_ij^4 ]
    = -[ -I/d_ij^2 + 2(p_i-p_j)(p_i-p_j)^T / d_ij^4 ]
    = I/d_ij^2 - 2(p_i-p_j)(p_i-p_j)^T / d_ij^4
    
    Hmm, but this is in ambient space. For Newton's method on the sphere, I need to work
    in the tangent space. This is getting complex. Let me use a simpler approach:
    use L-BFGS-B which approximates the inverse Hessian, and for the final polish,
    use a simpler line-search based Newton step with numerical Hessian.
    """
    pass  # We'll use L-BFGS-B instead which approximates inverse Hessian

def _optimize_lbfgs(pts, time_limit=10.0):
    """
    Optimize using L-BFGS-B in ambient space with sphere constraint handled
    by re-normalization after each step.
    """
    pts = pts.copy()
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Flatten for scipy
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(N, 3)
        # Normalize to handle any drift
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        p = p / norms
        e, g = _energy_and_grad(p)
        return e
    
    def jac(x):
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        p = p / norms
        e, g = _energy_and_grad(p)
        return g.flatten()
    
    start_time = time.time()
    
    # Use a callback to check time
    def callback(xk):
        if time.time() - start_time > time_limit * 0.95:
            return True
        return False
    
    result = minimize(
        objective, x0, jac=jac, method='L-BFGS-B',
        options={
            'maxiter': 500,
            'maxfun': 2000,
            'ftol': 1e-12,
            'gtol': 1e-10,
            'maxls': 50,
            'callback': callback
        }
    )
    
    best_pts = result.x.reshape(N, 3)
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    e, g = _energy_and_grad(best_pts)
    
    return best_pts, e

def _newton_polish(pts, time_limit=2.0, tol=1e-14):
    """
    Newton-Raphson polish in tangent space with exact Hessian.
    """
    pts = pts.copy()
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    start_time = time.time()
    best_energy = _energy_and_grad(pts)[0]
    best_pts = pts.copy()
    
    for it in range(100):
        if time.time() - start_time > time_limit:
            break
        
        e, g = _energy_and_grad(pts)
        
        # Project gradient to tangent space
        dot = np.sum(pts * g, axis=1, keepdims=True)
        g_tang = g - dot * pts
        
        # Check convergence
        norm_g = np.linalg.norm(g_tang)
        if norm_g < tol:
            break
        
        # Compute Hessian in ambient space
        n = N
        H = np.zeros((n*3, n*3))
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, 1.0)
        d = np.sqrt(d2)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal: H_ii = sum_{k!=i} [ 2(p_i-p_k)(p_i-p_k)^T / d_ik^4 - I/d_ik^2 ]
                    for k in range(n):
                        if k == i:
                            continue
                        dik = d[i, k]
                        if dik < 1e-12:
                            continue
                        diff_ik = pts[i] - pts[k]
                        outer = np.outer(diff_ik, diff_ik)
                        H[i*3:(i+1)*3, i*3:(i+1)*3] += 2.0 * outer / (dik**4) - np.eye(3) / (dik**2)
                else:
                    # Off-diagonal: H_ij = I/d_ij^2 - 2(p_i-p_j)(p_i-p_j)^T / d_ij^4
                    dij = d[i, j]
                    if dij < 1e-12:
                        continue
                    diff_ij = pts[i] - pts[j]
                    outer = np.outer(diff_ij, diff_ij)
                    block = np.eye(3) / (dij**2) - 2.0 * outer / (dij**4)
                    H[i*3:(i+1)*3, j*3:(j+1)*3] += block
                    H[j*3:(j+1)*3, i*3:(i+1)*3] += block.T
        
        # Project Hessian to tangent space
        # For each point i, the tangent space is {v : v . p_i = 0}
        # We need to solve: H_tang * delta = -g_tang
        # where H_tang is the Hessian projected to tangent space
        
        # Simple approach: use the ambient Hessian but constrain steps to tangent space
        # delta_i must satisfy delta_i . p_i = 0
        
        # Build the system with constraints
        # For each i, we have 3 unknowns (delta_i) with 1 constraint (delta_i . p_i = 0)
        # Total: 3N unknowns, N constraints
        
        # Use a penalty method or project the solution
        # Simpler: solve in ambient space, then project
        
        # Solve H * delta = -g (ambient)
        try:
            delta = np.linalg.solve(H, -g.flatten())
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, -g.flatten(), rcond=None)[0]
        
        delta = delta.reshape(N, 3)
        
        # Project delta to tangent space
        dot_d = np.sum(pts * delta, axis=1, keepdims=True)
        delta_tang = delta - dot_d * pts
        
        # Line search with damping
        alpha = 1.0
        found = False
        for _ in range(30):
            new_pts = pts + alpha * delta_tang
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_e = _energy_and_grad(new_pts)[0]
            
            if new_e < best_energy:
                best_energy = new_e
                best_pts = new_pts.copy()
                pts = new_pts
                found = True
                break
            else:
                alpha *= 0.5
                if alpha < 1e-10:
                    break
        
        if not found:
            break
    
    return best_pts, best_energy

def construct():
    total_deadline = 118.0  # Safety margin from 120s
    start_global = time.time()
    
    best_points = None
    best_energy = np.inf
    
    def fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def random_uniform():
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    
    # Phase 1: L-BFGS-B optimization from multiple starts
    n_restarts = 8
    time_per_restart = (total_deadline - 5.0) / n_restarts  # Leave 5s for polish
    
    inits = [fibonacci_spiral()]
    for _ in range(n_restarts - 1):
        inits.append(random_uniform())
    
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_global
        if elapsed > total_deadline - 5.0:
            break
        
        remaining = total_deadline - 5.0 - elapsed
        if remaining < 2.0:
            break
        
        # Allocate time for this restart
        alloc_time = min(time_per_restart, remaining)
        
        pts, energy = _optimize_lbfgs(init_pts, time_limit=alloc_time)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts
    
    # Phase 2: Newton polish on the best result
    if best_points is not None and time.time() - start_global < total_deadline - 1.0:
        remaining = total_deadline - 1.0 - (time.time() - start_global)
        if remaining > 0.5:
            best_points, best_energy = _newton_polish(best_points, time_limit=remaining, tol=1e-14)
    
    # Ensure valid output
    if best_points is None:
        best_points = fibonacci_spiral()
    
    # Final normalization and validation
    best_points = best_points.copy()
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    best_points = best_points / norms
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_spiral()
    
    return best_points
