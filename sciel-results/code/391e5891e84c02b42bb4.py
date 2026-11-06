"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())

def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere (vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    
    if np.any(d_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
    
    energy = math.fsum(np.log(1.0 / d_upper).tolist())
    
    # Gradient of sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    # d/dx_i E = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Note: log(1/d) = -log(d). d/dx (-log|d|) = - (1/|d|) * d|x_i-x_j|/dx_i = - (1/d) * (x_i-x_j)/d = -(x_i-x_j)/d^2
    # So grad_i = - sum_{j!=i} (x_i - x_j) / d_{ij}^2
    
    # weights_{ij} = 1/d_{ij}^2 for i != j, 0 for i=j
    weights = 1.0 / d2
    np.fill_diagonal(weights, 0.0)
    
    # grad_i = - sum_j weights_{ij} * (x_i - x_j)
    # = - (sum_j weights_{ij} * x_i - sum_j weights_{ij} * x_j)
    # = - x_i * sum_j weights_{ij} + sum_j weights_{ij} * x_j
    
    sum_weights = np.sum(weights, axis=1) # shape (n,)
    # sum_j weights_{ij} x_j is matrix-vector product weights @ pts
    weighted_sum = weights @ pts # shape (n, 3)
    
    grad = -pts * sum_weights[:, np.newaxis] + weighted_sum
    
    # Project gradient to tangent space
    # proj_v = v - (v . x) x
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - radial * pts
    
    return energy, grad_tangent

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n):
    """Generate points based on icosahedron symmetry, then jitter."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Duplicate and rotate to get N points
    # For N=100, we can take 12 vertices and rotate them
    # Or just use random starts and let optimizer do the work, but icosahedron is a good symmetric start
    # Let's just use Fibonacci for simplicity in this block, or a specific structured start
    # Actually, for N=100, a structured start like a grid on sphere or Fibonacci is standard.
    # I'll stick to Fibonacci and Random for diversity.
    return _fibonacci_sphere(n)

def _optimize_lbfgs(pts0, deadline, max_iter=1000):
    """Optimize using L-BFGS-B on the sphere (flattened)."""
    # Flatten points to 1D
    x0 = pts0.flatten()
    
    def objective(x):
        if time.time() > deadline:
            return 1e18
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _log_energy_grad(pts)
        return e
    
    def gradient(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _log_energy_grad(pts)
        return g.flatten()
        
    try:
        res = minimize(
            objective, 
            x0, 
            jac=gradient, 
            method='L-BFGS-B', 
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
        )
        best_pts = _project_to_sphere(res.x.reshape(-1, 3))
        best_energy = _log_energy(best_pts)
        return best_pts, best_energy
    except Exception:
        return pts0, _log_energy(pts0)

def _optimize_cg(pts0, deadline, max_iter=200):
    """
    Conjugate Gradient on the sphere.
    Faster than L-BFGS for large N sometimes, and very robust.
    """
    pts = pts0.copy()
    energy, grad = _log_energy_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    p = -grad
    # Normalize p? No, just use as direction.
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Line search
        # Simple backtracking line search
        alpha = 0.01
        min_alpha = 1e-6
        max_iter_ls = 20
        found_alpha = False
        
        for _ in range(max_iter_ls):
            new_pts = _project_to_sphere(pts + alpha * p)
            new_energy, new_grad = _log_energy_grad(new_pts)
            
            if new_energy < energy:
                found_alpha = True
                break
            alpha *= 0.5
            
        if not found_alpha:
            # Try smaller alpha or break
            # If we can't improve, we might be stuck or at minimum
            # Try a random perturbation if stuck? No, CG should handle it.
            # If alpha is too small, break.
            if alpha < min_alpha:
                break
            # Force accept if it's close? No, reject.
            # Let's try a tiny step
            new_pts = _project_to_sphere(pts + 1e-6 * p)
            new_energy, new_grad = _log_energy_grad(new_pts)
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                grad = new_grad
            else:
                break
                
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # CG update
        # beta = (grad_new . grad_new) / (grad_old . grad_old)
        # Polak-Ribiere: beta = (grad_new . (grad_new - grad_old)) / (grad_old . grad_old)
        # Fletcher-Reeves: beta = (grad_new . grad_new) / (grad_old . grad_old)
        
        old_grad = grad
        grad = new_grad
        
        num = np.sum(grad * (grad - old_grad))
        den = np.sum(old_grad * old_grad)
        
        if den < 1e-12:
            beta = 0.0
        else:
            beta = num / den
            if beta < 0:
                beta = 0.0 # Restart CG
        
        p = -grad + beta * p
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-8:
            break
            
    return best_pts, best_energy

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pts = None
    best_energy = np.inf
    
    # Generate diverse starts
    # 1. Fibonacci
    # 2. Random seeds
    # 3. Spherical coordinates grid (approx)
    
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        starts.append(_random_sphere(N, seed))
        
    # Add a structured grid start
    # Approximate spherical grid
    # This is tricky to do perfectly, but random + fib is usually sufficient.
    # Let's add a "random walk" start? No, random is fine.
    
    # Allocate time to each start
    # We have ~140s.
    # L-BFGS is fast. CG is fast.
    # Let's try L-BFGS on all starts first, then polish with CG on the best ones.
    
    # Phase 1: L-BFGS on all starts
    results = []
    for i, pts0 in enumerate(starts):
        if time.time() > deadline - 20:
            break
            
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline - 10)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
        results.append((e_opt, pts_opt))
        
    # Phase 2: Polish top 3 results with CG
    # Sort by energy
    results.sort(key=lambda x: x[0])
    
    for e, pts in results[:3]:
        if time.time() > deadline - 5:
            break
        pts_cg, e_cg = _optimize_cg(pts, deadline - 2)
        if e_cg < best_energy:
            best_energy = e_cg
            best_pts = pts_cg
            
    # Phase 3: Final L-BFGS polish on the very best
    if best_pts is not None and time.time() < deadline - 2:
        pts_final, e_final = _optimize_lbfgs(best_pts, deadline - 1, max_iter=500)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final

    return best_pts
# EVOLVE-BLOCK-END
