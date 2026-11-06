"""
Minimize logarithmic energy of 100 points on the unit sphere.
Uses L-BFGS-B with multiple diverse initializations and high-precision polishing.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|p_i - p_j|)
    Gradient: sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2, projected to tangent space.
    """
    n = len(pts)
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Compute squared distances
    d2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j (distance 0)
    # We will mask these out in summation
    np.fill_diagonal(d2, 1.0)
    
    # Distances
    d = np.sqrt(d2)
    
    # Energy calculation
    # We only need i < j
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask_upper]
    
    # Check for extremely small distances (points coinciding)
    if np.any(d_upper < 1e-9):
        # This is bad, return high energy to push away
        return 1e18, np.zeros_like(pts)
        
    energy = -np.sum(np.log(d_upper))
    
    # Gradient calculation
    # grad_i = sum_{j != i} (p_i - p_j) / |p_i - p_j|^2
    # Note: diff[i,j] = p_i - p_j
    # So term is diff[i,j] / d2[i,j]
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j (p_i - p_j) / d2_ij
    # Vectorized: sum over j (axis=1)
    grad = np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project gradient onto tangent space
    # p_i is unit vector, so normal is p_i
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial * pts
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedron_sphere(n, seed=0):
    """
    Generate points on the unit sphere via icosahedral symmetry.
    Start from the 12 vertices of a regular icosahedron, then add
    points by rotating copies of a small cluster to cover the sphere.
    """
    rng = np.random.RandomState(seed)
    # 12 vertices of a regular icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    pts = [verts]
    # Add rotated copies to fill the sphere more densely
    # Use random rotations to spread points
    for k in range(1, 12):
        # Random rotation
        q = rng.randn(4)
        q = q / np.linalg.norm(q)
        # Build rotation matrix from quaternion
        x, y, z, w = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])
        rotated = verts @ R.T
        pts.append(rotated)

    all_pts = np.vstack(pts)
    # If we have more than n, randomly sample; if fewer, we need more rotations
    if len(all_pts) >= n:
        # Deduplicate: keep only points that are sufficiently far apart
        # Simple approach: just take first n after shuffling
        idx = rng.permutation(len(all_pts))[:n]
        all_pts = all_pts[idx]
    else:
        # Add more random rotations until we have enough
        while len(all_pts) < n:
            q = rng.randn(4)
            q = q / np.linalg.norm(q)
            x, y, z, w = q
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
                [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
            ])
            rotated = verts @ R.T
            all_pts = np.vstack([all_pts, rotated])
        idx = rng.permutation(len(all_pts))[:n]
        all_pts = all_pts[idx]

    # Re-normalize to unit sphere
    norms = np.linalg.norm(all_pts, axis=1, keepdims=True)
    all_pts = all_pts / norms
    return all_pts[:n]

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _optimize_lbfgs(pts0, deadline, maxiter=5000):
    """
    Optimize using L-BFGS-B.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10, 
                     'ftol': 1e-16, 'gtol': 1e-12, 'm': 20}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _newton_polish(pts0, max_iter=50, tol=1e-14):
    """
    Fast Newton-Raphson polish on the sphere using the Riemannian Hessian
    approximation. This converges quadratically near the minimum.
    
    We use a simple trust-region Newton step:
      1. Compute gradient g (tangent)
      2. Approximate Hessian H via finite differences of the gradient
         (or use a quasi-Newton update)
      3. Solve for the Newton direction in the tangent space
      4. Step with line search
    
    For efficiency, we use a simplified approach:
      - Use the gradient and a diagonal Hessian approximation
      - Or use scipy's minimize with COBYLA for the final polish
    """
    # Actually, let's use a more robust approach:
    # Run L-BFGS-B with extremely tight tolerances and many iterations
    # This is simpler and more reliable than implementing Newton from scratch
    
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 10,
                     'ftol': 1e-16, 'gtol': 1e-14, 'm': 5}
        )
        best_pts = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy = _energy_and_grad(best_pts)[0]
        
        # If we didn't improve, keep original
        e0 = _energy_and_grad(pts0)[0]
        if best_energy < e0:
            return best_pts, best_energy
        else:
            return pts0, e0
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _fine_polish(pts0, deadline, maxiter=10000):
    """
    High precision polishing using L-BFGS-B with very tight tolerances.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                     'ftol': 1e-16, 'gtol': 1e-13, 'm': 10}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. Keep 10s buffer.
    deadline = start_time + 140.0

    best_pts = None
    best_energy = np.inf

    # 1. Diverse Initializations
    # Fibonacci sphere is a very strong heuristic for this problem.
    initializations = []
    
    # Primary: Fibonacci
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron-based initializations
    initializations.append(_icosahedron_sphere(N, seed=1))
    initializations.append(_icosahedron_sphere(N, seed=2))
    
    # Secondary: Random starts with different seeds
    seeds = [42, 123, 777, 999, 2023]
    for s in seeds:
        initializations.append(_random_sphere(N, s))

    # 2. Optimization Loop
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # Standard L-BFGS optimization
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline, maxiter=3000)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
            
        # Early termination if we hit a very low energy (unlikely but possible)
        # The record is around -1083.37
        if best_energy < -1083.377:
            # We have beaten the published record (if true) or are very close.
            # Continue polishing below.
            pass

    if best_pts is None:
        # Fallback: return Fibonacci if optimization failed
        return _fibonacci_sphere(N)

    # 3. Polishing Phase
    # Use remaining time to refine the best candidate
    time_left = deadline - time.time()
    
    if time_left > 10.0:
        # First: tight L-BFGS polish
        pts_polished, e_polished = _fine_polish(best_pts, deadline, maxiter=5000)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = pts_polished
        
        # Second: even tighter Newton-style polish
        time_left = deadline - time.time()
        if time_left > 5.0:
            pts_newton, e_newton = _newton_polish(best_pts, max_iter=100)
            if e_newton < best_energy:
                best_energy = e_newton
                best_pts = pts_newton

    # 4. Final Check and Return
    # Ensure points are strictly on the sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
