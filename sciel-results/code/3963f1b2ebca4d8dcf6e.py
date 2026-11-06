"""
Minimize logarithmic energy of 100 points on the unit sphere.
Strategy:
  1. Multiple diverse initializations (Fibonacci, jittered, random, icosahedron-based).
  2. Vectorized projected gradient descent with adaptive step size.
  3. L-BFGS-B refinement with tight tolerances.
  4. Final high-precision polish.
"""
import numpy as np
import math
import time

N = 100

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient (projected to tangent space).
    Energy: sum_{i<j} log(1/|p_i - p_j|) = -sum_{i<j} log(d_ij)
    Gradient at p_i: sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2, projected to tangent.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)

    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask_upper]

    if np.any(d_upper < 1e-9):
        return 1e18, np.zeros_like(pts)

    energy = -math.fsum(math.log(di) for di in d_upper)

    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)

    grad = np.einsum('ij,ijk->ik', inv_d2, diff)

    radial = np.einsum('ij,ij->i', grad, pts)[:, np.newaxis]
    grad = grad - radial * pts

    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, jitter=0.0):
    """Generate Fibonacci spiral points on unit sphere with optional jitter."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        rng = np.random.RandomState(int(jitter * 1000))
        noise = rng.randn(n, 3) * jitter
        noise = noise - np.einsum('ij,ij->i', noise, pts)[:, np.newaxis] * pts
        pts = _project_to_sphere(pts + noise)
    return pts

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n):
    """Generate points starting from icosahedron vertices and filling in."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    pts = verts.copy()
    rng = np.random.RandomState(42)
    
    while len(pts) < n:
        candidates = rng.randn(1000, 3)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        best_c = None
        best_min_dist = -1
        for c in candidates:
            dists = np.linalg.norm(pts - c, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_c = c
        
        pts = np.vstack([pts, best_c.reshape(1, 3)])
    
    return pts[:n]

def _projected_gradient_descent(pts0, max_iter=5000, tol=1e-12, time_limit=None, start_time=None):
    """
    Projected gradient descent with adaptive step size on the sphere.
    Uses the fact that the gradient is already in tangent space.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 60.0
    
    pts = _project_to_sphere(pts0.copy())
    energy, grad = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    # Adaptive step size
    step_size = 0.01
    min_step = 1e-6
    max_step = 0.1
    patience = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        g_norm = np.linalg.norm(grad)
        if g_norm < tol:
            break
        
        # Adaptive step size based on gradient magnitude
        # Scale step by 1/sqrt(g_norm) for better conditioning
        alpha = step_size / max(g_norm, 1e-8)
        alpha = min(alpha, max_step)
        alpha = max(alpha, min_step)
        
        # Take step in tangent direction
        new_pts = _project_to_sphere(pts - alpha * grad)
        
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if new_energy < energy:
            # Accept step
            pts = new_pts
            energy = new_energy
            grad = new_grad
            patience = 0
            # Increase step size slightly
            step_size = min(step_size * 1.05, max_step)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        else:
            # Reject step, decrease step size
            step_size = max(step_size * 0.5, min_step)
            patience += 1
            if patience > 100:
                break
    
    return best_pts, best_energy

def _lbfgs_optimize(pts0, maxiter=5000, ftol=1e-13, gtol=1e-10, time_limit=None, start_time=None):
    """L-BFGS optimization with projection onto sphere."""
    from scipy.optimize import minimize
    
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 30.0
    
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
                     'ftol': ftol, 'gtol': gtol, 'm': 20}
        )
        best_pts = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy = _energy_and_grad(best_pts)[0]
        return best_pts, best_energy
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _conjugate_gradient_polish(pts0, max_iter=200, tol=1e-14, time_limit=None, start_time=None):
    """
    Riemannian conjugate gradient polish for final refinement.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 15.0
    
    pts = _project_to_sphere(pts0.copy())
    energy, grad = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    # Initial search direction
    d = -grad.copy()
    d = d - np.einsum('ij,ij->i', d, pts)[:, np.newaxis] * pts
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        g_norm = np.linalg.norm(grad)
        if g_norm < tol:
            break
        
        # Line search: find optimal step along direction d
        best_alpha = 0.01
        best_e = energy
        
        # Try several step sizes
        for alpha in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
            new_pts = _project_to_sphere(pts + alpha * d)
            e, _ = _energy_and_grad(new_pts)
            if e < best_e:
                best_e = e
                best_alpha = alpha
        
        new_pts = _project_to_sphere(pts + best_alpha * d)
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if new_energy >= energy:
            break
        
        # Fletcher-Reeves beta
        old_g2 = np.sum(grad * grad)
        new_g2 = np.sum(new_grad * new_grad)
        beta = new_g2 / max(old_g2, 1e-20)
        
        pts = new_pts
        energy = new_energy
        grad = new_grad
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Update direction
        d = -grad + beta * d
        d = d - np.einsum('ij,ij->i', d, pts)[:, np.newaxis] * pts
    
    return best_pts, best_energy

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer before 150s limit

    best_pts = None
    best_energy = np.inf

    # 1. Diverse Initializations
    initializations = []
    initializations.append(_fibonacci_sphere(N))
    initializations.append(_fibonacci_sphere(N, jitter=0.01))
    initializations.append(_fibonacci_sphere(N, jitter=0.05))
    initializations.append(_icosahedron_sphere(N))
    
    seeds = [42, 123, 777, 999, 2023, 555, 888, 1337, 31415, 27182, 
             111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
    for s in seeds[:15]:
        initializations.append(_random_sphere(N, s))

    # 2. Projected Gradient Descent Phase (fast, vectorized)
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 50.0:
            break
        
        time_left = deadline - time.time()
        max_iter = min(3000, int(time_left * 30))
        pts_opt, e_opt = _projected_gradient_descent(
            pts0, max_iter=max_iter, tol=1e-10, 
            time_limit=min(3.0, time_left * 0.1), start_time=start_time
        )
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()

    if best_pts is None:
        return _fibonacci_sphere(N)

    # 3. L-BFGS Refinement Phase
    time_left = deadline - time.time()
    if time_left > 20.0:
        pts_opt, e_opt = _lbfgs_optimize(
            best_pts, maxiter=3000, ftol=1e-13, gtol=1e-10,
            time_limit=min(15.0, time_left * 0.5), start_time=start_time
        )
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()

    # 4. Second round of gradient descent from best point with different params
    time_left = deadline - time.time()
    if time_left > 15.0:
        pts_opt, e_opt = _projected_gradient_descent(
            best_pts, max_iter=2000, tol=1e-12,
            time_limit=min(10.0, time_left * 0.5), start_time=start_time
        )
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()

    # 5. Conjugate Gradient Polish
    time_left = deadline - time.time()
    if time_left > 10.0:
        pts_cg, e_cg = _conjugate_gradient_polish(
            best_pts, max_iter=100, tol=1e-13,
            time_limit=min(8.0, time_left * 0.5), start_time=start_time
        )
        if e_cg < best_energy:
            best_energy = e_cg
            best_pts = pts_cg.copy()

    # 6. Final L-BFGS with very tight tolerance
    time_left = deadline - time.time()
    if time_left > 5.0:
        pts_final, e_final = _lbfgs_optimize(
            best_pts, maxiter=2000, ftol=1e-14, gtol=1e-12,
            time_limit=min(4.0, time_left * 0.7), start_time=start_time
        )
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final.copy()

    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points closer than 1e-9
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them
                rng = np.random.RandomState(12345)
                best_pts[j] = _project_to_sphere(best_pts[j] + 0.01 * rng.randn(3))
    
    return best_pts
