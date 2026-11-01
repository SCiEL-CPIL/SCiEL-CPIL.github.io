"""Direct approach for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy using math.fsum for precision."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Energy: sum over i<j of log(1/d)
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    energy = np.sum(np.log(1.0 / d_upper))
    
    # Gradient: for point i, grad_i = sum_{j!=i} (pts_i - pts_j) / d_ij^2
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    grad = pts * sum_inv[:, np.newaxis] - weighted_sum
    
    # Project to tangent space: grad_i -= dot(grad_i, pts_i) * pts_i
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _lbgfs_sphere(pts0, max_iter=500, m=10, time_limit=None, start_time=None):
    """L-BFGS optimization on the sphere with projected gradients and Wolfe line search."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    # L-BFGS memory
    s_list = []  # step directions
    y_list = []  # gradient differences
    
    step_size = 0.1
    min_step = 1e-12
    c1 = 1e-4  # Armijo condition
    c2 = 0.9   # Curvature condition for Wolfe
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # L-BFGS two-loop recursion to find search direction
        if len(s_list) == 0:
            direction = -grad
        else:
            # Two-loop recursion
            q = grad.copy()
            rhos = []
            alphas = []
            
            # Backward loop
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                si_yi = np.sum(si * yi)
                if abs(si_yi) < 1e-16:
                    rhos.append(0.0)
                    alphas.append(0.0)
                else:
                    rho = 1.0 / si_yi
                    rhos.append(rho)
                    ai = np.sum(si * q) * rho
                    alphas.append(ai)
                    q -= ai * yi
            
            # Initial curvature estimate
            if len(s_list) > 0:
                s_first = s_list[0]
                y_first = y_list[0]
                gk_y1 = np.sum(grad * y_first)
                gk_s1 = np.sum(grad * s_first)
                gamma = gk_s1 / gk_y1 if abs(gk_y1) > 1e-16 else 1.0 / (np.sum(grad * grad) + 1e-16)
            else:
                gamma = 1.0 / (np.sum(grad * grad) + 1e-16)
            
            r = q * gamma
            
            # Forward loop
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho_i = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta = rho_i * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        # Ensure direction is on tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Wolfe line search
        alpha = step_size
        accepted = False
        grad_dir = np.sum(grad * direction)
        
        if grad_dir >= 0:
            # Direction is not a descent direction, use negative gradient
            direction = -grad
            dot_dir = np.sum(direction * pts, axis=1)
            direction -= dot_dir[:, np.newaxis] * pts
            grad_dir = np.sum(grad * direction)
        
        for ls_iter in range(40):
            # Move along direction and project to sphere
            new_pts = pts + alpha * direction
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Check Wolfe conditions
            armijo = new_energy <= energy + c1 * alpha * grad_dir
            curvature = np.sum(new_grad * direction) >= c2 * grad_dir
            
            if armijo and curvature:
                accepted = True
                break
            elif armijo:
                # Good enough, accept
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            step_size *= 0.5
            if step_size < min_step:
                break
        
        # Update L-BFGS memory
        s = alpha * direction
        y = new_grad - grad
        sy = np.sum(s * y)
        
        if sy > 1e-14:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s)
            y_list.append(y)
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
        step_size = min(alpha * 1.5, 1.0)
    
    return pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
    
    # 3. Perturbed Fibonacci with different perturbation strengths
    for seed, scale in [(7, 0.05), (99, 0.1), (200, 0.02)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * scale
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    # 4. Grid-based initialization
    n_per_ring = int(np.sqrt(N))
    grid_pts = []
    for i in range(n_per_ring):
        for j in range(n_per_ring):
            if len(grid_pts) >= N:
                break
            theta = 2 * np.pi * j / n_per_ring
            phi = np.pi * (i + 0.5) / n_per_ring
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            grid_pts.append([x, y, z])
    if len(grid_pts) == N:
        grid_arr = np.array(grid_pts)
        grid_arr /= np.linalg.norm(grid_arr, axis=1, keepdims=True)
        inits.append(grid_arr)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Run L-BFGS on all initializations with moderate iterations
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - time.time()
        time_per_init = min(remaining / max(1, len(inits) - idx), 25.0)
        max_iter = int(time_per_init * 20)  # Estimate ~50ms per iteration
        max_iter = max(max_iter, 50)
        
        pts, energy = _lbgfs_sphere(
            init_pts, 
            max_iter=max_iter, 
            m=10,
            time_limit=time_per_init,
            start_time=time.time()
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Refine the best result with more iterations
    if best_pts is not None and time.time() < deadline - 10:
        remaining = deadline - time.time()
        best_pts, best_energy = _lbgfs_sphere(
            best_pts,
            max_iter=int(remaining * 15),
            m=20,
            time_limit=remaining - 3,
            start_time=time.time()
        )
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 3:
        remaining = deadline - time.time()
        best_pts, best_energy = _lbgfs_sphere(
            best_pts,
            max_iter=int(remaining * 20),
            m=20,
            time_limit=remaining - 1,
            start_time=time.time()
        )
    
    # Ensure valid output
    best_pts = best_pts.copy()
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Verify all points are valid and no two points are too close
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        # Check minimum distance
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False
    
    if not valid:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
