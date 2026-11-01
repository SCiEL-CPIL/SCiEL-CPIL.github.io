"""Stronger algorithm for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _energy_grad_fast(points):
    """Compute energy and gradient efficiently using dist_sq directly."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = np.sqrt(dist_sq[mask])
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    return energy, grad


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = np.sqrt(dist_sq[mask])
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _sphere_fun_grad(x):
    """Objective and gradient in ambient R^(3N) space with projection."""
    p = _project_to_sphere(x.reshape(-1, 3))
    e, g = _energy_grad_fast(p)
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()


def _optimize_single(points, time_budget, maxiter=5000, ftol=1e-14, gtol=1e-12):
    """Run L-BFGS-B optimization from a single starting point."""
    start_time = time.time()
    x0 = points.flatten().copy()
    
    try:
        res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


def _steepest_descent_sphere(points, time_budget, max_iter=5000):
    """Steepest descent on the sphere using Riemannian gradient."""
    start_time = time.time()
    p = points.copy()
    best_energy = _energy(p)
    best_p = p.copy()
    
    e, g = _energy_grad_fast(p)
    g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
    
    step = 0.01
    for _ in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-12:
            break
        
        new_p = p + step * g_tan
        new_p = _project_to_sphere(new_p)
        
        new_e = _energy(new_p)
        if new_e < best_energy:
            best_energy = new_e
            best_p = new_p.copy()
            _, g = _energy_grad_fast(new_p)
            g_tan = g - (g * new_p).sum(axis=1, keepdims=True) * new_p
            p = new_p
            step = min(step * 1.05, 0.1)
        else:
            step *= 0.5
            if step < 1e-10:
                break
    
    return best_p


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(points)


def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n, 3)
    return _project_to_sphere(points)


def _bands_initial(n, num_bands=5, seed=None):
    """Generate points in latitudinal bands for diverse initial configurations."""
    if seed is not None:
        np.random.seed(seed)
    points = []
    for b in range(num_bands):
        z_center = -1.0 + 2.0 * (b + 0.5) / num_bands
        z_spread = 0.8 / num_bands
        n_band = n // num_bands + (1 if b < n % num_bands else 0)
        for _ in range(n_band):
            z = z_center + np.random.uniform(-z_spread, z_spread)
            z = np.clip(z, -1.0, 1.0)
            r = np.sqrt(1.0 - z * z)
            phi = np.random.uniform(0, 2 * np.pi)
            points.append([r * np.cos(phi), r * np.sin(phi), z])
    points = np.array(points)
    return _project_to_sphere(points)


def _rings_initial(n, num_rings=3, seed=None):
    """Generate points on concentric rings for diverse initial configurations."""
    if seed is not None:
        np.random.seed(seed)
    points = []
    for r_idx in range(num_rings):
        z = -1.0 + 2.0 * (r_idx + 0.5) / num_rings
        r = np.sqrt(1.0 - z * z)
        n_ring = n // num_rings + (1 if r_idx < n % num_rings else 0)
        for i in range(n_ring):
            phi = 2.0 * np.pi * i / n_ring
            points.append([r * np.cos(phi), r * np.sin(phi), z])
    points = np.array(points)
    return _project_to_sphere(points)


def _simulated_annealing_refine(points, time_budget):
    """Refine points using simulated annealing with rotation and swap moves."""
    start_time = time.time()
    best_points = points.copy()
    best_energy = _energy(best_points)
    T = 0.03
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        move_type = np.random.random()
        
        if move_type < 0.7:
            n_moves = np.random.choice([1, 1, 1, 2, 2])
            indices = np.random.choice(N, size=n_moves, replace=False)
            
            new_points = best_points.copy()
            for idx in indices:
                angle = np.random.uniform(0, 2 * np.pi) * T
                axis = np.random.randn(3)
                axis /= np.linalg.norm(axis)
                
                c = math.cos(angle)
                s = math.sin(angle)
                k = axis
                k_outer_k = np.outer(k, k)
                K = np.array([[0, -k[2], k[1]],
                              [k[2], 0, -k[0]],
                              [-k[1], k[0], 0]])
                R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
                
                new_points[idx] = R @ new_points[idx]
        else:
            i, j = np.random.choice(N, size=2, replace=False)
            new_points = best_points.copy()
            
            angle = np.random.uniform(0, 2 * np.pi) * T * 0.5
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
            
            temp_i = new_points[i].copy()
            new_points[i] = R @ new_points[j]
            new_points[j] = R @ temp_i
        
        new_points = _project_to_sphere(new_points)
        
        new_energy = _energy(new_points)
        delta_e = new_energy - best_energy
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            best_points = new_points
            best_energy = new_energy
        
        T *= 0.9995
        if T < 0.005:
            T = 0.015
    
    return best_points


def _laplacian_relax(points, time_budget, max_iter=2000):
    """Laplacian relaxation: iteratively move each point toward centroid of neighbors."""
    start_time = time.time()
    p = points.copy()
    best_energy = _energy(p)
    best_p = p.copy()
    
    # Find k nearest neighbors for each point
    k = 6
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # For each point, find k nearest neighbors
    nn_indices = np.argsort(dist_sq, axis=1)[:, :k]
    
    alpha = 0.1
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        new_p = p.copy()
        for i in range(N):
            neighbors = p[nn_indices[i]]
            centroid = np.mean(neighbors, axis=0)
            new_p[i] = (1 - alpha) * p[i] + alpha * centroid
            new_p[i] /= np.linalg.norm(new_p[i])
        
        new_p = _project_to_sphere(new_p)
        new_e = _energy(new_p)
        
        if new_e < best_energy:
            best_energy = new_e
            best_p = new_p.copy()
            p = new_p
            alpha = min(alpha * 1.02, 0.3)
        else:
            alpha *= 0.8
            if alpha < 0.001:
                break
    
    return best_p


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy(best_points)
    
    # Phase 1: L-BFGS-B from Fibonacci spiral
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.25:
        optimized = _optimize_single(points, TIME_LIMIT * 0.2, maxiter=5000, ftol=1e-14, gtol=1e-12)
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 2: Multiple diverse random restarts with L-BFGS-B
    np.random.seed(123)
    restart_count = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.65:
            break
        restart_count += 1
        
        # Generate diverse random starting points
        if restart_count % 5 == 0:
            base = _fibonacci_spiral(N)
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, 2 * np.pi)
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
            random_points = (R @ base.T).T
        elif restart_count % 5 == 1:
            random_points = _random_points(N)
        elif restart_count % 5 == 2:
            base = _fibonacci_spiral(N)
            perturb = np.random.randn(N, 3) * 0.15
            random_points = _project_to_sphere(base + perturb)
        elif restart_count % 5 == 3:
            random_points = _bands_initial(N, num_bands=5, seed=100 + restart_count)
        else:
            random_points = _rings_initial(N, num_rings=3, seed=200 + restart_count)
        
        budget_per_restart = min(TIME_LIMIT * 0.035, (TIME_LIMIT * 0.65 - elapsed) * 0.85)
        if budget_per_restart < 1.0:
            break
        
        optimized = _optimize_single(random_points, budget_per_restart, maxiter=3000, ftol=1e-13, gtol=1e-11)
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 3: Laplacian relaxation
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.8:
        remaining = TIME_LIMIT * 0.8 - elapsed
        if remaining > 2.0:
            relaxed = _laplacian_relax(best_points, remaining * 0.7)
            e = _energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_points = relaxed
    
    # Phase 4: Simulated annealing refinement
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.9:
        remaining = TIME_LIMIT * 0.9 - elapsed
        if remaining > 2.0:
            np.random.seed(456)
            sa_points = _simulated_annealing_refine(best_points, remaining * 0.85)
            e = _energy(sa_points)
            if e < best_energy:
                best_energy = e
                best_points = sa_points
    
    # Phase 5: Fine-tune with tight tolerances
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.96:
        remaining = TIME_LIMIT * 0.96 - elapsed
        if remaining > 1.5:
            refined = _optimize_single(best_points, remaining, maxiter=10000, ftol=1e-15, gtol=1e-13)
            e = _energy(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final energy check
    final_energy = _energy(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    return best_points
