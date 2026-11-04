"""Rewritten program for logarithmic energy minimization of 150 points on the sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=None):
    """Generate n uniformly random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    x = rng.randn(n, 3)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _geodesic_dome(n):
    """Generate points on sphere using geodesic dome subdivision (icosahedron)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide until we have at least n vertices
    # 20 faces -> 12 verts (subdiv 0)
    # 80 faces -> 42 verts (subdiv 1)
    # 320 faces -> 162 verts (subdiv 2)
    for _ in range(2):
        new_faces = []
        new_verts = []
        vert_to_idx = {}
        
        for v in verts:
            key = (round(v[0], 12), round(v[1], 12), round(v[2], 12))
            vert_to_idx[key] = len(verts) + len(new_verts)
        
        for f in faces:
            a = verts[f[0]]
            b = verts[f[1]]
            c = verts[f[2]]
            
            ab = (a + b) / 2.0
            ab /= np.linalg.norm(ab)
            bc = (b + c) / 2.0
            bc /= np.linalg.norm(bc)
            ca = (c + a) / 2.0
            ca /= np.linalg.norm(ca)
            
            for v in [ab, bc, ca]:
                key = (round(v[0], 12), round(v[1], 12), round(v[2], 12))
                if key not in vert_to_idx:
                    vert_to_idx[key] = len(verts) + len(new_verts)
                    new_verts.append(v)
            
            ia = f[0]
            ib = f[1]
            ic = f[2]
            iab = vert_to_idx[(round(ab[0], 12), round(ab[1], 12), round(ab[2], 12))]
            ibc = vert_to_idx[(round(bc[0], 12), round(bc[1], 12), round(bc[2], 12))]
            ica = vert_to_idx[(round(ca[0], 12), round(ca[1], 12), round(ca[2], 12))]
            
            new_faces.append([ia, iab, ica])
            new_faces.append([iab, ib, ibc])
            new_faces.append([ica, ibc, ic])
            new_faces.append([iab, ibc, ica])
        
        verts = np.vstack([verts, np.array(new_verts)]) if new_verts else verts
        faces = new_faces
    
    if len(verts) >= n:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(verts), size=n, replace=False)
        return verts[indices]
    else:
        remaining = n - len(verts)
        extra = _random_uniform_sphere(remaining, seed=999)
        return np.vstack([verts, extra])

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _energy(x):
    """Compute the logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu[0], iu[1]])
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """Compute energy and gradient."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu[0], iu[1]])
    energy = -math.fsum(log_dists.tolist())
    
    inv_d2 = 1.0 / np.maximum(dists_sq, 1e-30)
    np.fill_diagonal(inv_d2, 0.0)
    # grad_i = -sum_j (x_i - x_j) / |x_i - x_j|^2
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _tangent_project(grad, x):
    """Project gradient to tangent space at each point."""
    x = x.reshape(N, 3)
    grad = grad.reshape(N, 3)
    radial = np.sum(grad * x, axis=1, keepdims=True)
    return grad - radial * x

def _exp_map(x, v):
    """Riemannian exponential map on the unit sphere."""
    x = x.reshape(N, 3)
    v = v.reshape(N, 3)
    radial_v = np.sum(v * x, axis=1, keepdims=True)
    v = v - radial_v * x
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    safe_theta = np.where(theta > 1e-15, theta, 1.0)
    sin_theta_over_theta = np.sin(safe_theta) / safe_theta
    cos_theta = np.cos(safe_theta)
    sin_theta_over_theta = np.where(theta > 1e-15, sin_theta_over_theta, 1.0)
    cos_theta = np.where(theta > 1e-15, cos_theta, 1.0)
    result = cos_theta * x + sin_theta_over_theta * v
    result_norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(result_norms, 1e-15)
    return result

def _local_minimize(x0, maxiter=1000, ftol=1e-15, gtol=1e-14, deadline=None):
    """Local minimization using L-BFGS-B with retraction."""
    
    def objective_and_grad(flat_x):
        x = _project_to_sphere(flat_x.reshape(N, 3))
        e, g = _energy_and_grad(x)
        grad_t = _tangent_project(g, x)
        return e, grad_t.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50, 'disp': False}
        )
        x_opt = _project_to_sphere(result.x.reshape(N, 3))
        e_opt = _energy(x_opt)
        return x_opt, e_opt
    except Exception:
        return x0, _energy(x0)

def _polish_high_precision(x, deadline=None, max_iter=2000):
    """High precision polish using gradient descent with adaptive step size."""
    x = _project_to_sphere(x.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    alpha = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(x)
        grad_t = _tangent_project(g, x)
        grad_norm = np.linalg.norm(grad_t)
        
        if grad_norm < 1e-16:
            break
        
        # Adaptive step size
        step_size = alpha / (1.0 + grad_norm)
        v = -step_size * grad_t
        x_trial = _exp_map(x, v)
        e_trial = _energy(x_trial)
        
        if e_trial < best_e:
            x = x_trial
            best_x = x.copy()
            best_e = e_trial
            alpha = min(alpha * 1.05, 0.1)
        else:
            alpha *= 0.5
            if alpha < 1e-18:
                break
                
    return best_x, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initializations
    initializations = []
    x_fib = _fibonacci_sphere(N)
    x_fib = _project_to_sphere(x_fib)
    initializations.append(x_fib)
    
    x_dome = _geodesic_dome(N)
    x_dome = _project_to_sphere(x_dome)
    initializations.append(x_dome)
    
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        x_rand = _random_uniform_sphere(N, seed=seed)
        initializations.append(x_rand)
    
    best_x = initializations[0].copy()
    best_e = _energy(best_x)
    
    # 2. Local minimization for each initialization
    for idx, x_init in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        x_opt, e_opt = _local_minimize(x_init, maxiter=200, ftol=1e-14, gtol=1e-12)
        if e_opt < best_e:
            best_x = x_opt
            best_e = e_opt
    
    # 3. High precision polish on the best result
    if time.time() < deadline - 30:
        x_polished, e_polished = _polish_high_precision(best_x, deadline=deadline - 25)
        if e_polished < best_e:
            best_x = x_polished
            best_e = e_polished
    
    # 4. Basin Hopping
    # We use the remaining time for basin hopping
    # Perturb, minimize, accept if better
    rng = np.random.RandomState(12345)
    scales = [0.1, 0.05, 0.02, 0.01]
    
    while time.time() < deadline - 5:
        # Pick a scale
        scale = scales[int(time.time() * 10) % len(scales)]
        # Perturb
        perturbation = rng.randn(N, 3) * scale
        x_perturbed = _project_to_sphere(best_x + perturbation)
        
        # Local minimize the perturbed version
        # Use fewer iterations for basin hopping steps to allow more restarts
        x_opt, e_opt = _local_minimize(x_perturbed, maxiter=100, ftol=1e-13, gtol=1e-11)
        
        if e_opt < best_e:
            best_x = x_opt
            best_e = e_opt
            
            # Polish the new best immediately
            if time.time() < deadline - 10:
                x_polished, e_polished = _polish_high_precision(best_x, deadline=deadline - 8)
                if e_polished < best_e:
                    best_x = x_polished
                    best_e = e_polished

    # Final polish
    if time.time() < deadline - 1:
        x_polished, e_polished = _polish_high_precision(best_x, deadline=deadline - 0.5)
        if e_polished < best_e:
            best_x = x_polished
            best_e = e_polished

    # Final projection and validation
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-9:
                rng = np.random.RandomState(12345 + i * 100 + j)
                best_x[j] += 1e-10 * rng.randn(3)
                best_x = _project_to_sphere(best_x)
    
    # Final check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
        
    return best_x
