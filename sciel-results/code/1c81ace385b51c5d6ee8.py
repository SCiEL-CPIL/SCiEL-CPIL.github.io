"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    E = -np.sum(np.log(d_upper))
    
    inv_d2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return E, grad

def _project_to_sphere(X):
    """Normalize rows of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _geodesic_dome(n):
    """Generate n points using subdivided icosahedron (geodesic dome)."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float64)
    verts = _project_to_sphere(verts)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces until we have enough points
    def subdivide(vertices, faces, depth):
        for _ in range(depth):
            new_faces = []
            mid_cache = {}
            
            def get_midpoint(i, j):
                key = min(i, j) * 10000 + max(i, j)
                if key not in mid_cache:
                    mid = _project_to_sphere((vertices[i] + vertices[j]) / 2.0)
                    mid_cache[key] = len(vertices)
                    vertices.append(mid)
                return mid_cache[key]
            
            for face in faces:
                a, b, c = face
                ab = get_midpoint(a, b)
                bc = get_midpoint(b, c)
                ca = get_midpoint(c, a)
                new_faces.extend([
                    [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]
                ])
            
            return vertices, new_faces
    
    # Determine subdivision depth
    # 12 vertices, 20 faces -> after k subdivisions: 10*4^k + 2 vertices
    depth = 0
    while 10 * (4 ** depth) + 2 < n:
        depth += 1
    
    verts_list = [v.copy() for v in verts]
    verts_list, faces = subdivide(verts_list, faces, depth)
    
    # Take first n points
    result = np.array(verts_list[:n])
    return _project_to_sphere(result)

def _random_sphere(n, seed):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)

def _riemannian_grad_descent(X, time_budget, lr_init=1e-2, min_lr=1e-10):
    """Riemannian gradient descent with exponential map (geodesic) updates."""
    X = _project_to_sphere(X.copy())
    lr = lr_init
    min_lr = 1e-10
    max_iter = 5000
    patience = 0
    max_patience = 100
    E_prev = _energy_and_grad(X)[0]
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        
        E, g = _energy_and_grad(X)
        if not np.isfinite(E):
            break
        
        # Project gradient to tangent space
        g_tan = g - np.sum(g * X, axis=1, keepdims=True) * X
        
        # Exponential map: move along geodesic
        # For unit sphere, exp_x(v) = x*cos(|v|) + (v/|v|)*sin(|v|)
        v = -lr * g_tan
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        v_safe = np.where(v_norm > 1e-15, v / np.maximum(v_norm, 1e-15), 0.0)
        
        # exp_x(v) = x*cos(theta) + u*sin(theta) where theta = |v|, u = v/|v|
        cos_t = np.cos(v_norm)
        sin_t = np.sin(v_norm)
        X_new = X * cos_t + v_safe * sin_t
        X_new = _project_to_sphere(X_new)
        
        E_new = _energy_and_grad(X_new)[0]
        
        if E_new < E_prev - 1e-14:
            X = X_new
            E_prev = E_new
            patience = 0
            if lr < 1e-1:
                lr = min(lr * 1.05, 1e-1)
        else:
            patience += 1
            lr *= 0.7
            if lr < min_lr or patience > max_patience:
                break
    
    return X, _energy_and_grad(X)[0]

def _lbfgs_optimize(X0, time_budget):
    """Run L-BFGS-B optimization starting from X0."""
    X0 = _project_to_sphere(X0.copy())
    
    def fun(X):
        E, g = _energy_and_grad(X)
        return E, g
    
    try:
        res = minimize(
            fun, X0, method='L-BFGS-B', jac=True,
            options={'maxiter': 3000, 'gtol': 1e-14, 'maxls': 50}
        )
        X_opt = _project_to_sphere(res.x)
        E_opt = _energy_and_grad(X_opt)[0]
        return X_opt, E_opt
    except Exception:
        return X0, _energy_and_grad(X0)[0]

def _final_polish(X, time_budget):
    """High-precision final polish with very tight tolerances."""
    X = _project_to_sphere(X.copy())
    
    # First, a quick Riemannian GD pass
    X, E = _riemannian_grad_descent(X, time_budget, lr_init=1e-3, min_lr=1e-12)
    
    # Then L-BFGS with extremely tight tolerance
    elapsed = time.time()
    remaining = time_budget - elapsed
    if remaining > 2.0:
        def fun(X):
            E, g = _energy_and_grad(X)
            return E, g
        
        try:
            res = minimize(
                fun, X, method='L-BFGS-B', jac=True,
                options={'maxiter': 2000, 'gtol': 1e-15, 'maxls': 30}
            )
            X = _project_to_sphere(res.x)
            E = _energy_and_grad(X)[0]
        except Exception:
            pass
    
    return X, E

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Geodesic dome
    try:
        starts.append(_geodesic_dome(N))
    except Exception:
        pass
    
    # 3. Random starts with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        starts.append(_random_sphere(N, seed))
    
    # 4. Anisotropic scaling variations of Fibonacci
    for scale in [0.8, 1.2, 0.6, 1.5]:
        Xf = _fibonacci_sphere(N)
        Xs = Xf.copy()
        Xs[:, 0] *= scale
        starts.append(_project_to_sphere(Xs))
    
    # 5. Spherical cap perturbations
    Xf = _fibonacci_sphere(N)
    rng = np.random.RandomState(999)
    noise = rng.randn(N, 3) * 0.1
    starts.append(_project_to_sphere(Xf + noise))
    
    best_X = None
    best_E = np.inf
    
    for i, X0 in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 8.0:
            break
        
        # Allocate time: L-BFGS then Riemannian GD
        opt_time = min(remaining - 4.0, 25.0)
        if opt_time < 3.0:
            break
        
        # Phase 1: L-BFGS
        X_opt, E_opt = _lbfgs_optimize(X0, time.time() + opt_time * 0.6)
        
        # Phase 2: Riemannian GD polish
        X_opt, E_opt = _riemannian_grad_descent(X_opt, time.time() + opt_time * 0.4, lr_init=1e-3)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
        
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
    
    # Final high-precision polish
    if best_X is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5.0:
            best_X, best_E = _final_polish(best_X, time.time() + remaining - 3.0)
    
    # Final check
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    
    # Ensure all points are on unit sphere and valid
    best_X = _project_to_sphere(best_X)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                rng = np.random.RandomState(i*1000+j)
                best_X[j] += rng.randn(3) * 1e-8
                best_X[j] = _project_to_sphere(best_X[j])
    
    return best_X
# EVOLVE-BLOCK-END
