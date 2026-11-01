"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time

N = 150

def _energy_and_grad(X):
    """Compute logarithmic energy and Riemannian gradient for points on the unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Riemannian Gradient w.r.t. x_i = (I - x_i x_i^T) * (sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2)
    """
    n = X.shape[0]
    # Compute all pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # avoid division by zero on diagonal
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Euclidean Gradient: grad_eucl[i] = sum_{j!=i} (x_j - x_i) / dist[i,j]^2
    # Note: diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    # grad_eucl[i] = - sum_j diff[i,j] / dist2[i,j]
    eucl_grad = -np.sum(diff / dist2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: grad_riem[i] = eucl_grad[i] - (eucl_grad[i] . X[i]) X[i]
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X
    
    return energy, riem_grad

def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_init(n, num_dims=3):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        points[i - 1] = [halton(i, 2), halton(i, 3), halton(i, 5)]
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    # Add some jitter
    rng = np.random.RandomState(42)
    points += rng.normal(0, 0.01, size=points.shape)
    return _project_to_sphere(points)

def _icosahedron_init(n):
    """Initialize using icosahedron vertices and their symmetric subdivisions."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Start with icosahedron vertices
    points = [verts]
    count = 12
    
    # Subdivide: for each edge, add midpoint projected to sphere
    # Repeat until we have at least n points
    while count < n:
        new_points = []
        # For each pair of adjacent vertices, add midpoint
        # Icosahedron has 30 edges
        # Find edges by checking which vertex pairs are close
        for i in range(len(points[-1])):
            for j in range(i + 1, len(points[-1])):
                d = np.linalg.norm(points[-1][i] - points[-1][j])
                if d < 1.2:  # heuristic for adjacent vertices
                    mid = (points[-1][i] + points[-1][j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    new_points.append(mid)
        
        if not new_points:
            break
            
        points.append(np.array(new_points))
        count += len(new_points)
    
    # Combine all points
    all_points = np.vstack(points)
    
    # If we have more than n, take first n
    if len(all_points) > n:
        all_points = all_points[:n]
    elif len(all_points) < n:
        # Pad with random points
        rng = np.random.RandomState(999)
        extra = rng.randn(n - len(all_points), 3)
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        all_points = np.vstack([all_points, extra])
    
    return all_points

def _riemannian_gd(X0, max_iter=1000, tol=1e-10, time_limit=10.0):
    """Riemannian Gradient Descent with momentum and backtracking line search."""
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < tol:
        return X, energy
    
    alpha = 0.5  # initial step size
    c = 1e-4     # Armijo parameter
    momentum = 0.0
    velocity = np.zeros_like(X)
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Ramp up momentum
        if k >= 50:
            momentum = 0.4
        elif k >= 20:
            momentum = 0.3
        elif k >= 10:
            momentum = 0.15
        
        # Update velocity with momentum
        velocity = momentum * velocity - (1.0 - momentum) * grad
        
        # Backtracking line search
        step = alpha * velocity
        X_new = X - step
        X_new = _project_to_sphere(X_new)
        
        energy_new, _ = _energy_and_grad(X_new)
        
        # Armijo condition
        while energy_new > energy - c * alpha * np.sum(velocity * grad):
            alpha *= 0.5
            step = alpha * velocity
            X_new = X - step
            X_new = _project_to_sphere(X_new)
            energy_new, _ = _energy_and_grad(X_new)
            if alpha < 1e-15:
                break
        
        if alpha < 1e-15:
            break
        
        X = X_new
        energy = energy_new
        _, grad = _energy_and_grad(X)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Reset alpha periodically to allow larger steps
        if k % 20 == 0 and k > 0:
            alpha = min(alpha * 1.5, 2.0)
            
    return X, energy

def _riemannian_cg(X0, max_iter=500, tol=1e-10, time_limit=15.0):
    """Riemannian Conjugate Gradient on the sphere."""
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < tol:
        return X, energy
    
    # Initial search direction
    d = -grad
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Backtracking line search
        alpha = 1.0
        c = 1e-4
        X_new = _project_to_sphere(X + alpha * d)
        energy_new, grad_new = _energy_and_grad(X_new)
        
        while energy_new > energy - c * alpha * np.sum(grad * d):
            alpha *= 0.5
            X_new = _project_to_sphere(X + alpha * d)
            energy_new, grad_new = _energy_and_grad(X_new)
            if alpha < 1e-15:
                break
        
        if alpha < 1e-15:
            break
        
        X = X_new
        energy = energy_new
        grad = grad_new
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Polak-Ribiere beta
        grad_diff = grad_new - grad
        beta = np.sum(grad_new * grad_diff) / max(np.sum(grad * grad), 1e-15)
        beta = max(0.0, min(beta, 1.0))
        
        d = -grad_new + beta * d
        
        # Ensure d is in tangent space
        dot = np.sum(d * X, axis=1, keepdims=True)
        d = d - dot * X
        
        # Restart if d not descent direction
        if np.sum(grad * d) > 0:
            d = -grad
            beta = 0.0
            
    return X, energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_init(N))
    
    # 2. Halton sequence
    initializations.append(_halton_init(N))
    
    # 3. Icosahedron-based initialization
    initializations.append(_icosahedron_init(N))
    
    # 4. Random with fixed seed
    rng = np.random.RandomState(123)
    rand_pts = rng.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts))
    
    # 5. Random with different seed
    rng2 = np.random.RandomState(456)
    rand_pts2 = rng2.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts2))
    
    # 6. Another random seed
    rng3 = np.random.RandomState(789)
    rand_pts3 = rng3.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts3))
    
    # 7. Fibonacci with slight perturbation
    fib_perturbed = _fibonacci_init(N).copy()
    rng4 = np.random.RandomState(321)
    fib_perturbed += rng4.normal(0, 0.05, size=fib_perturbed.shape)
    initializations.append(_project_to_sphere(fib_perturbed))
    
    best_energy = np.inf
    best_points = initializations[0]
    
    # Phase 1: Riemannian GD + CG from multiple starts
    for init_idx, X0 in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
            
        # Use a time limit for each run to ensure we finish
        time_budget = (deadline - 25.0 - time.time()) / max(1, len(initializations) - init_idx)
        time_budget = max(time_budget, 5.0)
        
        try:
            # First: Riemannian GD to get close to a local minimum
            gd_time = min(time_budget * 0.4, 10.0)
            X_opt, e_opt = _riemannian_gd(X0, max_iter=300, tol=1e-7, time_limit=gd_time)
            
            # Second: Riemannian CG for faster convergence
            if time.time() < deadline - 20.0:
                cg_time = min(time_budget * 0.4, 15.0)
                X_cg, e_cg = _riemannian_cg(X_opt, max_iter=200, tol=1e-9, time_limit=cg_time)
                if e_cg < e_opt:
                    X_opt = X_cg
                    e_opt = e_cg
            
            # Third: Final L-BFGS-B polish with tighter tolerance
            if time.time() < deadline - 15.0:
                def objective(x_flat):
                    X = x_flat.reshape(N, 3)
                    X = _project_to_sphere(X)
                    e, _ = _energy_and_grad(X)
                    return e
                
                def gradient(x_flat):
                    X = x_flat.reshape(N, 3)
                    X = _project_to_sphere(X)
                    e, g = _energy_and_grad(X)
                    return g.flatten()
                    
                result = minimize(
                    objective,
                    X_opt.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                X_polished = _project_to_sphere(result.x.reshape(N, 3))
                e_polished, _ = _energy_and_grad(X_polished)
                
                if e_polished < e_opt:
                    X_opt = X_polished
                    e_opt = e_polished
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt
                
        except Exception:
            pass

    # Phase 2: Final high-precision polish on the best result
    if time.time() < deadline - 3.0:
        try:
            X0 = best_points.copy()
            
            # Riemannian CG for high precision
            cg_time = min(10.0, deadline - 3.0 - time.time())
            X_cg, e_cg = _riemannian_cg(X0, max_iter=300, tol=1e-12, time_limit=cg_time)
            
            if e_cg < best_energy:
                best_points = X_cg
                best_energy = e_cg
            
            # Final L-BFGS-B polish
            if time.time() < deadline - 1.0:
                def objective(x_flat):
                    X = x_flat.reshape(N, 3)
                    X = _project_to_sphere(X)
                    e, _ = _energy_and_grad(X)
                    return e
                
                def gradient(x_flat):
                    X = x_flat.reshape(N, 3)
                    X = _project_to_sphere(X)
                    e, g = _energy_and_grad(X)
                    return g.flatten()
                    
                result = minimize(
                    objective,
                    best_points.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                
                X_opt = _project_to_sphere(result.x.reshape(N, 3))
                e_opt, _ = _energy_and_grad(X_opt)
                
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_points = X_opt
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9:
        # Fall back to Fibonacci initialization
        return _fibonacci_init(N)
    
    return best_points
# EVOLVE-BLOCK-END
