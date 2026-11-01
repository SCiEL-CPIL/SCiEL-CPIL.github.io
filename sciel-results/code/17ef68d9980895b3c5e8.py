"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e18
            total += math.log(1.0 / d)
    return total

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy on R^3 (before projection).
    Energy = sum_{i<j} -log(|x_i - x_j|)
    d/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-30:
                continue
            d = math.sqrt(d2)
            # d/dx_i of -log(d) = -(1/d) * (diff/d) = -diff/d^2
            grad[i] -= diff / d2
    return grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _gradient_descent_on_sphere(pts0, max_iter=5000, tol=1e-12, deadline=None):
    """Gradient descent on the sphere with backtracking line search."""
    pts = pts0.copy().astype(float)
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    step_size = 0.01
    min_step = 1e-10
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient and project to tangent space
        grad = _log_energy_grad(pts)
        # Project: grad_tangent = grad - (grad . x) * x
        dot = np.einsum('ij,ij->i', grad, pts)
        grad_tangent = grad - dot[:, None] * pts
        
        # Normalize gradient direction
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
        
        direction = -grad_tangent / grad_norm  # descent direction
        
        # Backtracking line search
        alpha = step_size
        current_energy = best_energy
        improved = False
        for _ in range(30):
            new_pts = pts + alpha * direction
            new_pts = _project_to_sphere(new_pts)
            new_energy = _log_energy(new_pts)
            if new_energy < current_energy - 1e-15:
                improved = True
                break
            alpha *= 0.5
            if alpha < min_step:
                break
        
        if improved:
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
            # Increase step size slightly
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-12:
                break
    
    return best_pts, best_energy

def _fibonacci_spiral():
    """Golden-angle Fibonacci spiral on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(rng):
    """Random points on the unit sphere."""
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_based():
    """Start from icosahedron vertices and subdivide for better initial spread."""
    # Use Fibonacci spiral as a good baseline, then perturb
    pts = _fibonacci_spiral()
    rng = np.random.default_rng(42)
    # Add small random perturbation to break symmetry
    noise = rng.standard_normal(pts.shape) * 0.01
    pts = pts + noise
    return _project_to_sphere(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate multiple initial configurations
    rng = np.random.default_rng(12345)
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_spiral())
    
    # 2. Random points
    for _ in range(5):
        inits.append(_random_points(rng))
    
    # 3. Perturbed Fibonacci
    inits.append(_icosahedron_based())
    
    # 4. More random with different seeds
    for seed in [1, 7, 42, 123, 999]:
        r = np.random.default_rng(seed)
        inits.append(_random_points(r))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Coarse optimization with moderate iterations
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        pts, energy = _gradient_descent_on_sphere(
            init, max_iter=2000, tol=1e-10, deadline=deadline - 20.0
        )
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Polish the best result with tighter tolerance and more iterations
    if best_pts is not None and time.time() < deadline - 5.0:
        pts, energy = _gradient_descent_on_sphere(
            best_pts, max_iter=5000, tol=1e-13, deadline=deadline - 5.0
        )
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 3: Final ultra-precise polish
    if best_pts is not None and time.time() < deadline - 2.0:
        # Use smaller step sizes for final convergence
        pts = best_pts.copy()
        step_size = 0.001
        min_step = 1e-14
        for it in range(2000):
            if time.time() > deadline - 2.0:
                break
            grad = _log_energy_grad(pts)
            dot = np.einsum('ij,ij->i', grad, pts)
            grad_tangent = grad - dot[:, None] * pts
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-14:
                break
            direction = -grad_tangent / grad_norm
            alpha = step_size
            current_energy = _log_energy(pts)
            improved = False
            for _ in range(20):
                new_pts = pts + alpha * direction
                new_pts = _project_to_sphere(new_pts)
                new_energy = _log_energy(new_pts)
                if new_energy < current_energy - 1e-16:
                    improved = True
                    break
                alpha *= 0.5
                if alpha < min_step:
                    break
            if improved:
                pts = new_pts
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
                step_size = min(step_size * 1.05, 0.01)
            else:
                step_size *= 0.5
                if step_size < 1e-15:
                    break
    
    # Final normalization
    if best_pts is None:
        best_pts = _fibonacci_spiral()
    
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            best_pts = _fibonacci_spiral()
            break
    
    return best_pts
# EVOLVE-BLOCK-END

