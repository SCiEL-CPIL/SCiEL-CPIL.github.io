"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            d = np.linalg.norm(pi - points[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _grad_log_energy_tangent(points):
    """Compute gradient of log energy projected to tangent space of sphere.
    
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{k != i} (x_i - x_k) / d_ik^2
    
    Then project to tangent space: grad_tangent = grad - (grad . x_i) * x_i
    """
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        xi = points[i]
        for k in range(n):
            if k == i:
                continue
            diff = xi - points[k]
            d2 = np.dot(diff, diff)
            if d2 < 1e-24:
                return None
            grad[i] -= diff / d2
        # Project to tangent space (orthogonal to xi)
        grad[i] -= np.dot(grad[i], xi) * xi
    return grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).astype(np.float64)


def _random_sphere(n, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _gradient_descent(points, deadline, max_iters=500, lr_init=0.01):
    """Run projected gradient descent with tangent-space gradient."""
    pts = points.copy()
    energy = _log_energy(pts)
    lr = lr_init
    lr_decay = 0.995
    lr_shrink = 0.5
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        grad = _grad_log_energy_tangent(pts)
        if grad is None:
            # Points too close, perturb
            pts = pts + 0.01 * np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
            continue
        
        # Try the step
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy = _log_energy(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            lr *= lr_decay
        else:
            lr *= lr_shrink
            if lr < 1e-10:
                break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
    
    return pts, energy


def _polish_with_lbfgs(points, deadline):
    """Final high-precision polish using L-BFGS-B with sphere constraint via parameterization.
    
    We use the fact that for small perturbations, we can work in local coordinates.
    Alternatively, use a penalty-free approach: minimize over angles with tight tolerance.
    
    Here we use a simpler approach: run a few more GD iterations with very small lr,
    then try scipy minimize with a constraint.
    """
    pts = points.copy()
    energy = _log_energy(pts)
    
    # First, do a very fine gradient descent polish
    lr = 0.001
    for it in range(200):
        if time.time() > deadline:
            break
        grad = _grad_log_energy_tangent(pts)
        if grad is None:
            break
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy = _log_energy(new_pts)
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            lr *= 0.99
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return pts, energy


def construct():
    """Place N points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Start with Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    # Phase 1: Gradient descent from Fibonacci initialization
    pts, energy = _gradient_descent(best_pts, deadline - 30.0, max_iters=300, lr_init=0.01)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts.copy()
    
    # Phase 2: Try random restarts with gradient descent
    np.random.seed(12345)
    for attempt in range(10):
        if time.time() > deadline - 20.0:
            break
        rand_pts = _random_sphere(N, seed=attempt * 7919 + 13)
        pts, energy = _gradient_descent(rand_pts, deadline - 15.0, max_iters=200, lr_init=0.005)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 3: Final polish with tighter tolerances
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _polish_with_lbfgs(best_pts, deadline - 2.0)
    
    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Check minimum distance
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9 or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
