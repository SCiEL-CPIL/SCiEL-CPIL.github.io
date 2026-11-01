"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)"""
    # Use math.fsum for accuracy
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10  # invalid
            terms.append(-math.log(d))
    return math.fsum(terms)


def _log_energy_grad(points):
    """Compute energy and gradient on the sphere.
    
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient of E w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    This is the gradient in R^3. For Riemannian optimization, we project to tangent space.
    """
    n = len(points)
    energy = 0.0
    grad = np.zeros_like(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return 1e10, grad
            # -log(d) contributes to energy
            energy += -math.log(d)
            # Gradient of -log(d) w.r.t. x_i: (x_i - x_j) / d^2
            # Gradient of -log(d) w.r.t. x_j: (x_j - x_i) / d^2
            factor = 1.0 / (d * d)
            grad[i] += factor * diff
            grad[j] -= factor * diff
    
    return energy, grad


def _project_tangent(points, grad):
    """Project gradient to tangent space of the sphere."""
    # Tangent projection: g_tan = g - (g · x) * x for unit vectors
    tangent = grad.copy()
    for i in range(len(points)):
        proj = np.dot(grad[i], points[i])
        tangent[i] = grad[i] - proj * points[i]
    return tangent


def _retract_sphere(points, tangent, step):
    """Retraction: move along tangent then normalize."""
    new_points = points + step * tangent
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-15, 1.0, norms)
    return new_points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Start with Fibonacci sphere
    points = _fibonacci_sphere(N).astype(np.float64)
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    
    # Riemannian conjugate gradient optimization
    # We'll implement a simple RCG with line search
    
    def rcg_optimize(pts, max_iter=500, tol=1e-10):
        nonlocal best_points, best_energy
        pts = pts.copy()
        energy, grad = _log_energy_grad(pts)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
        
        # Initial search direction
        tangent = _project_tangent(pts, grad)
        # Negative gradient direction
        d = -tangent
        
        # Beta for conjugate gradient
        beta = 0.0
        
        for iteration in range(max_iter):
            # Check time
            if time.time() > deadline:
                break
            
            # Line search: backtracking
            alpha = 0.1
            min_alpha = 1e-12
            max_line_search = 30
            success = False
            
            for ls in range(max_line_search):
                new_pts = _retract_sphere(pts, d, alpha)
                new_energy, _ = _log_energy_grad(new_pts)
                
                # Check for improvement (Armijo condition simplified)
                if new_energy < energy - 1e-4 * alpha * np.sum(tangent * d):
                    # Accept step
                    # Update beta
                    old_tangent = tangent
                    pts = new_pts
                    energy, grad = _log_energy_grad(pts)
                    tangent = _project_tangent(pts, grad)
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_points = pts.copy()
                    
                    # Check convergence
                    if np.linalg.norm(tangent) < tol:
                        break
                    
                    # Compute beta (Polak-Ribiere)
                    beta = np.sum(tangent * (tangent - old_tangent)) / max(np.sum(old_tangent * old_tangent), 1e-15)
                    beta = max(0.0, beta)
                    d = -tangent + beta * d
                    success = True
                    break
                else:
                    alpha *= 0.5
            
            if not success:
                # Reset direction if line search failed
                d = -tangent
                beta = 0.0
            
            # Check time
            if time.time() > deadline:
                break
        
        return pts, energy
    
    # Run RCG
    points, energy = rcg_optimize(points)
    
    # If time remains, try a few more restarts with perturbations
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the best points
            perturbed = best_points.copy()
            noise = rng.normal(0, 0.1, size=perturbed.shape)
            # Project noise to tangent space
            for i in range(N):
                proj = np.dot(noise[i], perturbed[i])
                noise[i] -= proj * perturbed[i]
            perturbed = _retract_sphere(perturbed, noise, 1.0)
            
            # Run RCG from perturbed start
            _, e = rcg_optimize(perturbed, max_iter=200)
    
    # Final check: ensure points are on unit sphere and valid
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms
    
    # Check minimum distance
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-9:
                # Points too close, use original Fibonacci
                return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
