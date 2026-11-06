"""Optimized logarithmic energy minimizer for 100 points on the unit sphere."""
import numpy as np
import math
import time

N = 100
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _energy(points):
    """Compute log(1/d) energy sum over all pairs using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_grad(points):
    """Compute gradient of log(1/d) energy w.r.t. point positions."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_vec = points[i] - points[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-15:
                continue
            # d/dx_i log(1/|x_i - x_j|) = -d_vec / d^2
            grad[i] += -d_vec / (d * d)
    return grad

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _spherical_harmonic_density(n_harm, theta, phi):
    """
    Evaluate a low-order spherical harmonic density on the sphere.
    Uses real spherical harmonics up to degree n_harm.
    Returns density values (non-negative after softmax).
    """
    # For simplicity, use a truncated expansion: rho(theta, phi) = exp(sum_{l,m} c_{lm} Y_{lm}(theta,phi))
    # We'll use a simple basis: just the first few l modes
    # Y_00 = 1/sqrt(4*pi)
    # For l=1: Y_10 ~ cos(theta), Y_11 ~ sin(theta)cos(phi), Y_-11 ~ sin(theta)sin(phi)
    # For l=2: quadrupole terms
    # This is a simplified approach; we optimize coefficients directly
    
    # Actually, let's use a different approach: parameterize the density via a small number of 
    # Gaussian-like bumps on the sphere, or use the fact that for 100 points, we want roughly uniform.
    # The optimal distribution for log energy is close to uniform (since log kernel is weakly singular).
    # So the spherical harmonic approach may not give much benefit over just using a good initialization.
    # Let me instead focus on a robust discrete optimization with multiple restarts.
    pass

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5 + offset * n
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=0):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _gradient_descent(points, max_iters=500, lr=0.01, tol=1e-10):
    """Projected gradient descent with backtracking line search."""
    points = _project_to_sphere(points.copy())
    current_energy = _energy(points)
    
    for it in range(max_iters):
        grad = _energy_grad(points)
        
        # Project gradient: remove component along radial direction
        # For sphere constraint, the projected gradient is grad - (grad . x) * x
        proj_grad = grad.copy()
        for i in range(len(points)):
            proj_grad[i] -= np.dot(grad[i], points[i]) * points[i]
        
        # Backtracking line search
        step = lr
        best_points = points
        best_energy = current_energy
        
        for _ in range(30):
            new_points = _project_to_sphere(points - step * proj_grad)
            new_energy = _energy(new_points)
            if new_energy < best_energy - 1e-12:
                best_points = new_points
                best_energy = new_energy
                break
            step *= 0.5
        
        if np.allclose(best_points, points, atol=1e-12):
            break
        
        points = best_points
        current_energy = best_energy
        
        if abs(current_energy - best_energy) < tol:
            break
    
    return points, current_energy

def construct():
    start_time = time.time()
    
    # Multiple restart strategies
    candidates = []
    
    # 1. Fibonacci spiral
    pts = _fibonacci_spiral(N)
    candidates.append(pts)
    
    # 2. Fibonacci with small perturbations
    for offset in [0.01, 0.05, 0.1, 0.2]:
        pts = _fibonacci_spiral(N, offset=offset)
        candidates.append(pts)
    
    # 3. Random initializations with different seeds
    for seed in [1, 42, 123, 777, 2024]:
        pts = _random_points(N, seed=seed)
        candidates.append(pts)
    
    # 4. Spherical harmonic-inspired: use low-frequency perturbations of uniform distribution
    # Create a grid of low-frequency modes and perturb the Fibonacci points
    base = _fibonacci_spiral(N)
    for freq in [2, 3, 5]:
        for amp in [0.01, 0.05]:
            # Add a small sinusoidal perturbation in the angular coordinates
            theta = np.arccos(np.clip(base[:, 2], -1, 1))
            phi = np.arctan2(base[:, 1], base[:, 0])
            theta_p = theta + amp * np.sin(freq * theta) * np.cos(freq * phi)
            phi_p = phi + amp * np.cos(freq * theta) * np.sin(freq * phi)
            r = np.sqrt(np.maximum(0.0, 1.0 - np.cos(theta_p)**2))
            pts = np.column_stack([
                r * np.cos(phi_p),
                r * np.sin(phi_p),
                np.cos(theta_p)
            ])
            pts = _project_to_sphere(pts)
            candidates.append(pts)
    
    # Optimize each candidate with gradient descent
    best_points = None
    best_energy = float('inf')
    
    for idx, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:  # Leave 10s margin
            break
        
        # Use fewer iterations for later candidates if time is running out
        remaining = DEADLINE - 10 - elapsed
        max_iters = min(500, max(50, int(remaining * 50)))
        
        optimized, energy = _gradient_descent(init_pts, max_iters=max_iters, lr=0.005, tol=1e-12)
        
        if energy < best_energy:
            best_energy = energy
            best_points = optimized
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Final high-precision polish if time permits
    if best_points is not None and time.time() - start_time < DEADLINE - 15:
        # Run a few more iterations with tighter tolerance
        best_points, best_energy = _gradient_descent(best_points, max_iters=200, lr=0.001, tol=1e-14)
    
    # Ensure validity
    best_points = _project_to_sphere(best_points)
    
    # Final check: no points too close
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            min_dist = min(min_dist, d)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_spiral(N)
    
    return best_points
