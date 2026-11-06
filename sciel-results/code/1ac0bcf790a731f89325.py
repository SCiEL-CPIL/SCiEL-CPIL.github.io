"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e100
            total += math.log(1.0 / d)
    return math.fsum([total])  # just to use fsum as per spec, but we already summed

def _energy_fast(points):
    """Fast vectorized energy computation."""
    n = len(points)
    # Compute all pairwise distances
    # points is (n, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # We only want upper triangle, i < j
    iu = np.triu_indices(n, k=1)
    vals = dists[iu]
    # Avoid log of very small values
    vals = np.maximum(vals, 1e-15)
    return math.fsum(np.log(1.0 / vals).tolist())

def _energy_and_grad(points):
    """Compute energy and gradient. Gradient of sum_{i<j} log(1/|x_i-x_j|) 
    = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2 for each point i."""
    n = len(points)
    energy = 0.0
    grad = np.zeros_like(points)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return 1e100, grad
            # Energy term: log(1/d) = -log(d)
            # Each pair counted twice in the double loop, so we divide by 2 at the end
            # Or we can just compute sum over i<j
            # Let's do i<j to be cleaner
            pass
    
    # Redo with i<j
    energy = 0.0
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return 1e100, grad
            energy += math.log(1.0 / d)
            # dE/dd = -1/d
            # d(dd)/dx_i = (x_i - x_j)/d
            # So dE/dx_i = (-1/d) * (x_i - x_j)/d = -(x_i - x_j)/d^2
            # dE/dx_j = (-1/d) * (x_j - x_i)/d = (x_i - x_j)/d^2
            factor = 1.0 / (d * d)
            grad[i] -= factor * diff
            grad[j] += factor * diff
    
    return energy, grad

def _energy_and_grad_fast(points):
    """Vectorized energy and gradient."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of log(1/d_ij)
    iu = np.triu_indices(n, k=1)
    d_vals = dists[iu]
    d_vals = np.maximum(d_vals, 1e-15)
    energy = math.fsum(np.log(1.0 / d_vals).tolist())
    
    # Gradient: for each i, grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = -sum_{j != i} diff[i,j] / dists[i,j]^2
    inv_d2 = np.zeros_like(dists)
    inv_d2[iu] = 1.0 / (d_vals ** 2)
    iu_rev = (iu[1], iu[0])
    inv_d2[iu_rev] = inv_d2[iu]
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_d2[i,j]
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)

def _projected_gradient_descent(points, max_iter=2000, lr=0.01, tol=1e-10, deadline=None):
    """Projected gradient descent on the sphere."""
    n = len(points)
    best_energy = _energy_and_grad_fast(points)[0]
    best_points = points.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad_fast(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . x) x
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Adaptive step size
        # Use backtracking line search
        step = lr
        for _ in range(20):
            new_points = points - step * grad_tangent
            new_points = _project_to_sphere(new_points)
            
            new_energy = _energy_and_grad_fast(new_points)[0]
            if new_energy < energy:
                points = new_points
                break
            step *= 0.5
        
        # Slowly decay learning rate
        if it % 100 == 99:
            lr *= 0.95
    
    return best_points, best_energy

def _simulated_annealing(points, max_iter=5000, t_start=0.1, t_end=0.001, deadline=None):
    """Simple simulated annealing on the sphere."""
    n = len(points)
    best_energy = _energy_and_grad_fast(points)[0]
    best_points = points.copy()
    current_energy = best_energy
    current_points = points.copy()
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        rng = np.random.RandomState(it + 42)
        perturbation = rng.randn(n, 3) * 0.01
        new_points = _project_to_sphere(current_points + perturbation)
        
        new_energy = _energy_and_grad_fast(new_points)[0]
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / t):
            current_points = new_points
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points.copy()
        
        t *= cooling
    
    return best_points, best_energy

def construct():
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Start from Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points, best_energy = _projected_gradient_descent(points, max_iter=3000, lr=0.02, tol=1e-12, deadline=deadline)
    
    # Try multiple random restarts if time permits
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        seed = 12345 + restart * 1000
        rand_points = _random_points(N, seed)
        
        # Short gradient descent from random start
        pts, e = _projected_gradient_descent(rand_points, max_iter=1000, lr=0.01, tol=1e-10, deadline=deadline)
        
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        best_points, best_energy = _projected_gradient_descent(best_points, max_iter=2000, lr=0.005, tol=1e-14, deadline=deadline)
    
    # Ensure valid output
    best_points = _project_to_sphere(best_points)
    
    # Check for any points too close
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-9:
                # Nudge them apart
                best_points[j] += np.random.randn(3) * 1e-6
                best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
