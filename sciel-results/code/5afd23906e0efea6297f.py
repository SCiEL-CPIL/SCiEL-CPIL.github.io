"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for N points on sphere."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)  # Avoid log(0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    energy = -np.log(dist).sum() / 2.0  # Each pair counted twice
    
    # Gradient: d/dx_i [log(1/d_ij)] = (x_i - x_j) / d_ij^2
    # For all pairs i<j, gradient contribution to x_i is (x_i - x_j)/d_ij^2
    # And to x_j is (x_j - x_i)/d_ij^2
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j (x_i - x_j) / d_ij^2
    # = sum_j x_i / d_ij^2 - sum_j x_j / d_ij^2
    # = x_i * sum_j 1/d_ij^2 - sum_j x_j / d_ij^2
    sum_inv_dist2 = inv_dist2.sum(axis=1)  # Shape (n,)
    # Weighted sum: sum_j x_j / d_ij^2
    weighted_sum = (inv_dist2[:, :, np.newaxis] * pts[np.newaxis, :, :]).sum(axis=1)
    grad = pts * sum_inv_dist2[:, np.newaxis] - weighted_sum
    
    # Project gradient onto tangent space: remove radial component
    radial_comp = (grad * pts).sum(axis=1, keepdims=True)
    grad_tangent = grad - radial_comp * pts
    
    return energy, grad_tangent

def _optimize(pts, max_iter=500, lr_init=0.01, lr_min=1e-6, tol=1e-10):
    """Optimize points on sphere using gradient descent with tangent projection."""
    pts = pts.copy()
    n = len(pts)
    energy, grad = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    for it in range(max_iter):
        # Update in tangent space
        step = lr_init * np.exp(-0.01 * it)
        if step < lr_min:
            step = lr_min
        
        new_pts = pts - step * grad
        # Re-normalize to sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        if abs(new_energy - energy) < tol:
            break
        
        energy = new_energy
        grad = new_grad
        pts = new_pts
    
    return best_pts, best_energy

def _simulated_annealing(pts, t_start=0.1, t_end=0.001, max_iter=200, cooling_rate=0.98):
    """Refine with simulated annealing to escape local minima."""
    pts = pts.copy()
    n = len(pts)
    energy, _ = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    t = t_start
    for it in range(max_iter):
        # Generate random perturbation in tangent space
        noise = np.random.randn(n, 3)
        # Project to tangent space
        radial_comp = (noise * pts).sum(axis=1, keepdims=True)
        noise_tangent = noise - radial_comp * pts
        noise_tangent *= 0.01 * t  # Scale by temperature
        
        new_pts = pts + noise_tangent
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        new_energy, _ = _energy_and_grad(new_pts)
        delta_e = new_energy - energy
        
        # Accept with probability exp(-delta_e / t)
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / t):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        t *= cooling_rate
        if t < t_end:
            break
    
    return best_pts, best_energy

def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci sphere points with random rotation."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Apply random rotation
    theta = rng.uniform(0, 2 * np.pi)
    phi_rot = rng.uniform(-np.pi/2, np.pi/2)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi_rot), np.sin(phi_rot)
    
    # Rotation matrix
    R = np.array([
        [cos_t, sin_t, 0],
        [-sin_t, cos_t, 0],
        [0, 0, 1]
    ])
    R2 = np.array([
        [1, 0, 0],
        [0, cos_p, sin_p],
        [0, -sin_p, cos_p]
    ])
    R_total = R2 @ R
    pts = pts @ R_total.T
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Multiple restarts with different initializations
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Fibonacci sphere
            pts = _fibonacci_sphere(N, seed=restart)
        elif restart < 5:
            # Random points
            rng = np.random.RandomState(restart * 100 + 42)
            pts = rng.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart < 10:
            # Fibonacci with different seeds
            pts = _fibonacci_sphere(N, seed=restart + 100)
        else:
            # Clustered starts
            rng = np.random.RandomState(restart * 200 + 7)
            centers = rng.randn(5, 3)
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
            pts = []
            for i in range(N):
                center = centers[i % 5]
                noise = rng.randn(3) * 0.1
                pt = center + noise
                pt = pt / np.linalg.norm(pt)
                pts.append(pt)
            pts = np.array(pts)
        
        # Optimize
        try:
            opt_pts, opt_energy = _optimize(pts, max_iter=300)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
        except Exception:
            continue
        
        # Simulated annealing refinement
        if time.time() < deadline - 5:
            try:
                sa_pts, sa_energy = _simulated_annealing(opt_pts, t_start=0.05, t_end=0.0005, max_iter=100)
                if sa_energy < best_energy:
                    best_energy = sa_energy
                    best_pts = sa_pts.copy()
            except Exception:
                continue
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N, seed=0)
    
    return best_pts
# EVOLVE-BLOCK-END
