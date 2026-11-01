"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(points):
    """
    Calculate the logarithmic energy and its gradient.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    d_ij = ||p_i - p_j||
    
    Gradient with respect to p_i:
    grad_i = - sum_{j != i} (p_i - p_j) / d_ij^2
    
    This is the gradient in Euclidean space.
    To project onto the tangent space of the sphere (since points must remain on the sphere),
    we project the gradient: grad_tangent = grad - (grad . p_i) * p_i
    """
    if not np.all(np.isfinite(points)):
        return float('inf'), np.zeros_like(points)
    
    # Compute pairwise distances
    # dists[i, j] = ||p_i - p_j||
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Check for invalid close points
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf'), np.zeros_like(points)
        
    # Energy
    # Only sum upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient
    # grad[i] = - sum_{j != i} (p_i - p_j) / d_ij^2
    grad = np.zeros_like(points)
    for i in range(N):
        diff = points[i] - points
        dist_sq = dists[i, :]**2
        # Avoid division by zero just in case, though diagonal is 1.
        dist_sq[i] = 1.0 
        term = diff / dist_sq[:, np.newaxis]
        term[i] = 0.0
        grad[i] = -np.sum(term, axis=0)
        
    # Project gradient onto tangent space
    # grad_tangent[i] = grad[i] - (grad[i] . points[i]) * points[i]
    for i in range(N):
        dot = np.dot(grad[i], points[i])
        grad[i] -= dot * points[i]
        
    return energy, grad

def _energy(points):
    """Calculate the logarithmic energy. Returns inf if invalid."""
    e, _ = _energy_and_grad(points)
    return e

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n):
    """Generate random points on the sphere."""
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def _riemannian_grad_descent(points, deadline, max_iter=200, tol=1e-8):
    """
    Perform Riemannian gradient descent on the sphere.
    Uses a backtracking line search to ensure convergence.
    """
    pts = points.copy()
    current_energy, current_grad = _energy_and_grad(pts)
    if not np.isfinite(current_energy):
        return pts, current_energy
    
    step_size = 0.1
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Take a step in the negative gradient direction
        # proj_grad is already tangent
        new_pts = pts - step_size * current_grad
        
        # Project back to sphere
        norms = np.linalg.norm(new_pts, axis=1)
        if np.any(norms < 1e-9):
            step_size *= 0.5
            continue
        new_pts = new_pts / norms[:, np.newaxis]
        
        # Check energy
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if np.isfinite(new_energy) and new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
            current_grad = new_grad
            step_size = min(step_size * 1.5, 1.0)
            
            # Check convergence
            if np.linalg.norm(current_grad) < tol:
                break
        else:
            # Backtrack
            step_size *= 0.5
            if step_size < 1e-10:
                break
                
    return pts, current_energy

def _basin_hopping(points, deadline, num_restarts=10, max_iterations=50):
    """
    Perform Basin Hopping optimization.
    1. Start from initial points.
    2. Local minimize using Riemannian GD.
    3. Perturb points (kick).
    4. Local minimize again.
    5. Accept if energy is lower or with probability exp(-delta/T).
    """
    best_pts = points.copy()
    best_energy = _energy(best_pts)
    
    # Simple simulated annealing schedule
    T = 1.0
    cooling = 0.95
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Kick: perturb points
        # Use random rotation or small Gaussian noise projected to tangent space
        noise = np.random.randn(N, 3) * 0.1
        # Project noise to tangent space
        for i in range(N):
            dot = np.dot(noise[i], best_pts[i])
            noise[i] -= dot * best_pts[i]
        
        kicked_pts = best_pts + noise
        # Normalize
        norms = np.linalg.norm(kicked_pts, axis=1)
        if np.any(norms < 1e-9):
            continue
        kicked_pts = kicked_pts / norms[:, np.newaxis]
        
        # Local minimize the kicked points
        new_pts, new_energy = _riemannian_grad_descent(kicked_pts, deadline, max_iter=50)
        
        if not np.isfinite(new_energy):
            continue
            
        # Metropolis acceptance criterion
        if new_energy < best_energy:
            best_pts = new_pts
            best_energy = new_energy
        else:
            delta = new_energy - best_energy
            if np.random.rand() < math.exp(-delta / T):
                best_pts = new_pts
                best_energy = new_energy
                
        T *= cooling
        
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Initialize with Fibonacci sphere
    best_points = _fibonacci_sphere(N)
    best_energy = _energy(best_points)
    
    # First, refine the initial guess with Riemannian GD
    refined_points, refined_energy = _riemannian_grad_descent(best_points, deadline, max_iter=200)
    if refined_energy < best_energy:
        best_points = refined_points
        best_energy = refined_energy
        
    # Now perform Basin Hopping from this refined state
    # We will run multiple independent BH runs with different random seeds/initial kicks
    # But to save time, we start from the best known point.
    
    # Let's try a few different random initializations as well, refined then BH
    # Strategy:
    # 1. Refine Fibonacci
    # 2. Run BH on Refined Fibonacci
    # 3. If time permits, try random starts:
    #    - Generate random start
    #    - Refine with GD
    #    - If better than current best, update best
    #    - Run short BH on it? Or just keep the refined one?
    #    Actually, BH is expensive. Let's do a few short BH runs.
    
    # Time check
    if time.time() < deadline - 20.0:
        bh_points, bh_energy = _basin_hopping(best_points, deadline - 5.0, num_restarts=1, max_iterations=30)
        if bh_energy < best_energy:
            best_points = bh_points
            best_energy = bh_energy
            
    # Try a few random starts if time remains
    attempts = 5
    for i in range(attempts):
        if time.time() > deadline - 10.0:
            break
            
        random_pts = _random_sphere(N)
        # Quick refine
        refined_rand, rand_energy = _riemannian_grad_descent(random_pts, deadline - 5.0, max_iter=100)
        
        if rand_energy < best_energy:
            best_points = refined_rand
            best_energy = rand_energy
            
            # If we found a new best, maybe do a quick BH on it?
            if time.time() < deadline - 15.0:
                bh_rand, bh_rand_energy = _basin_hopping(best_points, deadline - 5.0, num_restarts=1, max_iterations=20)
                if bh_rand_energy < best_energy:
                    best_points = bh_rand
                    best_energy = bh_rand_energy
                    
    # Final polish with L-BFGS-B if time allows
    if time.time() < deadline - 2.0:
        def objective(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1)
            if np.any(norms < 1e-9):
                return float('inf'), np.zeros_like(x)
            pts = pts / norms[:, np.newaxis]
            e, g = _energy_and_grad(pts)
            if not np.isfinite(e):
                return float('inf'), np.zeros_like(x)
            return e, g.flatten()
            
        res = minimize(objective, best_points.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 100, 'gtol': 1e-8})
        
        if res.success or np.isfinite(res.fun):
            final_pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(final_pts, axis=1)
            if np.all(norms > 1e-9):
                final_pts = final_pts / norms[:, np.newaxis]
                final_energy = _energy(final_pts)
                if final_energy < best_energy:
                    best_points = final_pts
                    best_energy = final_energy

    return best_points
# EVOLVE-BLOCK-END
