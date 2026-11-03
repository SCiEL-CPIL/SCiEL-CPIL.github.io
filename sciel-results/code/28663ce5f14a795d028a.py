# EVOLVE-BLOCK-START
import numpy as np
import math
import random


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10  # penalty for overlapping points
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _gradient_energy(points):
    """Compute gradient of logarithmic energy w.r.t. each point."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # Gradient contribution to point i: -(x_i - x_j)/d^2
            # Gradient contribution to point j: +(x_i - x_j)/d^2
            term = diff / (d * d)
            grad[i] -= term
            grad[j] += term
    return grad


def _project_to_sphere(points):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _local_refinement(points, max_iters=500):
    """Perform local gradient descent refinement."""
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    
    for _ in range(max_iters):
        grad = _gradient_energy(best_points)
        # Project gradient to tangent space
        g_tangent = grad - np.sum(grad * best_points, axis=1, keepdims=True) * best_points
        
        # Adaptive learning rate
        lr = 1e-5
        # Try a few step sizes if the first one fails
        improved = False
        for lr_try in [1e-4, 1e-5, 1e-6, 1e-7]:
            new_points = best_points - lr_try * g_tangent
            new_points = _project_to_sphere(new_points)
            new_energy = _log_energy(new_points)
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points
                improved = True
                break
        
        if not improved:
            break
            
    return best_points, best_energy


def construct():
    N = 150
    np.random.seed(42)
    random.seed(42)
    
    # Initial configuration: Fibonacci spiral is a good start, but we will anneal from a perturbed version
    # to allow the annealing to explore different basins if needed, or just start from it.
    # Let's start from a random configuration to ensure we don't bias towards the spiral's specific local min
    # immediately, but a random config on sphere is often far from optimal.
    # A better strategy: Start from Fibonacci, but run SA with high T to "shake" it out if needed, 
    # or just start from Fibonacci and refine. 
    # Given the time limit (120s), we should be careful. 
    # Let's start with a perturbed Fibonacci spiral to give SA a head start but still allow movement.
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add initial noise
    points = points + np.random.normal(0, 0.1, size=(N, 3))
    points = _project_to_sphere(points)
    
    current_energy = _log_energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 1e-6
    n_iterations = 200000  # Adjust based on time limit
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    # Precompute indices for speed? No, N is small enough.
    
    for it in range(n_iterations):
        T = T_start * (cooling_rate ** it)
        
        # Select a random point to move
        idx = random.randint(0, N - 1)
        
        # Generate a random move on the sphere
        # Add a small random vector and re-normalize
        # The magnitude of the move should scale with T or be constant?
        # Constant small moves are safer for convergence.
        move_size = 0.1 
        # Scale move size with temperature? 
        # Larger T -> larger moves to escape minima.
        # move_size = 0.01 + 0.1 * (T / T_start)
        
        new_points = points.copy()
        new_points[idx] += np.random.normal(0, move_size, size=3)
        new_points[idx] /= np.linalg.norm(new_points[idx])
        
        new_energy = _log_energy(new_points)
        delta_E = new_energy - current_energy
        
        if delta_E < 0:
            # Accept improvement
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            # Accept with probability exp(-delta_E / T)
            if random.random() < math.exp(-delta_E / T):
                points = new_points
                current_energy = new_energy
            else:
                # Reject: points remains unchanged
                pass
                
    # Final local refinement
    best_points, best_energy = _local_refinement(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
