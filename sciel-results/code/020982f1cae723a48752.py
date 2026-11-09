"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math


def _log_energy(pts):
    """Compute log(1/d) energy sum over all pairs using chordal distances."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    # log(1/d) = -0.5 * log(d_sq)
    d_sq = np.clip(d_sq, 1e-30, None)
    terms = [-0.5 * math.log(di) for di in d_sq]
    return math.fsum(terms)


def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_and_grad(x):
    """Compute energy and gradient for L-BFGS. x is (N,3) flattened."""
    pts = x.reshape(N, 3)
    # Project to sphere for correct gradient
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    
    # Compute pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.clip(dist_sq, 1e-15, None)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(d_ij^2)
    # = -0.5 * sum_{i<j} log(d_ij^2)
    iu = np.triu_indices(N, k=1)
    d_sq_ij = dist_sq[iu]
    energy = -0.5 * np.sum(np.log(d_sq_ij))
    
    # Gradient: dE/dx_i where E = -0.5 * sum_{i<j} log(|x_i - x_j|^2)
    # dE/dx_i = -0.5 * sum_{j!=i} d/dx_i log(|x_i - x_j|^2)
    #         = -0.5 * sum_{j!=i} 2*(x_i - x_j)/|x_i - x_j|^2
    #         = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    # 
    # But we need to account for the projection constraint. Since we're 
    # optimizing on the sphere, we use the projected gradient.
    # For simplicity, we compute the gradient of the energy w.r.t. pts
    # and then project.
    
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d_sq = dist_sq[i, j]
            grad[i] += (pts[j] - pts[i]) / d_sq
    
    # Project gradient to tangent space of sphere
    # grad_tangent = grad - (grad . x_i / |x_i|^2) * x_i
    # Since |x_i| = 1, grad_tangent = grad - (grad . x_i) * x_i
    for i in range(N):
        dot = np.dot(grad[i], pts[i])
        grad[i] -= dot * pts[i]
    
    return energy, grad.flatten()


def _optimize_lbfgs(start_pts, deadline, maxiter=10000):
    """Run L-BFGS optimization on the sphere."""
    x0 = start_pts.flatten()
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        e, _ = _energy_and_grad(x)
        return e
    
    def jac(x):
        if time.time() > deadline:
            return np.ones(N * 3) * 1e6
        _, g = _energy_and_grad(x)
        return g
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                                'ftol': 1e-12, 'gtol': 1e-8})
        result = _project_to_sphere(res.x.reshape(N, 3))
        return result
    except Exception:
        return None


def _anneal_optimize(start_pts, deadline):
    """Simulated annealing style optimization."""
    pts = start_pts.copy()
    energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    T = 0.1
    cooling = 0.99
    step_size = 0.05
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Random perturbation
        idx = np.random.randint(N)
        noise = np.random.randn(3) * step_size
        new_pts = pts.copy()
        new_pts[idx] += noise
        new_pts[idx] /= np.linalg.norm(new_pts[idx])
        
        new_energy = _log_energy(new_pts)
        delta_e = new_energy - energy
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        T *= cooling
        step_size = max(0.005, step_size * 0.995)
    
    return best_pts


def construct():
    """Main entry point: generate many candidates, optimize, keep best."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    fib_pts = _initial_fibonacci()
    fib_energy = _log_energy(fib_pts)
    best_pts = fib_pts
    best_energy = fib_energy
    
    # Try multiple random initializations with L-BFGS
    np.random.seed(12345)
    
    n_random_starts = 8
    for i in range(n_random_starts):
        if time.time() > deadline - 20:
            break
        
        # Random points on sphere
        random_pts = np.random.randn(N, 3)
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        result = _optimize_lbfgs(random_pts, deadline - 15, maxiter=3000)
        if result is not None:
            energy = _log_energy(result)
            if energy < best_energy:
                best_energy = energy
                best_pts = result
    
    # Optimize the Fibonacci start
    if time.time() < deadline - 20:
        result = _optimize_lbfgs(fib_pts, deadline - 15, maxiter=5000)
        if result is not None:
            energy = _log_energy(result)
            if energy < best_energy:
                best_energy = energy
                best_pts = result
    
    # Annealing refinement
    if time.time() < deadline - 10:
        refined = _anneal_optimize(best_pts, deadline - 5)
        refined_energy = _log_energy(refined)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_pts = refined
    
    # Final L-BFGS polish
    if time.time() < deadline - 5:
        result = _optimize_lbfgs(best_pts, deadline - 2, maxiter=2000)
        if result is not None:
            energy = _log_energy(result)
            if energy < best_energy:
                best_energy = energy
                best_pts = result
    
    # Final normalization and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = fib_pts
    
    return best_pts
# EVOLVE-BLOCK-END
