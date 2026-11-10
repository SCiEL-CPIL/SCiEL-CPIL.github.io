"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
from math import log, fsum
import time
import sys

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import log, fsum
import time
import sys

N = 150
DEADLINE_FRACTION = 0.85  # safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30  # penalty for coincident points
            terms.append(-log(d))
    return fsum(terms)

def _energy_and_grad(pts):
    """
    Computes the logarithmic energy and its gradient projected to the tangent space.
    E = sum_{i<j} -log(||xi - xj||)
    dE/dxi = sum_{j!=i} (xi - xj) / ||xi - xj||^2
    """
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    mask = dists > 1e-12
    inv_d2 = np.zeros_like(dists)
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    
    # Gradient calculation
    # grad_i = sum_{j!=i} (xi - xj) / d_ij^2
    # Note: diff[i,j] = xi - xj
    # So term for grad_i is diff[i,j] * inv_d2[i,j]
    grad = np.sum(diff * inv_d2[:, np.newaxis, :], axis=1)
    
    # Project gradient to tangent space: g_tangent = g - (g . p) p
    for i in range(n):
        p = pts[i]
        grad[i] = grad[i] - np.dot(grad[i], p) * p
        
    # Energy calculation
    # Only sum upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    energy = -np.sum(np.log(d_upper))
    
    return energy, grad

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lbfgs_polish(pts, max_iter=200, deadline=None):
    """
    Performs a short L-BFGS-B polish to converge to a local minimum.
    """
    n = len(pts)
    
    def objective(x):
        p = _normalize(x.reshape(n, 3))
        e, _ = _energy_and_grad(p)
        return e
        
    def grad_func(x):
        p = _normalize(x.reshape(n, 3))
        _, g = _energy_and_grad(p)
        return g.flatten()
        
    try:
        res = minimize(
            objective, pts.flatten(), jac=grad_func, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        new_pts = _normalize(res.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts

def construct():
    start_time = time.time()
    total_time = 150.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Start with Fibonacci spiral.
    # 2. Run Simulated Annealing with periodic L-BFGS polishes.
    # 3. Track best energy and return best configuration.
    
    best_pts = _fibonacci_spiral(N)
    best_energy = _log_energy(best_pts)
    
    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 1e-4
    alpha = 0.99  # Cooling rate
    T = T_start
    
    # Number of steps between polishes
    steps_per_polish = 100
    current_step = 0
    
    while time.time() < deadline:
        current_step += 1
        
        # Perform a random perturbation
        # Perturb a random subset of points to allow larger moves
        num_perturb = max(1, N // 10)
        idx = rng.choice(N, num_perturb, replace=False)
        
        # Generate random directions in tangent space
        # For simplicity, perturb in ambient space and re-normalize, 
        # but scale by sqrt(T) to control step size
        noise = rng.standard_normal((num_perturb, 3)) * np.sqrt(T)
        
        new_pts = best_pts.copy()
        new_pts[idx] += noise
        new_pts = _normalize(new_pts)
        
        new_energy, _ = _energy_and_grad(new_pts)
        
        # Acceptance criterion
        if new_energy < best_energy:
            best_pts = new_pts
            best_energy = new_energy
        else:
            # Accept with probability exp(-delta_E / T)
            delta_E = new_energy - best_energy
            if delta_E < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_E / T)
                
            if rng.random() < accept_prob:
                best_pts = new_pts
                best_energy = new_energy
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            
        # Periodic Polish
        if current_step % steps_per_polish == 0:
            polished_pts = _lbfgs_polish(best_pts, max_iter=50, deadline=deadline)
            polished_energy, _ = _energy_and_grad(polished_pts)
            
            if polished_energy < best_energy:
                best_pts = polished_pts
                best_energy = polished_energy
                
            # Check time
            if time.time() > deadline:
                break
    
    # Final Polish
    if time.time() < deadline - 5:
        final_pts = _lbfgs_polish(best_pts, max_iter=200, deadline=deadline)
        final_energy, _ = _energy_and_grad(final_pts)
        
        if final_energy < best_energy:
            best_pts = final_pts
            best_energy = final_energy
            
    return best_pts
# EVOLVE-BLOCK-END
