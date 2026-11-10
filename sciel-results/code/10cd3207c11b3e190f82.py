"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
from math import log, fsum
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import log, fsum
import time

N = 150
DEADLINE_FRACTION = 0.85  # safety margin

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

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(pts):
    """
    Compute vectorized log energy and its gradient on the sphere.
    Returns: energy (float), grad (N, 3) projected to tangent space.
    """
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} -log(d_ij)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient of E w.r.t. pts:
    # dE/dp_i = sum_{j!=i} (p_i - p_j) / ||p_i - p_j||^2
    # We compute the matrix G where G[i,j] = (p_i - p_j) / ||p_i - p_j||^2
    # Then grad_i = sum_j G[i,j]
    
    # Avoid division by zero for diagonal
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-15
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grads[i] = sum_j (pts[i] - pts[j]) * inv_d2[i,j]
    # This is equivalent to: 
    # Let A = pts. 
    # grad_i = sum_j A_i * inv_d2[i,j] - sum_j A_j * inv_d2[i,j]
    # = A_i * sum_j inv_d2[i,j] - (A @ inv_d2)[i]
    
    row_sums = np.sum(inv_d2, axis=1)
    grads = pts * row_sums[:, np.newaxis] - (pts @ inv_d2)
    
    # Project gradient to tangent space: grad - (grad . p) * p
    # grad_proj = grads - np.sum(grads * pts, axis=1, keepdims=True) * pts
    proj_factor = np.sum(grads * pts, axis=1, keepdims=True)
    grads = grads - proj_factor * pts
    
    return energy, grads

def _stochastic_gradient_descent(pts, time_budget, rng):
    """
    Run SGD with momentum and simulated annealing on the sphere.
    """
    current_pts = _normalize(pts.copy())
    current_energy = _log_energy(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    momentum = np.zeros_like(current_pts)
    lr = 0.01
    temp = 0.05
    lr_decay = 0.999
    temp_decay = 0.999
    
    start_time = time.time()
    steps = 0
    
    while time.time() < time_budget:
        energy, grad = _energy_and_grad(current_pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = current_pts.copy()
            
        # Update momentum
        momentum = 0.9 * momentum - lr * grad
        
        # Update points
        new_pts = current_pts + momentum
        
        # Normalize to sphere
        new_pts = _normalize(new_pts)
        
        # Compute new energy
        new_energy, _ = _energy_and_grad(new_pts)
        
        # Simulated annealing acceptance
        delta_e = new_energy - energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-9)):
            current_pts = new_pts
        else:
            # Revert
            current_pts = _normalize(current_pts)
            
        # Decay learning rate and temperature
        lr *= lr_decay
        temp *= temp_decay
        
        steps += 1
        
        # If lr is too small, break
        if lr < 1e-6:
            break
            
    return best_pts, best_energy

def construct():
    start_time = time.time()
    total_time = 150.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_spiral(N)
    best_energy = _log_energy(best_pts)
    
    # 1. Initial L-BFGS polish to get close to a local minimum
    def objective(x):
        pts = _normalize(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def grad(x):
        pts = _normalize(x.reshape(N, 3))
        _, g = _energy_and_grad(pts)
        return g.flatten()
        
    try:
        result = minimize(
            objective, best_pts.flatten(), method='L-BFGS-B', jac=grad,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        new_pts = _normalize(result.x.reshape(N, 3))
        new_energy = _log_energy(new_pts)
        if new_energy < best_energy:
            best_pts = new_pts
            best_energy = new_energy
    except Exception:
        pass
    
    if time.time() >= deadline:
        return best_pts
        
    # 2. Multi-restart SGD with Simulated Annealing
    # Use remaining time for multiple short SGD runs from diverse starts
    remaining_time = deadline - time.time()
    n_restarts = 10
    time_per_restart = remaining_time / n_restarts * 0.9 # leave 10% for final polish
    
    for i in range(n_restarts):
        if time.time() >= deadline:
            break
            
        # Create diverse starting points
        if i % 2 == 0:
            # Perturb best known
            pts_init = best_pts + rng.standard_normal((N, 3)) * 0.1
        else:
            # Random or Fibonacci with large perturbation
            pts_init = _fibonacci_spiral(N)
            pts_init = pts_init + rng.standard_normal((N, 3)) * 0.5
            
        pts_init = _normalize(pts_init)
        
        # Run SGD
        pts, energy = _stochastic_gradient_descent(pts_init, time.time() + time_per_restart, rng)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
            
        # Quick L-BFGS polish on the result of SGD
        if time.time() < deadline - 5:
            try:
                result = minimize(
                    objective, pts.flatten(), method='L-BFGS-B', jac=grad,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                new_pts = _normalize(result.x.reshape(N, 3))
                new_energy = _log_energy(new_pts)
                if new_energy < best_energy:
                    best_pts = new_pts
                    best_energy = new_energy
            except Exception:
                pass

    # 3. Final high-precision polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                objective, best_pts.flatten(), method='L-BFGS-B', jac=grad,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_pts = _normalize(result.x.reshape(N, 3))
            new_energy = _log_energy(new_pts)
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
