import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _log_energy(X):
    """Compute the pure logarithmic energy sum_{i<j} log(1/d_ij) with math.fsum for accuracy."""
    dists = cdist(X, X)
    np.fill_diagonal(dists, 1.0)
    d_safe = np.maximum(dists, 1e-12)
    
    # Extract upper triangle to sum each pair once
    iu = np.triu_indices(N, k=1)
    vals = -np.log(d_safe[iu])
    
    # Use math.fsum for highest accuracy as per contract
    return math.fsum(vals.tolist())

def _energy_and_grad_spherical(X_flat):
    """
    Objective and gradient for L-BFGS-B.
    We use a penalty method in R^3 to handle the sphere constraint.
    This is robust and fast.
    """
    X = X_flat.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1)
    # Avoid division by zero
    norms_safe = np.maximum(norms, 1e-12)
    X_norm = X / norms_safe[:, np.newaxis]
    
    # Compute distances
    dists = cdist(X_norm, X_norm)
    np.fill_diagonal(dists, 1.0)
    d_safe = np.maximum(dists, 1e-12)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d)
    # We compute the sum over all pairs i != j and divide by 2
    log_dists = np.log(d_safe)
    E_log = -0.5 * np.sum(log_dists)
    
    # Penalty for sphere constraint
    # P * sum(||x_i||^2 - 1)^2
    P = 1e4
    norms_sq = np.sum(X**2, axis=1)
    penalty_term = P * np.sum((norms_sq - 1.0)**2)
    
    E_total = E_log + penalty_term
    
    # Gradient
    # dE_log/dx_i = -0.5 * sum_{j!=i} (x_i - x_j) / d_ij^2
    diff = X_norm[:, np.newaxis, :] - X_norm[np.newaxis, :, :]
    inv_dists_sq = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_dists_sq[mask] = 1.0 / (dists[mask] ** 2)
    
    grad_log = -0.5 * np.sum(diff * inv_dists_sq[:, np.newaxis], axis=1)
    
    # dPenalty/dx_i = 2 * P * (||x_i||^2 - 1) * x_i
    grad_pen = 2 * P * (norms_sq - 1.0)[:, np.newaxis] * X
    
    grad_total = grad_log + grad_pen
    
    return E_total, grad_total.flatten()

def _fibonacci_sphere(n, seed=0):
    """Generate a Fibonacci lattice on the sphere."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Add slight noise to break symmetry
    pts += rng.randn(n, 3) * 0.01
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _random_sphere(n, seed=0):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _optimize_local(X_init, deadline):
    """
    Run L-BFGS-B from a given start.
    Returns the optimized points and energy.
    """
    X_flat = X_init.flatten()
    
    def objective(x):
        val, _ = _energy_and_grad_spherical(x)
        return val
    
    def jac(x):
        _, g = _energy_and_grad_spherical(x)
        return g
    
    try:
        res = minimize(objective, X_flat, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 100, 'gtol': 1e-6, 'maxls': 20})
        
        X_opt = res.x.reshape(N, 3)
        # Renormalize to ensure exact unit sphere
        X_opt = X_opt / np.linalg.norm(X_opt, axis=1, keepdims=True)
        
        E = _log_energy(X_opt)
        return X_opt, E, res.success
    except Exception as e:
        return X_init, float('inf'), False

def construct():
    start_time = time.time()
    # Safety margin: 145s deadline (5s buffer)
    deadline = start_time + 145.0
    
    best_X = None
    best_energy = float('inf')
    
    # Phase 1: Multi-start optimization
    # We try a variety of initial configurations to escape local minima.
    # 1. Fibonacci lattice (good global structure)
    # 2. Random starts (diverse)
    # 3. Spherical code approximations (if we had them, but random is fine)
    
    inits = []
    
    # Fibonacci with different noise seeds
    for s in range(5):
        inits.append(_fibonacci_sphere(N, seed=s))
        
    # Random starts
    for s in range(20):
        inits.append(_random_sphere(N, seed=100 + s))
        
    # Shuffle inits to ensure diversity in time budget
    rng = np.random.RandomState(42)
    rng.shuffle(inits)
    
    for i, X_init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        X_opt, E, success = _optimize_local(X_init, deadline)
        
        if E < best_energy:
            best_energy = E
            best_X = X_opt.copy()
            
    if best_X is None:
        # Fallback
        best_X = _fibonacci_sphere(N, seed=0)
        best_energy = _log_energy(best_X)
        
    # Phase 2: Simulated Annealing / Perturbation to escape local minima
    # We perturb the best solution and re-optimize.
    # This is a form of "basin hopping".
    
    X_current = best_X.copy()
    E_current = best_energy
    
    # Adaptive temperature
    # Start with a moderate perturbation size
    pert_size = 0.1
    rng = np.random.RandomState(123)
    
    # Number of basin hops
    # We will run until time is up
    hop_count = 0
    
    while time.time() < deadline - 5.0:
        hop_count += 1
        
        # Perturb current best
        X_pert = X_current.copy()
        noise = rng.randn(N, 3) * pert_size
        X_pert += noise
        X_pert = X_pert / np.linalg.norm(X_pert, axis=1, keepdims=True)
        
        # Optimize from perturbed start
        X_opt, E_opt, success = _optimize_local(X_pert, deadline)
        
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt.copy()
            X_current = X_opt.copy()
            # Reduce perturbation size slightly as we find better minima
            pert_size *= 0.95
        else:
            # If no improvement, increase perturbation size to explore more
            # But cap it to avoid completely random walks
            pert_size = min(pert_size * 1.05, 0.5)
            
    # Final normalization
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
