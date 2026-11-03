"""
Logarithmic energy minimization for N points on the unit sphere.
Uses Riemannian Simulated Annealing combined with frequent local BFGS polishing
to escape local minima and converge to a high-precision global minimum.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
# We aim to finish well before the 120s limit.
# We will set a dynamic deadline inside construct().

def _project(X):
    """Project points onto the unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n

def _energy(pts):
    """
    Fast logarithmic energy calculation.
    Returns sum_{i<j} log(1 / |xi - xj|).
    """
    n = pts.shape[0]
    # Squared distances
    # d_ij^2 = |xi|^2 + |xj|^2 - 2 xi . xj
    # Since points are on sphere, |xi|^2 = 1
    # d_ij^2 = 2 - 2 xi . xj
    dots = pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0) # Numerical stability
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0) # Avoid log(0) on diagonal
    dist_sq = np.maximum(dist_sq, 1e-15) # Avoid log(0) or log(negative) due to precision
    
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    # log(1/d) = -log(d)
    return float(-np.sum(np.log(dist[iu])))

def _energy_grad(pts):
    """
    Compute Riemannian gradient of the logarithmic energy on the sphere.
    Energy E = sum_{i<j} -log(||xi - xj||)
    Gradient wrt xi: sum_{j != i} (xi - xj) / ||xi - xj||^2
    Riemannian gradient: Proj_{T_xi S^2} (grad_Ei) = grad_Ei - (grad_Ei . xi) xi
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    # We need sum_{j != i} (xi - xj) / ||xi - xj||^2
    # Let's compute the vector sum for each i.
    # V_i = sum_{j != i} (xi - xj) / d_ij^2
    # Note: (xi - xj) / d_ij^2 = (xi - xj) / (2 - 2 xi.xj)
    
    # To do this efficiently without loops:
    # We can compute the matrix M where M_ij = (xi - xj) / d_ij^2 is not directly a matrix of scalars.
    # Instead, note that grad_E = sum_{j != i} (xi - xj) / d_ij^2
    # = sum_{j != i} xi / d_ij^2 - sum_{j != i} xj / d_ij^2
    # = xi * (sum_{j != i} 1/d_ij^2) - (sum_{j != i} xj / d_ij^2)
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True) # shape (n, 1)
    # sum_{j != i} xj / d_ij^2 = W.T @ pts (since W is symmetric)
    Wx = W.T @ pts # shape (n, 3)
    
    # Euclidean gradient
    raw_grad = sum_W * pts - Wx
    
    # Project to tangent space
    # grad_R = raw_grad - (raw_grad . xi) xi
    dot_prod = np.sum(raw_grad * pts, axis=1, keepdims=True)
    grad_R = raw_grad - dot_prod * pts
    
    # Energy
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    return energy, grad_R

def _precise_energy(pts):
    """Compute energy using math.fsum for high precision."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _fast_precise_energy(pts):
    """
    Slightly faster precise energy by using NumPy for distances 
    but math.fsum for the sum.
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    ds = dist[iu]
    if np.any(ds < 1e-9):
        return float('inf')
    # Convert to list for math.fsum
    # math.fsum is significantly more accurate than np.sum
    return math.fsum(list(np.log(1.0 / ds)))

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)

def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]

def _riemannian_step(pts, rng, step_size):
    """
    Move points along random geodesics by step_size.
    exp_x(v) = cos(||v||)x + sin(||v||)(v/||v||)
    """
    v = rng.standard_normal((N, 3))
    # Project v to tangent space
    dot_vx = np.sum(v * pts, axis=1, keepdims=True)
    v_tangent = v - dot_vx * pts
    
    v_norms = np.linalg.norm(v_tangent, axis=1, keepdims=True)
    v_norms = np.where(v_norms < 1e-12, 1.0, v_norms)
    v_unit = v_tangent / v_norms
    
    cos_s = np.cos(step_size)
    sin_s = np.sin(step_size)
    
    new_pts = cos_s * pts + sin_s * v_unit
    return _project(new_pts)

def _local_optimize(pts, max_iter=100, tol=1e-10):
    """
    Perform local BFGS optimization on the sphere.
    We optimize in the ambient R^3 space but project the gradient to the tangent space.
    """
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        # Project to sphere to ensure we are on the manifold
        p = _project(p)
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
        
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='BFGS',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': tol,
                'ftol': 1e-12
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts

def construct():
    start_time = time.time()
    # Hard deadline to ensure we return before 120s limit
    deadline = start_time + 110.0 
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse initializations
    # Reduced number of initializations to save time
    initializations = []
    
    # Fibonacci spirals
    for offset in [0.3, 0.7]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # Sobol sequences
    for s in range(5):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # Random points
    for s in range(5):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))

    # 2. Quick local optimization of all initializations
    best_pts = None
    best_energy = float('inf')
    
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        # Quick polish: fewer iterations, looser tolerance for speed
        opt_pts = _local_optimize(init_pts, max_iter=30, tol=1e-6)
        e = _energy(opt_pts) # Use fast energy for comparison during search
        if e < best_energy:
            best_energy = e
            best_pts = opt_pts.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_energy = _energy(best_pts)

    # 3. Simulated Annealing
    T0 = 10.0
    alpha = 5.0 # Cooling rate
    
    current_pts = best_pts.copy()
    current_energy = best_energy
    step_size_base = 0.5
    
    # Annealing Loop
    # We will run this until we have used up the budget reserved for SA
    sa_deadline = deadline - 10.0 # Leave 10s for final polish
    
    last_polish_time = time.time()
    polish_interval = 2.0 # Polish every 2 seconds
    
    while time.time() < sa_deadline:
        # Temperature
        elapsed = time.time() - start_time
        T = T0 * math.exp(-alpha * elapsed / 100.0) # Normalize cooling over ~100s window
        
        # Step size decreases slightly as we cool
        step_size = step_size_base * (0.5 + 0.5 * T / T0)
        
        # Generate perturbation
        new_pts = _riemannian_step(current_pts, rng, step_size)
        new_energy = _energy(new_pts)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 1e-6 and math.log(rng.random()) < -delta_e / T):
            current_pts = new_pts
            current_energy = new_energy
            
            # If we found a new global best, update it
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
        
        # Periodic local polishing
        if time.time() > last_polish_time + polish_interval:
            last_polish_time = time.time()
            polished = _local_optimize(current_pts, max_iter=20, tol=1e-6)
            pol_energy = _energy(polished)
            
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_pts = polished.copy()
            
            # Update current to the polished version
            current_pts = polished
            current_energy = pol_energy

    # 4. Final High-Precision Polish
    # Use remaining time for tight optimization
    final_pts = best_pts.copy()
    
    # Run a few final optimizations with tight tolerance
    while time.time() < deadline - 2.0:
        final_pts = _local_optimize(final_pts, max_iter=100, tol=1e-10)
        # Check if we converged or if it's taking too long
        # If energy doesn't change much, break
        if time.time() > deadline - 5.0:
            break

    # Final validity check and precise energy calculation for safety
    if not _valid(final_pts):
        final_pts = best_pts
        if not _valid(final_pts):
            final_pts = _fibonacci_spiral(N, offset=0.5)
            final_pts = _local_optimize(final_pts, max_iter=50, tol=1e-6)

    return final_pts

def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True
