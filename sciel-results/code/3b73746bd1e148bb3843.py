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

def _repulsive_init(n, iterations=100, temp=0.1):
    """
    Initialize points using a repulsive force simulation to spread them out.
    This often yields a better starting configuration than a static spiral.
    """
    pts = _fibonacci_spiral(n)
    pts = pts + np.random.randn(n, 3) * 0.01
    pts = _normalize(pts)
    
    for _ in range(iterations):
        forces = np.zeros_like(pts)
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = np.random.randn(3) * 1e-6
                    dist = np.linalg.norm(diff)
                # Force proportional to 1/dist^2 (softened)
                f_mag = 1.0 / (dist**2 + 0.01)
                forces[i] += f_mag * diff / dist
                forces[j] -= f_mag * diff / dist
        
        # Update positions
        pts = pts + temp * forces
        pts = _normalize(pts)
        temp *= 0.99
    return pts

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

def _spectral_newton_step(pts, grad, max_iter=50, tol=1e-8):
    """
    Performs a few steps of Newton's method using a finite-difference Hessian 
    approximation or a simplified quasi-Newton update. 
    Given the cost of a full Hessian (N^2 x N^2 is too big, but block structure exists),
    we will use a trust-region style approach with a diagonal+off-diagonal approximation 
    or simply a very aggressive L-BFGS with high max_corr.
    
    Actually, for N=150, a full Hessian is 450x450, which is manageable.
    Let's compute the Hessian via finite differences of the gradient.
    H_ij = (g(x+e_j) - g(x-e_j)) / 2
    This requires 2 * 450 gradient evaluations. Each grad eval is O(N^2).
    Total: 900 * 150^2 ~ 20M ops. Very fast.
    """
    n = len(pts)
    dim = n * 3
    x = pts.flatten()
    
    # Compute Hessian via central differences
    H = np.zeros((dim, dim))
    eps = 1e-5
    
    # To save time, we might not need the full Hessian every step.
    # But for the first few steps, it's worth it.
    # Let's do a limited number of Newton steps.
    
    # We'll implement a simple Newton iteration with line search.
    for step in range(max_iter):
        # Compute Hessian
        # Optimization: We can exploit that H is symmetric.
        # But computing 450x450 via FD is heavy if done naively in Python loops.
        # Let's vectorize the FD computation if possible, or just do it.
        
        # Actually, 450 FD calls * O(N^2) might be too slow in pure Python if not careful.
        # Let's try a different approach: Use L-BFGS-B but with a much higher maxiter 
        # and a very tight tolerance, effectively acting as a high-precision polish.
        # The "Spectral" idea was good, but implementation complexity/time is a risk.
        # Let's stick to a robust L-BFGS-B but with a better initialization and 
        # a "polish" phase that runs L-BFGS-B again with tighter tolerances.
        pass
        
    # Fallback to a very tight L-BFGS-B which is robust and fast.
    # We'll define the objective and gradient for L-BFGS-B.
    
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
            objective, x, jac=grad_func, method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
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
    # 1. Generate a few high-quality initializations using repulsive force + random jitter.
    # 2. For each, run L-BFGS-B with standard tolerances.
    # 3. Take the best result and run a "Polish" phase with extremely tight tolerances.
    
    best_pts = None
    best_energy = 1e30
    
    # Phase 1: Multi-restart with diverse initializations
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            init_pts = _fibonacci_spiral(N)
        else:
            # Repulsive initialization with random seed
            init_pts = _repulsive_init(N, iterations=50, temp=0.05)
            # Add small random noise to break symmetry
            init_pts = init_pts + rng.standard_normal((N, 3)) * 0.01
            init_pts = _normalize(init_pts)
            
        # Optimize
        def objective(x):
            p = _normalize(x.reshape(N, 3))
            e, _ = _energy_and_grad(p)
            return e
            
        def grad_func(x):
            p = _normalize(x.reshape(N, 3))
            _, g = _energy_and_grad(p)
            return g.flatten()
            
        try:
            res = minimize(
                objective, init_pts.flatten(), jac=grad_func, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            cand_pts = _normalize(res.x.reshape(N, 3))
            cand_energy, _ = _energy_and_grad(cand_pts)
            
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pts = cand_pts
        except Exception:
            continue
            
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_spiral(N)
        
    # Phase 2: Polish the best solution with tighter tolerances
    if time.time() < deadline - 10:
        def objective_polish(x):
            p = _normalize(x.reshape(N, 3))
            e, _ = _energy_and_grad(p)
            return e
            
        def grad_polish(x):
            p = _normalize(x.reshape(N, 3))
            _, g = _energy_and_grad(p)
            return g.flatten()
            
        try:
            res = minimize(
                objective_polish, best_pts.flatten(), jac=grad_polish, method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            cand_pts = _normalize(res.x.reshape(N, 3))
            cand_energy, _ = _energy_and_grad(cand_pts)
            
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pts = cand_pts
        except Exception:
            pass
            
    return best_pts
# EVOLVE-BLOCK-END
