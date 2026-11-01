"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_start_time = time.time()
DEADLINE = 150.0  # seconds

def _energy_fsum(pts):
    """Compute logarithmic energy using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """Compute energy and gradient for logarithmic energy."""
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return float('inf'), grad
            # d/dx_i log(1/d) = -d/dx_i log(d) = -(x_i - x_j)/d^2
            energy += math.log(1.0 / d)
            grad[i] -= diff / (d * d)
    
    # Each pair counted twice in gradient, but energy should count each pair once
    # Actually, let me recompute: energy = sum_{i<j} log(1/d_ij)
    # Gradient wrt x_i: sum_{j!=i} (x_i - x_j)/d_ij^2
    # Let me redo this more carefully
    
    energy = 0.0
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i+1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return float('inf'), grad
            energy += math.log(1.0 / d)
            # d/dx_i log(1/d) = -1/d * d/dx_i d = -1/d * (x_i-x_j)/d = -(x_i-x_j)/d^2
            # d/dx_j log(1/d) = -1/d * d/dx_j d = -1/d * (x_j-x_i)/d = (x_i-x_j)/d^2
            grad[i] -= diff / (d * d)
            grad[j] += diff / (d * d)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _riemannian_grad(pts, grad):
    """Project gradient to tangent space of sphere."""
    # Tangent space at x: {v : v·x = 0}
    # Projection: v - (v·x)x
    proj = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - proj * pts

def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(n, seed=None):
    """Random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    v = rng.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _icosahedral_like_init(n, seed=None):
    """Icosahedral-like initialization with perturbations."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Start with Fibonacci and add noise
    base = _fibonacci_init(n)
    noise = rng.randn(n, 3) * 0.1
    pts = base + noise
    return _project_to_sphere(pts)

def _optimize_from_start(pts, max_time=10.0):
    """Optimize from a given starting point using Riemannian methods."""
    start = time.time()
    pts = _project_to_sphere(pts.copy())
    
    # Phase 1: Riemannian gradient descent with backtracking line search
    lr = 0.01
    min_lr = 1e-6
    max_iter = 5000
    
    best_energy = _energy_fsum(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() - start > max_time:
            break
        
        energy, grad = _energy_and_grad(pts)
        if not np.isfinite(energy):
            break
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Project gradient to tangent space
        rgrad = _riemannian_grad(pts, grad)
        
        # Check convergence
        rgrad_norm = np.linalg.norm(rgrad)
        if rgrad_norm < 1e-10:
            break
        
        # Backtracking line search
        step = lr
        accepted = False
        for _ in range(20):
            # Move in tangent direction and reproject
            new_pts = pts + step * rgrad
            new_pts = _project_to_sphere(new_pts)
            new_energy = _energy_fsum(new_pts)
            
            if np.isfinite(new_energy) and new_energy < energy:
                pts = new_pts
                accepted = True
                break
            step *= 0.5
        
        if not accepted:
            lr *= 0.5
            if lr < min_lr:
                break
        else:
            # Slowly increase learning rate
            lr = min(lr * 1.05, 0.1)
    
    # Phase 2: L-BFGS-B polishing with penalty method
    if time.time() - start < max_time - 1.0:
        pts = _project_to_sphere(best_pts)
        
        def sphere_fun(flat_pts):
            """Penalty-based objective for L-BFGS-B."""
            pts_3d = flat_pts.reshape(-1, 3)
            # Penalty for not being on sphere
            norms = np.linalg.norm(pts_3d, axis=1)
            penalty = 1000.0 * np.sum((norms - 1.0)**2)
            energy = _energy_fsum(pts_3d)
            if not np.isfinite(energy):
                return 1e10
            return energy + penalty
        
        x0 = pts.flatten()
        try:
            res = minimize(sphere_fun, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12})
            if res.success and res.fun < best_energy:
                new_pts = _project_to_sphere(res.x.reshape(-1, 3))
                new_energy = _energy_fsum(new_pts)
                if np.isfinite(new_energy) and new_energy < best_energy:
                    best_pts = new_pts
                    best_energy = new_energy
        except Exception:
            pass
    
    return _project_to_sphere(best_pts), best_energy

def construct():
    """Main entry point: construct 100 points minimizing logarithmic energy."""
    global _start_time
    start = time.time()
    
    # Multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_init(N))
    
    # 2. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024]:
        inits.append(_random_init(N, seed=seed))
    
    # 3. Icosahedral-like with perturbations
    for seed in [7, 17, 27]:
        inits.append(_icosahedral_like_init(N, seed=seed))
    
    # 4. Rotated Fibonacci
    for angle in [0.1, 0.5, 1.0, 2.0]:
        base = _fibonacci_init(N)
        # Random rotation
        rng = np.random.RandomState(int(angle * 100))
        v = rng.randn(3, 3)
        u, _, vt = np.linalg.svd(v)
        rot = u @ vt
        inits.append(_project_to_sphere(base @ rot.T))
    
    # Optimize each initialization, keeping track of time
    best_pts = inits[0].copy()
    best_energy = _energy_fsum(best_pts)
    
    # Allocate time budget: ~10s per optimization, but check deadline
    n_inits = len(inits)
    time_per_init = min(10.0, (DEADLINE - 5.0 - (time.time() - start)) / max(1, n_inits))
    
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start
        remaining = DEADLINE - 5.0 - elapsed
        if remaining < 2.0:
            break
        
        alloc = min(time_per_init, remaining * 0.8)
        if alloc < 1.0:
            break
        
        pts, energy = _optimize_from_start(init_pts, max_time=alloc)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish if time remains
    elapsed = time.time() - start
    remaining = DEADLINE - 3.0 - elapsed
    if remaining > 2.0:
        pts, energy = _optimize_from_start(best_pts, max_time=min(remaining * 0.9, 15.0))
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Final check
    final_energy = _energy_fsum(best_pts)
    if not np.isfinite(final_energy):
        # Fallback to Fibonacci
        best_pts = _fibonacci_init(N)
    
    return best_pts
# EVOLVE-BLOCK-END
