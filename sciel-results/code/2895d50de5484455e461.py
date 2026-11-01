import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space: g - (g . p) p
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _riemannian_lbfgs(pts, time_limit=10.0, max_iter=200, m=10):
    """Riemannian L-BFGS on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # L-BFGS memory
    S = []  # step vectors (tangent)
    Y = []  # gradient differences
    
    # Initial search direction
    d = -grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # L-BFGS two-loop recursion to compute new direction
        if len(S) > 0:
            r = -grad.copy()
            alphas = []
            # Backward loop
            for i in range(len(S) - 1, -1, -1):
                si = S[i]
                yi = Y[i]
                yi_norm_sq = np.sum(yi * yi)
                if yi_norm_sq > 1e-30:
                    beta_i = np.sum(si * r) / yi_norm_sq
                else:
                    beta_i = 0.0
                r = r - beta_i * yi
                alphas.append(beta_i)
            
            # Initial Hessian approximation
            s_last = S[-1]
            y_last = Y[-1]
            gamma = np.sum(s_last * y_last) / np.sum(y_last * y_last)
            if gamma < 1e-30:
                gamma = 1.0
            
            r = gamma * r
            
            # Forward loop
            for i in range(len(S)):
                si = S[i]
                yi = Y[i]
                alpha_i = alphas[i]
                if np.sum(yi * yi) > 1e-30:
                    beta_i = np.sum(yi * r) / np.sum(yi * yi)
                else:
                    beta_i = 0.0
                r = r + (alpha_i - beta_i) * si
            
            d = r
        
        # Backtracking line search
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        gd = np.sum(grad * d)
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Sufficient decrease (Armijo)
            if new_energy > energy + c1 * alpha * gd:
                break
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            if new_energy > energy:
                break
                
            alpha *= 0.7
            
            if alpha < 1e-12:
                break
                
        if best_alpha == 0.0:
            break
            
        # Update L-BFGS memory
        step = best_alpha * d
        grad_diff = best_grad - grad
        step_norm = np.linalg.norm(step)
        grad_diff_norm = np.linalg.norm(grad_diff)
        
        if step_norm > 1e-10 and grad_diff_norm > 1e-10:
            S.append(step)
            Y.append(grad_diff)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-13:
            break
        
    return current, energy

def _simulated_annealing(pts, time_limit=20.0, max_iter=5000):
    """Simulated Annealing on the sphere."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    best_pts = current.copy()
    best_energy = energy
    
    T = 1.0
    cooling = 0.999
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
            
        # Generate random tangent perturbation
        rng = np.random.default_rng(it)
        perturbations = rng.standard_normal((N, 3))
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        # Scale perturbation
        step_size = 0.1 * T + 0.01
        perturbations *= step_size
        
        new_pts = _exponential_map(current, perturbations)
        new_energy = _log_energy_fast(new_pts)
        
        delta_e = new_energy - energy
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(T, 1e-10)):
            current = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
        
        T *= cooling
        if T < 1e-5:
            break
            
    return best_pts, best_energy

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _spherical_design_init(n, seed=42):
    """Generate points using a more structured approach with random rotations."""
    rng = np.random.default_rng(seed)
    pts = _fibonacci_sphere(n)
    for _ in range(3):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        pts = pts @ R.T
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initializations - diverse set
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_spherical_design_init(N, seed=42))
    inits.append(_spherical_design_init(N, seed=123))
    inits.append(_spherical_design_init(N, seed=999))
    
    # Random inits
    for s in range(3):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    best_pts = None
    best_energy = float('inf')
    
    # Time budget allocation
    # 1. Global Search (SA) + Local Polish (L-BFGS) for each init
    # 2. Final Polish on the best result
    
    # We will iterate through inits, spending some time on SA and some on L-BFGS
    # Total time available for search: ~130s
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
            
        # Allocate time for this init
        # Give more time to structured inits
        if idx < 5:
            sa_time = 10.0
            lbfgs_time = 10.0
        else:
            sa_time = 5.0
            lbfgs_time = 5.0
            
        try:
            # Phase 1: Simulated Annealing to escape local minima
            sa_pts, sa_energy = _simulated_annealing(init_pts, time_limit=sa_time)
            
            # Phase 2: Riemannian L-BFGS to polish
            opt_pts, opt_energy = _riemannian_lbfgs(sa_pts, time_limit=lbfgs_time)
            opt_energy = _log_energy_fast(opt_pts)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    # If no valid points found, fallback to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final high-precision polish with L-BFGS
    if time.time() < deadline - 5.0:
        polish_time = deadline - time.time() - 2.0
        try:
            final_pts, final_energy = _riemannian_lbfgs(best_pts, time_limit=polish_time, max_iter=500, m=15)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    return _normalize(best_pts)
