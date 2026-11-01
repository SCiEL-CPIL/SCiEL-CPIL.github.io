import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE_MARGIN = 5.0

def _riemannian_cg(pts, time_limit=15.0, max_iter=100):
    """Riemannian conjugate gradient on the sphere with adaptive step size."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # Initial direction: negative gradient (steepest descent)
    d = -grad.copy()
    # Project to tangent space
    dot_d = np.sum(d * current, axis=1)
    d -= dot_d[:, np.newaxis] * current
    
    # Momentum coefficient
    beta = 0.0
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Compute line search step
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        
        # If directional derivative is positive, flip direction
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        armijo_satisfied = False
        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            # Armijo condition
            if new_energy <= energy + c1 * alpha * directional_deriv:
                armijo_satisfied = True
                break
            
            alpha *= 0.8
            if alpha < 1e-14:
                break
        
        if best_alpha == 0.0:
            # Try smaller step
            alpha_try = 0.01
            for _ in range(10):
                new_pts = _exponential_map(current, alpha_try * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha_try
                    break
                alpha_try *= 0.5
            
            if best_alpha == 0.0:
                break
        
        # Update gradient and compute new direction
        old_grad = grad
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-12:
            break
        
        # Fletcher-Reeves beta
        if it > 0:
            grad_norm_sq = np.sum(grad * grad)
            old_grad_norm_sq = np.sum(old_grad * old_grad)
            if old_grad_norm_sq > 1e-30:
                beta = grad_norm_sq / old_grad_norm_sq
                # Cap beta to prevent divergence
                beta = min(beta, 1.0)
        
        # New conjugate direction
        d_new = -grad + beta * d
        # Project to tangent space
        dot_d_new = np.sum(d_new * current, axis=1)
        d_new -= dot_d_new[:, np.newaxis] * current
        
        # Check if direction is meaningful
        dir_norm = np.linalg.norm(d_new)
        if dir_norm < 1e-14:
            d = -grad.copy()
            dot_d = np.sum(d * current, axis=1)
            d -= dot_d[:, np.newaxis] * current
        else:
            d = d_new
    
    return current, energy

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
    
    # Project gradient to tangent space
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
    d_sq = np.maximum(dists_sq[iu], 1e-30)
    return -0.5 * np.sum(np.log(d_sq))

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

def _lbfgs_sphere(pts, time_limit=10.0, max_iter=200, m=10):
    """L-BFGS on the sphere using exponential maps and parallel transport."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # L-BFGS history
    s_history = []
    y_history = []
    g_history = []
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        q = grad.copy()
        rho = []
        
        # Two-loop recursion (backward)
        for k in range(len(s_history) - 1, -1, -1):
            s_k = s_history[k]
            y_k = y_history[k]
            g_k = g_history[k]
            
            alpha_k = np.sum(s_k * q) / (np.sum(y_k * y_k) + 1e-30)
            rho.append(alpha_k)
            q -= alpha_k * y_k
        
        r = q
        
        # Two-loop recursion (forward)
        d = r.copy()
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            g_k = g_history[k]
            
            beta_k = np.sum(y_k * r) / (np.sum(y_k * y_k) + 1e-30)
            d += (rho[k] - beta_k) * s_k
        
        # Ensure d is in tangent space
        dot_d = np.sum(d * current, axis=1)
        d -= dot_d[:, np.newaxis] * current
        
        # Check if direction is meaningful
        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break
        
        # Line search (Armijo backtracking with continued search)
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        
        # If directional derivative is positive, flip direction
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        armijo_satisfied = False
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            # Armijo condition
            if new_energy <= energy + c1 * alpha * directional_deriv:
                armijo_satisfied = True
                break
            
            alpha *= 0.7
            if alpha < 1e-14:
                break
        
        if best_alpha == 0.0:
            # Try smaller step
            alpha_try = 0.01
            for _ in range(8):
                new_pts = _exponential_map(current, alpha_try * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha_try
                    break
                alpha_try *= 0.5
            
            if best_alpha == 0.0:
                break
        
        # Update L-BFGS history
        s_vec = best_pts - current
        y_vec = best_grad - grad
        
        # Project to tangent space
        dot_s = np.sum(s_vec * current, axis=1)
        s_vec -= dot_s[:, np.newaxis] * current
        
        dot_y = np.sum(y_vec * best_pts, axis=1)
        y_vec -= dot_y[:, np.newaxis] * best_pts
        
        # Check if y and s are sufficiently different
        y_norm = np.linalg.norm(y_vec)
        s_norm = np.linalg.norm(s_vec)
        if y_norm > 1e-12 and s_norm > 1e-12:
            s_history.append(s_vec)
            y_history.append(y_vec)
            g_history.append(grad)
            
            if len(s_history) > m:
                s_history.pop(0)
                y_history.pop(0)
                g_history.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return current, energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=12):
    """Perform local optimization followed by basin hopping using Riemannian CG."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization using Riemannian CG
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _riemannian_cg(current, time_limit=min(12, time_left - 2))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop with diverse perturbation strategies
    temp = 1.0
    decay = 0.85
    
    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 8:
            break
        
        rng = np.random.default_rng(rh + 123)
        
        # Choose perturbation strategy based on iteration
        if rh % 3 == 0:
            # Gaussian perturbation
            scale = 0.3 * (0.8 ** rh)
            perturbations = rng.standard_normal((N, 3)) * scale
        elif rh % 3 == 1:
            # Random rotation perturbation
            theta = rng.uniform(0.1, 0.5, size=N)
            angles = rng.uniform(0, 2 * np.pi, size=N)
            # Create small rotations
            perturbations = np.zeros((N, 3))
            for i in range(N):
                # Small perturbation in random direction
                pert_dir = rng.standard_normal(3)
                pert_dir -= np.sum(pert_dir * current[i]) * current[i]
                pert_dir /= np.linalg.norm(pert_dir)
                perturbations[i] = theta[i] * pert_dir
        else:
            # Swap-based perturbation: move a few points
            n_swaps = rng.integers(3, 8)
            perturbations = np.zeros((N, 3))
            for _ in range(n_swaps):
                i = rng.integers(0, N)
                # Move point i to a random nearby location
                new_pos = current[i] + rng.standard_normal(3) * 0.2
                new_pos /= np.linalg.norm(new_pos)
                perturbations[i] = new_pos - current[i]
        
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point using Riemannian CG
        opt_time = min(8, time_left - 4)
        if opt_time < 3:
            break
        new_opt_pts, new_energy = _riemannian_cg(new_pts, time_limit=opt_time)
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        accept = delta_e < 0
        if not accept and temp > 1e-6:
            try:
                accept = np.random.rand() < math.exp(-delta_e / temp)
            except OverflowError:
                accept = False
        
        if accept:
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
        
        temp *= decay
    
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

def _farthest_point_sampling(n, seed=0, n_candidates=200):
    """Generate points using farthest point sampling for good initial spread."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3))
    
    # Start with a random point
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    for i in range(1, n):
        # Sample random candidates
        candidates = rng.standard_normal((n_candidates, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Compute min distance from each candidate to all placed points
        # Vectorized: for each candidate, compute distance to all placed points
        placed = pts[:i]  # shape (i, 3)
        # dists shape: (n_candidates, i)
        dists = np.sqrt(np.sum((candidates[:, np.newaxis, :] - placed[np.newaxis, :, :])**2, axis=2))
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        pts[i] = candidates[best_idx]
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initializations - diverse starting points
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Farthest point sampling for better initial spread
    inits.append(_farthest_point_sampling(N, seed=0))
    inits.append(_farthest_point_sampling(N, seed=1))
    
    # Random inits
    for s in range(3):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Fibonacci
    for s in range(3):
        rng = np.random.default_rng(s + 100)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.15
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Halton
    rng = np.random.default_rng(200)
    pts = _halton_sphere(N) + rng.standard_normal((N, 3)) * 0.12
    pts = _normalize(pts)
    inits.append(pts)
    
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        time_per_restart = 16.0
        if idx == 0:
            time_per_restart = 20.0
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish with Riemannian CG
    if time.time() < deadline - 3:
        polish_time = deadline - time.time() - 1
        try:
            final_pts, final_energy = _riemannian_cg(best_pts, time_limit=polish_time, max_iter=200)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    return _normalize(best_pts)
