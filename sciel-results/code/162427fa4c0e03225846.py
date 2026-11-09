"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import scipy.optimize as so

_DEADLINE = 145.0  # seconds; safety margin before 150s kill

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return 1e18
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_fast(pts):
    """Vectorized energy computation."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    if np.any(d < 1e-9):
        return 1e18
    return float(np.sum(np.log(1.0 / d)))

def _tangent_grad(pts):
    """Compute energy and Riemannian gradient (tangent space) on the sphere.
    
    For E = sum_{i<j} log(1/|x_i - x_j|),
    dE/dx_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    
    The Riemannian gradient is the projection onto the tangent space:
    grad_T = grad - (grad · x_i) x_i
    """
    n = len(pts)
    # Vectorized pairwise computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    
    # dist_sq_inv = 1/d^2 for all pairs
    dist_sq_inv = 1.0 / dist_sq
    
    # grad[i] = -sum_j (x_i - x_j) / d_ij^2 = -sum_j diff[i,j] / dist_sq[i,j]
    # = -sum_j diff[i,j] * dist_sq_inv[i,j]
    grad = -np.sum(diff * dist_sq_inv[:, :, np.newaxis], axis=1)  # (n,3)
    
    # Project to tangent space: grad_T = grad - (grad · x) x
    grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)  # (n,1)
    grad_tan = grad - grad_dot_x * pts  # (n,3)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    if np.any(d < 1e-9):
        return 1e18, np.zeros_like(pts)
    E = float(np.sum(np.log(1.0 / d)))
    
    return E, grad_tan

def _retract(pts, v, alpha):
    """Retraction on the sphere: x -> (x + alpha*v) / |x + alpha*v|"""
    new_pts = pts + alpha * v
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return new_pts / norms

def _retract_grad(pts, v, alpha):
    """Pullback of tangent vector after retraction.
    
    If y = retract(x, v) = (x + alpha*v)/|x + alpha*v|,
    then the pullback of a tangent vector w at y back to tangent space at x is:
    P_x (w - (w · y) y + ...) -- simplified for small alpha
    For our purposes, we approximate by projecting directly.
    """
    # Simple projection approximation
    return v - np.sum(v * pts, axis=1, keepdims=True) * pts

def _riemannian_lbfgs(pts, max_iter=5000, tol=1e-10):
    """Riemannian L-BFGS optimization using retraction.
    
    Uses L-BFGS-B in ambient space with Riemannian gradient,
    then re-projects onto the sphere after each step via callback.
    """
    from scipy.optimize import minimize
    
    pts = pts.copy()
    
    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        # Project to sphere for evaluation
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        x_sphere = x / norms
        E_val, _ = _tangent_grad(x_sphere)
        return E_val
    
    def gradient(x_flat):
        x = x_flat.reshape(N, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        x_sphere = x / norms
        _, g = _tangent_grad(x_sphere)
        return g.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol,
                               'maxls': 20})
    
    opt_pts = result.x.reshape(N, 3)
    norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    opt_pts = opt_pts / norms
    
    opt_E = _energy_fast(opt_pts)
    return opt_pts, opt_E

def _conjugate_gradient_riemannian(pts, max_iter=3000, tol=1e-10):
    """Polak-Ribiere conjugate gradient on the sphere with proper line search."""
    pts = pts.copy()
    E, g_tan = _tangent_grad(pts)
    s = -g_tan
    g_old = g_tan
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        # Line search: find alpha that decreases energy
        alpha = 0.1
        best_alpha = None
        best_E = E
        for _ in range(30):
            new_pts = _retract(pts, s, alpha)
            new_E = _energy_fast(new_pts)
            if new_E < best_E:
                best_E = new_E
                best_alpha = alpha
            # Armijo-like condition
            if new_E < E + 1e-4 * alpha * np.sum(g_tan * s):
                break
            alpha *= 0.5
        
        if best_alpha is None:
            break
        
        pts = _retract(pts, s, best_alpha)
        E = best_E
        
        E_new, g_new = _tangent_grad(pts)
        
        if np.linalg.norm(g_new) < tol:
            break
        
        # Polak-Ribiere beta
        dot_new_old = np.sum(g_new * g_new - g_new * g_old)
        dot_old_old = np.sum(g_old * g_old)
        if dot_old_old > 1e-15:
            beta = max(0.0, dot_new_old / dot_old_old)
        else:
            beta = 0.0
        
        s = -g_new + beta * s
        g_old = g_new
    
    return pts, E

def _particle_swarm(pts_init, max_iter=200, n_particles=10, tol=1e-10):
    """Particle swarm optimization on the sphere.
    
    Each particle is a full configuration of N points.
    Velocity is in tangent space.
    """
    rng = np.random.default_rng(12345)
    
    # Initialize particles around the initial configuration
    particles = []
    velocities = []
    for p in range(n_particles):
        if p == 0:
            particles.append(pts_init.copy())
        else:
            # Small random perturbation in tangent space
            noise = rng.standard_normal((N, 3)) * 0.05
            noise = noise - np.sum(noise * pts_init, axis=1, keepdims=True) * pts_init
            particles.append(_retract(pts_init, noise, 1.0))
        velocities.append(np.zeros_like(pts_init))
    
    E_all = [_energy_fast(p) for p in particles]
    best_local = [particles[i].copy() for i in range(n_particles)]
    best_local_E = E_all[:]
    
    global_best = particles[int(np.argmin(E_all))].copy()
    global_best_E = min(E_all)
    
    w = 0.9  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        
        for p in range(n_particles):
            # Update velocity in tangent space
            r1 = rng.random()
            r2 = rng.random()
            
            # Cognitive component: direction to personal best
            cog = best_local[p] - particles[p]
            cog = cog - np.sum(cog * particles[p], axis=1, keepdims=True) * particles[p]
            
            # Social component: direction to global best
            soc = global_best - particles[p]
            soc = soc - np.sum(soc * particles[p], axis=1, keepdims=True) * particles[p]
            
            # Velocity update
            velocities[p] = w * velocities[p] + c1 * r1 * cog + c2 * r2 * soc
            
            # Limit velocity
            v_norm = np.linalg.norm(velocities[p])
            v_max = 0.1
            if v_norm > v_max:
                velocities[p] *= v_max / v_norm
            
            # Update position via retraction
            particles[p] = _retract(particles[p], velocities[p], 1.0)
            
            # Evaluate
            E_p = _energy_fast(particles[p])
            
            # Update personal best
            if E_p < best_local_E[p]:
                best_local[p] = particles[p].copy()
                best_local_E[p] = E_p
            
            # Update global best
            if E_p < global_best_E:
                global_best = particles[p].copy()
                global_best_E = E_p
    
    return global_best, global_best_E

def _fibonacci_start():
    """Golden-angle Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _project(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _random_start(seed):
    """Generate a random set of points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project(pts)

def _jittered_fibonacci(jitter, seed):
    """Fibonacci start with small random jitter."""
    pts = _fibonacci_start()
    rng = np.random.default_rng(seed)
    pts = pts + rng.standard_normal((N, 3)) * jitter
    return _project(pts)

def _project(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _stochastic_local_search(pts, n_restarts=20, max_iter=500):
    """Stochastic local search: random perturbations + gradient descent.
    
    This helps escape local minima by occasionally making large random jumps.
    """
    rng = np.random.default_rng(999)
    best_pts = pts.copy()
    best_E = _energy_fast(pts)
    
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Start from best found, with a random perturbation
        if r == 0:
            current = pts.copy()
        else:
            # Random perturbation: mix of small and large jumps
            jump_size = 0.01 * (1 + 0.5 * r)
            noise = rng.standard_normal((N, 3)) * jump_size
            noise = noise - np.sum(noise * best_pts, axis=1, keepdims=True) * best_pts
            current = _retract(best_pts, noise, 1.0)
        
        # Gradient descent from this start
        opt_pts, E = _conjugate_gradient_riemannian(current, max_iter=max_iter, tol=1e-10)
        
        if E < best_E:
            best_E = E
            best_pts = opt_pts
    
    return best_pts, best_E

def construct():
    start_time = time.time()
    
    best_pts = None
    best_E = 1e18
    
    def _try_optimize(pts, max_iter_cg=2000, max_iter_lbfgs=3000):
        """Try multiple optimization methods and return the best result."""
        nonlocal best_pts, best_E
        
        # Method 1: Riemannian CG
        if time.time() < _DEADLINE - 10:
            opt_pts, E = _conjugate_gradient_riemannian(pts, max_iter=max_iter_cg, tol=1e-10)
            if E < best_E:
                best_E = E
                best_pts = opt_pts
        
        # Method 2: L-BFGS-B polish
        if best_pts is not None and time.time() < _DEADLINE - 10:
            try:
                polished_pts, polished_E = _riemannian_lbfgs(best_pts, max_iter=max_iter_lbfgs, tol=1e-9)
                if polished_E < best_E:
                    best_E = polished_E
                    best_pts = polished_pts
            except Exception:
                pass
        
        return best_pts, best_E
    
    # Phase 1: Try multiple diverse initializations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_start())
    
    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024, 31415, 27182]:
        if time.time() > _DEADLINE - 40:
            break
        initializations.append(_random_start(seed))
    
    # Jittered Fibonacci
    for jitter in [0.01, 0.05, 0.1, 0.2]:
        if time.time() > _DEADLINE - 40:
            break
        initializations.append(_jittered_fibonacci(jitter, seed=77 + int(jitter * 100)))
    
    # Optimize each initialization
    for i, init_pts in enumerate(initializations):
        if time.time() > _DEADLINE - 40:
            break
        
        # Use fewer iterations for later initializations to save time
        remaining = _DEADLINE - 40 - time.time()
        n_remaining = len(initializations) - i
        iter_per_init = max(500, int(remaining / max(1, n_remaining)))
        
        _try_optimize(init_pts, max_iter_cg=min(2000, iter_per_init), 
                     max_iter_lbfgs=min(3000, iter_per_init))
    
    # Phase 2: Particle swarm on the best configuration
    if best_pts is not None and time.time() < _DEADLINE - 25:
        try:
            ps_pts, ps_E = _particle_swarm(best_pts, max_iter=100, n_particles=8)
            if ps_E < best_E:
                best_E = ps_E
                best_pts = ps_pts
        except Exception:
            pass
    
    # Phase 3: Stochastic local search to escape local minima
    if best_pts is not None and time.time() < _DEADLINE - 20:
        try:
            sls_pts, sls_E = _stochastic_local_search(best_pts, n_restarts=15, max_iter=400)
            if sls_E < best_E:
                best_E = sls_E
                best_pts = sls_pts
        except Exception:
            pass
    
    # Phase 3b: L-BFGS polish after SLS
    if best_pts is not None and time.time() < _DEADLINE - 15:
        try:
            polished_pts, polished_E = _riemannian_lbfgs(best_pts, max_iter=3000, tol=1e-10)
            if polished_E < best_E:
                best_E = polished_E
                best_pts = polished_pts
        except Exception:
            pass
    
    # Phase 4: Final polish with tight tolerance
    if best_pts is not None and time.time() < _DEADLINE - 5:
        try:
            final_pts, final_E = _conjugate_gradient_riemannian(best_pts, max_iter=2000, tol=1e-12)
            if final_E < best_E:
                best_E = final_E
                best_pts = final_pts
        except Exception:
            pass
    
    # Phase 5: High-precision L-BFGS polish
    if best_pts is not None and time.time() < _DEADLINE - 3:
        try:
            polished_pts, polished_E = _riemannian_lbfgs(best_pts, max_iter=5000, tol=1e-12)
            if polished_E < best_E:
                best_E = polished_E
                best_pts = polished_pts
        except Exception:
            pass
    
    # Phase 6: Final CG with very tight tolerance
    if best_pts is not None and time.time() < _DEADLINE - 1:
        try:
            final_pts, final_E = _conjugate_gradient_riemannian(best_pts, max_iter=3000, tol=1e-14)
            if final_E < best_E:
                best_E = final_E
                best_pts = final_pts
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return Fibonacci
        return _fibonacci_start()
    
    return _project(best_pts)
# EVOLVE-BLOCK-END
