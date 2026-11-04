"""Combined approach: Gram-matrix formulation + Riemannian optimization on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150
TIME_LIMIT = 135.0  # Safety margin before 150s kill


def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|)
    and the Riemannian gradient (projected to tangent space) on the sphere.
    
    Uses cdist for fast pairwise distance computation.
    """
    n = len(pts)
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)
    
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        energy = 0.0
    else:
        energy = -np.sum(np.log(dist_upper))
    
    dist_sq = dist ** 2
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Euclidean gradient: grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    row_sums = np.sum(inv_dist_sq, axis=1)
    weighted_sum = inv_dist_sq @ pts
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum
    
    # Project to tangent space
    dot_products = np.sum(grad * pts, axis=1)
    rgrad = grad - dot_products[:, np.newaxis] * pts
    
    return energy, rgrad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        return 0.0
    return -np.sum(np.log(dist_upper))


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_spiral(n, seed=None):
    """Generate n points on sphere using golden angle spiral."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_based(n):
    """Generate points using icosahedron vertices as anchors for better initial distribution."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    fib = _fibonacci_spiral(n)
    result = fib.copy()
    for i in range(12):
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _riemannian_gradient_descent(pts, time_budget, lr_init=0.01, lr_decay=0.9999):
    """
    Riemannian gradient descent with adaptive step size and momentum.
    """
    t_start = time.time()
    n = len(pts)
    current_energy, rgrad = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    lr = lr_init
    patience = 0
    max_patience = 300
    
    # Momentum
    velocity = np.zeros_like(pts)
    momentum = 0.9
    
    while time.time() - t_start < time_budget:
        if time.time() - t_start >= time_budget:
            break
        
        e, rgrad = _energy_and_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Check convergence
        gnorm = np.linalg.norm(rgrad)
        if gnorm < 1e-12:
            break
        
        # Adaptive step size with momentum
        step = lr / (1.0 + gnorm)
        
        # Update velocity with momentum
        velocity = momentum * velocity - step * rgrad
        
        new_pts = pts + velocity
        new_pts = _project_to_sphere(new_pts)
        
        # Check if it improved
        new_e, _ = _energy_and_grad(new_pts)
        if new_e < e:
            pts = new_pts
            current_energy = new_e
            lr = min(lr * 1.02, 0.05)
        else:
            lr *= 0.7
            velocity *= 0.5
            if lr < 1e-10:
                break
    
    return best_pts, best_energy


def _adam_optimize(pts, time_limit=10.0):
    """Adam optimizer on the sphere with adaptive learning rate."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    lr_initial = 0.01
    lr_min = 0.0005
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr_decay = 0.997

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    lr = lr_initial

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    no_improve_count = 0
    patience = 800

    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Project gradient and momentum to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        m_tangent = m_hat - np.sum(m_hat * pts, axis=1, keepdims=True) * pts

        pts = pts - lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)

        lr = max(lr_min, lr * lr_decay)

        if t % 200 == 0:
            _, g_check = _energy_and_grad(pts)
            if np.linalg.norm(g_check) < 1e-10:
                break

    return best_pts, best_energy


def _simulated_annealing(pts, time_limit=15.0, seed=42):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.005
    T_min = 1e-7
    cooling = 0.996
    max_steps = 8000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        # Random perturbation
        noise_scale = T * 0.05
        perturbation = rng.randn(n, 3) * noise_scale
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        new_energy = _energy_only(new_pts)

        # Metropolis criterion
        if new_energy < current_energy or (T > 0 and rng.rand() < math.exp(min(0, -(new_energy - current_energy) / T))):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy


def _basin_hopping(pts, time_budget, n_restarts=8):
    """
    Basin hopping: perturb points, then polish with gradient descent/Adam.
    """
    t_start = time.time()
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    rng = np.random.default_rng(123)
    
    remaining = time_budget - (time.time() - t_start)
    if remaining < 10:
        return best_pts, best_energy
    
    per_restart = remaining / n_restarts
    
    for restart in range(n_restarts):
        if time.time() - t_start >= time_budget:
            break
        
        # Create perturbed version with varying noise scales
        perturbed = pts.copy()
        noise_scales = [0.15, 0.10, 0.05]
        for scale in noise_scales:
            noise = rng.standard_normal((len(pts), 3)) * scale * (1.0 - restart / (n_restarts + 1))
            perturbed = _project_to_sphere(perturbed + noise)
        
        # Polish with Adam
        adam_budget = per_restart * 0.6
        polished, e_polished = _adam_optimize(perturbed, adam_budget)
        
        # Quick GD polish
        gd_budget = per_restart * 0.3
        if time.time() - t_start < time_budget - 5:
            polished, e_polished = _riemannian_gradient_descent(polished, gd_budget, lr_init=0.003)
        
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = polished.copy()
    
    return best_pts, best_energy


def _lbfgs_polish(pts, time_budget):
    """
    L-BFGS optimization using scipy, operating on the flattened sphere coordinates
    with a projection-based approach.
    """
    t_start = time.time()
    n = len(pts)
    
    def objective_flat(x):
        if time.time() - t_start > time_budget:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - t_start > time_budget:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()
    
    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 3000,
            'maxfun': 8000,
            'ftol': 1e-15,
            'gtol': 1e-11,
            'maxls': 50,
            'maxcor': 25
        }
    )
    
    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    final_e = _energy_only(final_pts)
    
    current_e = _energy_only(pts)
    if final_e < current_e:
        return final_pts, final_e
    
    return pts, current_e


def _final_polish(pts, time_budget):
    """
    Final high-precision polish using L-BFGS with very tight tolerances.
    """
    t_start = time.time()
    n = len(pts)
    
    def objective_flat(x):
        if time.time() - t_start > time_budget:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - t_start > time_budget:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()
    
    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 8000,
            'maxfun': 15000,
            'ftol': 1e-18,
            'gtol': 1e-13,
            'maxls': 100,
            'maxcor': 30
        }
    )
    
    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    final_e = _energy_only(final_pts)
    
    current_e = _energy_only(pts)
    if final_e < current_e:
        return final_pts, final_e
    
    return pts, current_e


def _newton_polish(pts, time_budget):
    """
    Newton-type polish using approximate Hessian information via L-BFGS with
    very tight convergence criteria, followed by a few steps of projected Newton.
    """
    t_start = time.time()
    n = len(pts)
    
    # First, run L-BFGS with very tight settings
    def objective_flat(x):
        if time.time() - t_start > time_budget:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - t_start > time_budget:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()
    
    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'maxfun': 12000,
            'ftol': 1e-16,
            'gtol': 1e-14,
            'maxls': 80,
            'maxcor': 30
        }
    )
    
    best_pts = _project_to_sphere(result.x.reshape(-1, 3))
    best_e = _energy_only(best_pts)
    
    # Try a few more manual Newton-like steps
    for _ in range(5):
        if time.time() - t_start > time_budget - 2:
            break
        
        p = best_pts
        e, g = _energy_and_grad(p)
        
        if np.linalg.norm(g) < 1e-13:
            break
        
        # Approximate inverse Hessian via finite differences of gradient
        # Use a simple diagonal approximation in tangent space
        step_size = 0.001 / (np.linalg.norm(g) + 1e-10)
        new_pts = _project_to_sphere(p - step_size * g)
        new_e = _energy_only(new_pts)
        
        if new_e < best_e:
            best_pts = new_pts
            best_e = new_e
    
    return best_pts, best_e


def construct():
    t_start = time.time()
    
    # Start with multiple initializations and pick the best
    init_candidates = [
        _fibonacci_spiral(N),
        _fibonacci_spiral(N, seed=1),
        _icosahedron_based(N),
    ]
    
    # Quick GD on each to pick the best starting point
    best_init = None
    best_init_e = np.inf
    
    for init in init_candidates:
        if time.time() - t_start > TIME_LIMIT - 100:
            break
        init_copy = init.copy()
        _, e_quick = _riemannian_gradient_descent(init_copy, 2.0, lr_init=0.01)
        if e_quick < best_init_e:
            best_init_e = e_quick
            best_init = init_copy
    
    if best_init is None:
        best_init = _fibonacci_spiral(N)
    
    pts = best_init
    
    # Phase 1: Riemannian gradient descent with momentum (fast, gets us close)
    remaining = TIME_LIMIT - (time.time() - t_start)
    gd_budget = min(15.0, remaining * 0.15)
    pts, e_gd = _riemannian_gradient_descent(pts, gd_budget, lr_init=0.015)
    
    # Phase 2: Adam optimization for better convergence
    remaining = TIME_LIMIT - (time.time() - t_start)
    adam_budget = min(20.0, remaining * 0.2)
    pts, e_adam = _adam_optimize(pts, adam_budget)
    
    # Phase 3: Basin hopping to escape local minima
    remaining = TIME_LIMIT - (time.time() - t_start)
    bh_budget = min(35.0, remaining * 0.3)
    pts, e_bh = _basin_hopping(pts, bh_budget, n_restarts=8)
    
    # Phase 4: L-BFGS polish
    remaining = TIME_LIMIT - (time.time() - t_start)
    lbfgs_budget = min(25.0, remaining * 0.25)
    pts, e_lbfgs = _lbfgs_polish(pts, lbfgs_budget)
    
    # Phase 5: Newton-type polish for higher precision
    remaining = TIME_LIMIT - (time.time() - t_start)
    newton_budget = min(15.0, remaining * 0.2)
    pts, e_newton = _newton_polish(pts, newton_budget)
    
    # Phase 6: Final high-precision polish
    remaining = TIME_LIMIT - (time.time() - t_start)
    final_budget = min(remaining * 0.5, 10.0)
    if final_budget > 3:
        pts, e_final = _final_polish(pts, final_budget)
    
    # Final check and validation
    pts = _project_to_sphere(pts)
    
    if not np.all(np.isfinite(pts)):
        pts = _fibonacci_spiral(N)
    
    # Ensure minimum separation
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                pts[j] = pts[j] + 1e-8 * np.random.randn(3)
                pts[j] = pts[j] / np.linalg.norm(pts[j])
    
    return pts


# EVOLVE-BLOCK-END
