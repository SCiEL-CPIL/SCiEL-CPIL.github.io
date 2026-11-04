"""Combined approach: Multi-phase Riemannian optimization for logarithmic energy minimization."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 135.0  # Safety margin before 150s kill


def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|)
    and the Euclidean gradient on the sphere.
    """
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    np.fill_diagonal(dist, 1.0)
    
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        energy = 0.0
    else:
        energy = -np.sum(np.log(dist_upper))
    
    # Gradient: grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sums = np.sum(inv_dist_sq, axis=1)
    weighted_sum = inv_dist_sq @ pts
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum
    
    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
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


def _random_init(n, seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


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
    max_patience = 500
    
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

    lr_initial = 0.015
    lr_min = 0.0008
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr_decay = 0.996

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    lr = lr_initial

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    no_improve_count = 0
    patience = 600

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

        if t % 150 == 0:
            _, g_check = _energy_and_grad(pts)
            if np.linalg.norm(g_check) < 1e-11:
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

    T = 0.008
    T_min = 1e-7
    cooling = 0.995
    max_steps = 10000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        # Random perturbation with varying scale
        noise_scale = T * 0.08
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


def _basin_hopping(pts, time_budget, n_restarts=10):
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
        noise_scale = 0.12 * (1.0 - restart / (n_restarts + 1))
        noise = rng.standard_normal((len(pts), 3)) * noise_scale
        perturbed = _project_to_sphere(perturbed + noise)
        
        # Polish with Adam
        adam_budget = per_restart * 0.55
        if adam_budget > 1.0:
            polished, e_polished = _adam_optimize(perturbed, adam_budget)
        else:
            polished, e_polished = perturbed, _energy_only(perturbed)
        
        # Quick GD polish
        gd_budget = per_restart * 0.3
        if time.time() - t_start < time_budget - 5 and gd_budget > 0.5:
            polished, e_polished = _riemannian_gradient_descent(polished, gd_budget, lr_init=0.005)
        
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = polished.copy()
    
    return best_pts, best_energy


def _lbfgs_polish(pts, time_budget, ftol=1e-14, gtol=1e-12, maxiter=5000):
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
            'maxiter': maxiter,
            'maxfun': 12000,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 80,
            'maxcor': 30
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
    Final high-precision polish using L-BFGS with very tight tolerances,
    followed by multiple refinement passes.
    """
    t_start = time.time()
    n = len(pts)
    best_pts = pts.copy()
    best_e = _energy_only(pts)
    
    # Multiple L-BFGS passes with decreasing tolerances
    tolerances = [
        (1e-13, 1e-11, 3000),
        (1e-15, 1e-13, 4000),
        (1e-16, 1e-14, 5000),
    ]
    
    for ftol, gtol, maxiter in tolerances:
        if time.time() - t_start > time_budget - 2:
            break
        
        budget = min(8.0, time_budget - (time.time() - t_start))
        if budget < 1.0:
            break
        
        polished, e_polished = _lbfgs_polish(best_pts, budget, ftol=ftol, gtol=gtol, maxiter=maxiter)
        
        if e_polished < best_e:
            best_e = e_polished
            best_pts = polished
    
    return best_pts, best_e


def construct():
    t_start = time.time()
    
    # Start with multiple initializations and pick the best
    init_candidates = [
        _fibonacci_spiral(N),
        _fibonacci_spiral(N, seed=1),
        _fibonacci_spiral(N, seed=42),
        _icosahedron_based(N),
        _random_init(N, seed=7),
        _random_init(N, seed=123),
    ]
    
    # Quick GD on each to pick the best starting point
    best_init = None
    best_init_e = np.inf
    
    for init in init_candidates:
        if time.time() - t_start > TIME_LIMIT - 110:
            break
        init_copy = init.copy()
        _, e_quick = _riemannian_gradient_descent(init_copy, 3.0, lr_init=0.01)
        if e_quick < best_init_e:
            best_init_e = e_quick
            best_init = init_copy
    
    if best_init is None:
        best_init = _fibonacci_spiral(N)
    
    pts = best_init
    
    # Phase 1: Riemannian gradient descent with momentum (fast, gets us close)
    remaining = TIME_LIMIT - (time.time() - t_start)
    gd_budget = min(12.0, remaining * 0.12)
    pts, e_gd = _riemannian_gradient_descent(pts, gd_budget, lr_init=0.012)
    
    # Phase 2: Adam optimization for better convergence
    remaining = TIME_LIMIT - (time.time() - t_start)
    adam_budget = min(18.0, remaining * 0.18)
    pts, e_adam = _adam_optimize(pts, adam_budget)
    
    # Phase 3: Simulated annealing to escape local minima
    remaining = TIME_LIMIT - (time.time() - t_start)
    sa_budget = min(15.0, remaining * 0.15)
    if sa_budget > 3.0:
        pts, e_sa = _simulated_annealing(pts, sa_budget, seed=42)
        if e_sa < _energy_only(pts):
            pts, e_sa = pts, e_sa
    
    # Phase 4: Basin hopping to escape local minima
    remaining = TIME_LIMIT - (time.time() - t_start)
    bh_budget = min(30.0, remaining * 0.3)
    pts, e_bh = _basin_hopping(pts, bh_budget, n_restarts=10)
    
    # Phase 5: L-BFGS polish
    remaining = TIME_LIMIT - (time.time() - t_start)
    lbfgs_budget = min(20.0, remaining * 0.2)
    pts, e_lbfgs = _lbfgs_polish(pts, lbfgs_budget, ftol=1e-13, gtol=1e-11, maxiter=4000)
    
    # Phase 6: Final high-precision polish
    remaining = TIME_LIMIT - (time.time() - t_start)
    final_budget = min(remaining * 0.4, 15.0)
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
    
    # Final safety check
    if time.time() - t_start > TIME_LIMIT - 5:
        # Emergency: just return what we have
        pass
    
    return pts
