import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _calc_score(x):
    """Calculate the actual packing score (min radius)."""
    d_min = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(x[i] - x[j])
            if d < d_min:
                d_min = d
        d_b = 1.0 - np.linalg.norm(x[i])
        if d_b < d_min:
            d_min = d_b
    return d_min / 2.0

def _hex_lattice_init():
    """Generate initial hexagonal lattice points."""
    # Optimal spacing for hex lattice in unit disk for N points
    # Approximate area per circle = pi / N
    # Side length of hex cell ~ sqrt(2*area/sqrt(3))
    # Simpler: just try a few spacings
    best_points = None
    best_score = -1
    
    for trial in range(5):
        # Spacing from 0.2 to 0.3
        spacing = 0.22 + trial * 0.02
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points well inside boundary
        margin = spacing / 2.0 + 0.02
        mask = radius <= 1.0 - margin
        points, radius = points[mask], radius[mask]
        
        if len(points) >= N:
            # Select N points: prefer those closer to center
            idx = np.argsort(radius, kind="stable")[:N]
            candidate = points[idx]
            score = _calc_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
    
    if best_points is None:
        # Fallback: uniform random
        rng = np.random.default_rng(0)
        while True:
            pts = rng.uniform(-0.8, 0.8, size=(N, 2))
            if np.all(np.linalg.norm(pts, axis=1) < 0.9):
                return pts
    return best_points

def _soft_min_energy_and_grad(x, alpha):
    """
    Compute the soft-min energy and gradient.
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    x = x.reshape(N, 2)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Avoid division by zero in gradient calculation for i=j
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances: 1 - |x_i|
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    b_dists = 1.0 - radii
    
    p_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([p_dists, b_dists])
    
    # Clamp distances to avoid numerical issues
    all_dists = np.maximum(all_dists, 1e-10)
    
    exponents = -alpha * all_dists
    max_exp = np.max(exponents)
    exponents -= max_exp
    
    weights = np.exp(exponents)
    sum_weights = np.sum(weights)
    
    if sum_weights < 1e-300:
        return 0.0, np.zeros_like(x)
        
    energy = (1.0 / alpha) * (np.log(sum_weights) + max_exp)
    
    grad = np.zeros_like(x)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    w_pairs = weights[:len(p_dists)]
    d_pairs = p_dists
    
    diff_vec = x[i_idx] - x[j_idx]
    mask = d_pairs > 1e-8
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    
    np.add.at(grad, i_idx, -weighted_grad / sum_weights)
    np.add.at(grad, j_idx, weighted_grad / sum_weights)
    
    w_bounds = weights[len(p_dists):]
    radii_safe = np.where(radii < 1e-8, 1e-8, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    
    weighted_bounds = w_bounds[:, np.newaxis] * grad_bounds
    grad += -weighted_bounds / sum_weights
    
    return energy, grad

def _optimize(x_start, alphas, max_iters, deadline):
    """Run optimization sequence."""
    current_x = x_start.copy()
    for alpha in alphas:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            current_x = res.x.reshape(N, 2)
        except Exception:
            pass
    return current_x

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. Use 140s.
    deadline = start_time + 140.0
    
    # 1. Initialization
    x0 = _hex_lattice_init()
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    # 2. Initial Coarse-to-Fine Optimization
    # Keep this fast to get a decent baseline
    alphas_init = [10.0, 50.0, 200.0]
    x_current = _optimize(x0, alphas_init, 800, deadline)
    score_current = _calc_score(x_current)
    
    if score_current > best_score:
        best_score = score_current
        best_x = x_current.copy()
    
    # 3. Restart / Annealing Phase
    # Use remaining time for multiple restarts with perturbations
    # This helps escape local minima
    rng = np.random.default_rng(42)
    
    # Define restart configurations
    # (perturbation_scale, alphas_to_use, max_iters)
    restart_configs = [
        (0.05, [50.0, 200.0], 500),
        (0.03, [100.0, 300.0], 600),
        (0.08, [20.0, 100.0], 400),
        (0.02, [200.0, 500.0], 800),
        (0.04, [50.0, 150.0], 500),
        (0.06, [30.0, 100.0], 400),
        (0.01, [300.0, 600.0], 1000),
        (0.05, [100.0, 400.0], 700),
    ]
    
    for scale, alphas, max_iters in restart_configs:
        if time.time() > deadline - 10:
            break
        
        # Perturb the best known solution
        x_perturbed = best_x + rng.normal(0, scale, size=best_x.shape)
        
        # Optimize the perturbed solution
        x_new = _optimize(x_perturbed, alphas, max_iters, deadline)
        score_new = _calc_score(x_new)
        
        if score_new > best_score:
            best_score = score_new
            best_x = x_new.copy()
    
    # 4. Final Polish
    if time.time() < deadline - 5:
        # Very fine alpha for final precision
        x_final = _optimize(best_x, [500.0, 1000.0], 1500, deadline)
        score_final = _calc_score(x_final)
        if score_final > best_score:
            best_score = score_final
            best_x = x_final.copy()
            
    return best_x
