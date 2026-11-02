"""Improved packing: smooth surrogate objective with soft-min for nonsmooth constraints."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # Safety margin before 150s kill

def _score(centers):
    """Compute the maximum common radius for given centers (exact, for evaluation)."""
    centers = np.asarray(centers)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    n = len(centers)
    if n < 2:
        return r_wall
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min(values, beta=50.0):
    """Smooth approximation to min via log-sum-exp (negative log-sum-exp)."""
    values = np.asarray(values)
    m = np.min(values)
    exponent = -beta * (values - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: minimize -soft_min(all constraints).
    Constraints: wall distances (0.5 - |c_ik|) and half pairwise distances.
    Each pair counted exactly once.
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: 0.5 - |c_ik| for each coordinate
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances (each pair once)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    iu = np.triu_indices_from(pair_constraints, k=1)
    pair_constraints = pair_constraints[iu]
    
    # Combine all constraints
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    
    # Soft-min of all constraints
    s = _soft_min(all_constraints, beta)
    
    # Penalty for going outside bounds (soft)
    out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
    penalty = 10.0 * np.sum(out_of_bounds ** 2)
    
    return -s + penalty

def _epigraph_obj_and_grad(x, beta=200.0):
    """
    Epigraph formulation: minimize (1 - t) subject to soft constraints.
    x = [centers (150), t (1)]
    Returns (objective, gradient).
    """
    n = len(x) - 1
    c = x[:n].reshape(-1, 3)
    t = x[n]
    
    # Wall constraints: 0.5 - |c_ik| for each coordinate
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances (each pair once)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    iu = np.triu_indices_from(pair_constraints, k=1)
    pair_constraints = pair_constraints[iu]
    
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    
    # Soft-min penalty: penalize if t > soft_min(constraints)
    s = _soft_min(all_constraints, beta)
    viol = np.maximum(0, t - s)
    penalty = 0.5 * beta * viol * viol
    
    obj = (1.0 - t) + penalty
    
    # Gradient
    grad = np.zeros_like(x)
    grad[n] = -1.0 + beta * viol
    
    # Gradient of soft-min w.r.t. constraints
    m = np.min(all_constraints)
    exp_vals = np.exp(-beta * (all_constraints - m))
    exp_vals = np.clip(exp_vals, 0, 1e300)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient w.r.t. centers from wall constraints
    dwall = weights[:3 * N]
    # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    wall_grad = np.zeros_like(c)
    wall_grad[:, 0] = -np.sign(c[:, 0]) * dwall[0::3]
    wall_grad[:, 1] = -np.sign(c[:, 1]) * dwall[1::3]
    wall_grad[:, 2] = -np.sign(c[:, 2]) * dwall[2::3]
    
    # Gradient w.r.t. centers from pair constraints
    dpair = weights[3 * N:]
    # d(dist_ij / 2) / d(c_i) = (c_i - c_j) / (2 * dist_ij)
    pair_grad = np.zeros_like(c)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = dists[i, j]
            if d > 1e-15:
                factor = dpair[idx] / (2.0 * d)
                pair_grad[i] += factor * (c[i] - c[j])
                pair_grad[j] -= factor * (c[i] - c[j])
            idx += 1
    
    # Chain rule: gradient of s w.r.t. centers is negative of constraint gradient
    # Because we minimize -s, but here we have penalty on (t - s), so d/dx = -beta*viol * ds/dx
    if viol > 0:
        grad[:n] = -beta * viol * (wall_grad + pair_grad)
    
    return obj, grad

def _smooth_gradient(x, beta=50.0):
    """Numerical gradient of the smooth objective."""
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (_smooth_objective(xp, beta) - _smooth_objective(xm, beta)) / (2 * eps)
    return grad

def _initial_grid():
    """FCC-inspired grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    return grid[perm[:N]].copy()

def _initial_fcc_like():
    """Initialize with FCC-like structure: layers of hexagonal packing."""
    centers = np.zeros((N, 3))
    n_layers = 5
    per_layer = N // n_layers
    remaining = N - per_layer * n_layers
    
    for layer in range(n_layers):
        z = (layer + 0.5) / n_layers - 0.5
        n_in_layer = per_layer + (1 if layer < remaining else 0)
        
        for i in range(n_in_layer):
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4.0 - 0.5
            y = (row + 0.5) / 3.0 - 0.5
            if row % 2 == 1:
                x += 0.25 / 4.0
            centers[layer * per_layer + i] = [x, y, z]
    
    idx = per_layer * n_layers
    while idx < N:
        centers[idx] = [0, 0, 0]
        idx += 1
    
    return centers

def _initial_bcc():
    """BCC-inspired initialization: body-centered cubic layers."""
    centers = []
    # 3x3x3 grid = 27, plus body centers
    m = 3
    ticks = (np.arange(m) + 0.5) / m - 0.5
    for z in ticks:
        for y in ticks:
            for x in ticks:
                centers.append([x, y, z])
            # Add body centers between
    # Also add face centers
    for z in ticks:
        for y in ticks:
            centers.append([0, y, z])
            centers.append([y, 0, z])
    # Add more to reach 50
    rng = np.random.RandomState(123)
    while len(centers) < N:
        centers.append(rng.uniform(-0.45, 0.45, 3))
    return np.array(centers[:N])

def _local_search_polish(centers, max_iters=500, time_limit=20.0):
    """Coordinate descent local search with adaptive step size."""
    cur = centers.copy()
    cur_score = _score(cur)
    step = 0.01
    t_start = time.time()
    
    for it in range(max_iters):
        if time.time() - t_start > time_limit:
            break
        improved = False
        for i in range(N):
            for coord in range(3):
                for delta in [step, -step, step*0.5, -step*0.5, step*2, -step*2, step*0.1, -step*0.1]:
                    new_c = cur.copy()
                    new_c[i, coord] += delta
                    if np.all(np.abs(new_c) <= 0.5):
                        s = _score(new_c)
                        if s > cur_score + 1e-12:
                            cur = new_c
                            cur_score = s
                            improved = True
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
    
    return cur, cur_score

def _high_precision_polish(centers, time_limit=10.0):
    """Final high-precision polish with very small steps + epigraph optimizer."""
    cur = centers.copy()
    cur_score = _score(cur)
    t_start = time.time()
    step = 1e-5
    
    for it in range(2000):
        if time.time() - t_start > time_limit:
            break
        improved = False
        for i in range(N):
            for coord in range(3):
                for delta in [step, -step]:
                    new_c = cur.copy()
                    new_c[i, coord] += delta
                    if np.all(np.abs(new_c) <= 0.5):
                        s = _score(new_c)
                        if s > cur_score + 1e-14:
                            cur = new_c
                            cur_score = s
                            improved = True
        if not improved:
            step *= 0.5
            if step < 1e-10:
                break
    
    # Epigraph-based final optimization
    if time.time() - t_start < time_limit - 2.0:
        try:
            x0 = np.concatenate([cur.ravel(), [cur_score]])
            res = minimize(
                lambda x: _epigraph_obj_and_grad(x, beta=500.0)[0],
                lambda x: _epigraph_obj_and_grad(x, beta=500.0)[1],
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * 3 * N + [(0.0, 0.5)],
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            new_c = res.x[:3 * N].reshape(-1, 3)
            new_score = _score(new_c)
            if new_score > cur_score:
                cur = new_c
                cur_score = new_score
        except Exception:
            pass
    
    # One more round of tiny-step polish after epigraph
    remaining = time_limit - (time.time() - t_start)
    if remaining > 1.0:
        step = 1e-6
        for it in range(500):
            if time.time() - t_start > time_limit:
                break
            improved = False
            for i in range(N):
                for coord in range(3):
                    for delta in [step, -step]:
                        new_c = cur.copy()
                        new_c[i, coord] += delta
                        if np.all(np.abs(new_c) <= 0.5):
                            s = _score(new_c)
                            if s > cur_score + 1e-15:
                                cur = new_c
                                cur_score = s
                                improved = True
            if not improved:
                step *= 0.5
                if step < 1e-11:
                    break
    
    return cur, cur_score

def _repulsion_anneal(centers, time_limit=15.0):
    """Vectorized repulsion-based annealing: push spheres apart using soft repulsion."""
    cur = centers.copy()
    t_start = time.time()
    
    # Anneal from high to low temperature
    for temp in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        if time.time() - t_start > time_limit:
            break
        
        for it in range(200):
            if time.time() - t_start > time_limit:
                break
            
            # Vectorized pairwise repulsion
            diff = cur[:, np.newaxis, :] - cur[np.newaxis, :, :]  # (N, N, 3)
            d = np.linalg.norm(diff, axis=2)  # (N, N)
            np.fill_diagonal(d, 1e10)
            # Soft repulsion force magnitude: 1/(d^2 + eps)
            f_mag = 1.0 / (d * d + 0.01)  # (N, N)
            np.fill_diagonal(f_mag, 0.0)
            # Force vector: f_mag * diff / d
            forces = np.sum(f_mag[:, :, np.newaxis] * diff / d[:, :, np.newaxis], axis=1)
            
            # Wall repulsion (vectorized)
            wall_force = np.zeros_like(cur)
            for coord in range(3):
                over = np.maximum(0, cur[:, coord] - 0.4)
                under = np.maximum(0, -0.4 - cur[:, coord])
                wall_force[:, coord] = -5.0 * over + 5.0 * under
            
            forces += wall_force
            
            # Update with annealing
            cur += temp * forces
            # Project back into cube
            cur = np.clip(cur, -0.499, 0.499)
    
    return cur

def _metropolis_mc(centers, time_limit=20.0, n_restarts=3):
    """Metropolis Monte Carlo to escape local minima.
    
    Uses the score function directly: propose random perturbations,
    accept if score improves or with probability exp(-delta/T).
    """
    cur = centers.copy()
    cur_score = _score(cur)
    t_start = time.time()
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() - t_start > time_limit:
            break
        
        # Anneal temperature
        temps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
        for T in temps:
            if time.time() - t_start > time_limit:
                break
            
            for it in range(300):
                if time.time() - t_start > time_limit:
                    break
                
                # Propose: perturb one sphere by a small random amount
                i = rng.randint(N)
                step_size = T * 5.0
                proposal = cur.copy()
                proposal[i] += rng.normal(0, step_size, 3)
                proposal[i] = np.clip(proposal[i], -0.499, 0.499)
                
                new_score = _score(proposal)
                
                # Metropolis criterion
                delta = new_score - cur_score
                if delta > 0 or (T > 0 and rng.random() < np.exp(delta / max(T, 1e-10))):
                    cur = proposal
                    cur_score = new_score
    
    return cur, cur_score

def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    # Try multiple initializations
    inits = [
        ("grid", _initial_grid()),
        ("fcc", _initial_fcc_like()),
        ("bcc", _initial_bcc()),
    ]
    
    best_centers = None
    best_score = -np.inf
    
    for name, init in inits:
        if time.time() > deadline - 25:
            break
        
        cur = init.copy()
        cur_score = _score(cur)
        
        if best_centers is None or cur_score > best_score:
            best_centers = cur.copy()
            best_score = cur_score
        
        # Repulsion anneal (now vectorized, much faster)
        if time.time() < deadline - 35:
            try:
                rep_cur = _repulsion_anneal(cur, time_limit=min(8.0, deadline - time.time() - 35))
                rep_score = _score(rep_cur)
                if rep_score > best_score:
                    best_centers = rep_cur.copy()
                    best_score = rep_score
                    cur = rep_cur.copy()
            except Exception:
                pass
        
        # Metropolis MC to escape local minima (new phase using freed time)
        if time.time() < deadline - 25:
            try:
                mc_cur, mc_score = _metropolis_mc(
                    cur, 
                    time_limit=min(10.0, deadline - time.time() - 25),
                    n_restarts=2
                )
                if mc_score > best_score:
                    best_centers = mc_cur.copy()
                    best_score = mc_score
                    cur = mc_cur.copy()
            except Exception:
                pass
        
        # Optimize with smooth objective (fewer betas to save time)
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 15:
                break
            
            try:
                res = minimize(
                    _smooth_objective,
                    cur.ravel(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                
                if new_score > best_score:
                    best_centers = new_centers.copy()
                    best_score = new_score
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # Epigraph optimization for finer control
        if time.time() < deadline - 12:
            try:
                x0 = np.concatenate([cur.ravel(), [_score(cur)]])
                res = minimize(
                    lambda x: _epigraph_obj_and_grad(x, beta=300.0)[0],
                    lambda x: _epigraph_obj_and_grad(x, beta=300.0)[1],
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N + [(0.0, 0.5)],
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 80}
                )
                new_centers = res.x[:3 * N].reshape(-1, 3)
                new_score = _score(new_centers)
                if new_score > best_score:
                    best_centers = new_centers.copy()
                    best_score = new_score
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # Polish with coordinate descent
        if time.time() < deadline - 8:
            try:
                pol_cur, pol_score = _local_search_polish(
                    best_centers, 
                    max_iters=200,
                    time_limit=min(8.0, deadline - time.time() - 8)
                )
                if pol_score > best_score:
                    best_centers = pol_cur.copy()
                    best_score = pol_score
            except Exception:
                pass
    
    # Final high-precision polish if time remains
    if time.time() < deadline - 3 and best_centers is not None:
        try:
            hp_cur, hp_score = _high_precision_polish(
                best_centers,
                time_limit=min(5.0, deadline - time.time() - 2)
            )
            if hp_score > best_score:
                best_centers = hp_cur.copy()
                best_score = hp_score
        except Exception:
            pass
    
    return best_centers
