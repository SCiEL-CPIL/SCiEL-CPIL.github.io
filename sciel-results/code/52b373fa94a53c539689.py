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

def _epigraph_objective(x, beta=50.0):
    """
    Epigraph formulation: minimize t + penalty, s.t. t >= soft_min(constraints).
    We minimize -soft_min(constraints) + bound penalty, which is equivalent.
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

def _epigraph_gradient(x, beta=50.0):
    """
    Analytic gradient of the epigraph objective.
    
    The soft-min gradient: d/dx_i [-soft_min(f)] = sum_j w_j * df_j/dx_i
    where w_j = exp(-beta*(f_j - m)) / sum_k exp(-beta*(f_k - m))
    """
    c = x.reshape(-1, 3)
    n = c.shape[0]
    
    # Wall constraints: f_w = 0.5 - |c_ik|
    # df_w/dx_ik = -sign(c_ik)
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    np.fill_diagonal(dists, 1e-10)
    pair_constraints = dists / 2.0
    iu = np.triu_indices_from(pair_constraints, k=1)
    pair_vals = pair_constraints[iu]
    
    # All constraint values
    all_constraints = np.concatenate([wall_constraints, pair_vals])
    m = np.min(all_constraints)
    exponent = -beta * (all_constraints - m)
    exponent = np.clip(exponent, -700, 700)
    weights = np.exp(exponent)
    weights /= np.sum(weights)
    
    # Gradient of -soft_min = sum_j w_j * (gradient of -f_j)
    # = -sum_j w_j * (gradient of f_j)
    # Actually: soft_min = m - (1/beta)*log(sum(exp(-beta*(f-m))))
    # d(soft_min)/dx = sum_j w_j * df_j/dx  where w_j = exp(-beta*(f_j-m))/sum(...)
    # So d(-soft_min)/dx = -sum_j w_j * df_j/dx
    
    grad = np.zeros_like(x)
    
    # Wall part: f_w_ik = 0.5 - |c_ik|, df/dx_ik = -sign(c_ik)
    wall_weights = weights[:3*n].reshape(n, 3)
    sign_c = np.sign(c)
    # d(soft_min)/dx from walls = sum wall_w * (-sign_c) = -wall_w * sign_c
    # d(-soft_min)/dx = wall_w * sign_c
    grad += wall_weights * sign_c
    
    # Pair part: f_p_ij = |c_i - c_j| / 2
    # df_p_ij/dc_i = (c_i - c_j) / (2 * |c_i - c_j|) = diff_ij / (2 * dist_ij)
    # df_p_ij/dc_j = -diff_ij / (2 * dist_ij)
    pair_weights = weights[3*n:]
    # iu[0] = i indices, iu[1] = j indices
    i_idx, j_idx = iu
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = pair_weights[k]
        d = diff[i, j]  # c_i - c_j
        dist = dists[i, j]
        if dist > 1e-10:
            unit = d / dist
            # d(soft_min)/dx_i += w * unit/2
            # d(soft_min)/dx_j += w * (-unit/2)
            # d(-soft_min)/dx_i += -w * unit/2
            # d(-soft_min)/dx_j += -w * (-unit/2) = w * unit/2
            grad[3*i:3*i+3] -= w * unit / 2.0
            grad[3*j:3*j+3] += w * unit / 2.0
    
    # Bound penalty gradient
    out = np.abs(c) - 0.4999
    mask = out > 0
    if np.any(mask):
        penalty_grad = np.zeros_like(x)
        # d/dx [10 * sum max(0, |x|-0.4999)^2] = 10 * 2 * max(0,|x|-0.4999) * sign(x)
        pg = 20.0 * np.where(mask, out * np.sign(c), 0)
        grad += pg.ravel()
    
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
    """Final high-precision polish with very small steps."""
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
    
    return cur, cur_score

def _gradient_polish(centers, time_limit=15.0, beta_start=200.0, beta_end=2000.0):
    """
    Gradient-based polish using the analytic gradient of the soft-min objective.
    Anneals beta from low to high for increasingly precise minimization.
    """
    cur = centers.copy()
    cur_score = _score(cur)
    t_start = time.time()
    
    # Anneal beta for progressively sharper soft-min
    n_steps = 8
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
        
        try:
            res = minimize(
                _epigraph_objective,
                cur.ravel(),
                args=(beta,),
                jac=_epigraph_gradient,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * 3 * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_score = _score(new_centers)
            
            if new_score > cur_score:
                cur = new_centers.copy()
                cur_score = new_score
        except Exception:
            pass
    
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
        if time.time() > deadline - 30:
            break
        
        cur = init.copy()
        cur_score = _score(cur)
        
        if best_centers is None or cur_score > best_score:
            best_centers = cur.copy()
            best_score = cur_score
        
        # Repulsion anneal (now vectorized, much faster)
        if time.time() < deadline - 40:
            try:
                rep_cur = _repulsion_anneal(cur, time_limit=min(8.0, deadline - time.time() - 40))
                rep_score = _score(rep_cur)
                if rep_score > best_score:
                    best_centers = rep_cur.copy()
                    best_score = rep_score
                    cur = rep_cur.copy()
            except Exception:
                pass
        
        # Metropolis MC to escape local minima
        if time.time() < deadline - 30:
            try:
                mc_cur, mc_score = _metropolis_mc(
                    cur, 
                    time_limit=min(10.0, deadline - time.time() - 30),
                    n_restarts=2
                )
                if mc_score > best_score:
                    best_centers = mc_cur.copy()
                    best_score = mc_score
                    cur = mc_cur.copy()
            except Exception:
                pass
        
        # Optimize with smooth objective using analytic gradient
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 20:
                break
            
            try:
                res = minimize(
                    _epigraph_objective,
                    cur.ravel(),
                    args=(beta,),
                    jac=_epigraph_gradient,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                
                if new_score > best_score:
                    best_centers = new_centers.copy()
                    best_score = new_score
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # Polish with coordinate descent
        if time.time() < deadline - 10:
            try:
                pol_cur, pol_score = _local_search_polish(
                    best_centers, 
                    max_iters=200,
                    time_limit=min(8.0, deadline - time.time() - 10)
                )
                if pol_score > best_score:
                    best_centers = pol_cur.copy()
                    best_score = pol_score
            except Exception:
                pass
    
    # Final gradient-based polish with annealed beta (new, more effective)
    if time.time() < deadline - 8 and best_centers is not None:
        try:
            gp_cur, gp_score = _gradient_polish(
                best_centers,
                time_limit=min(12.0, deadline - time.time() - 8),
                beta_start=100.0,
                beta_end=5000.0
            )
            if gp_score > best_score:
                best_centers = gp_cur.copy()
                best_score = gp_score
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
