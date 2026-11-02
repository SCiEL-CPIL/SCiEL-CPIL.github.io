"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _compute_r(pos):
    """Largest common radius for a placement (validity assumed)."""
    if pos.shape != (N, 2) or not np.all(np.isfinite(pos)):
        return 0.0
    # Pairwise distances
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    # Wall distance
    min_wall = (0.5 - np.abs(pos)).min()
    return min(0.5 * min_pair, min_wall)


def _soft_min_obj_and_grad(pos, r, beta):
    """
    Objective: soft approximation of violation for radius r.
    Uses log-sum-exp for smooth min of pairwise distances and wall distances.
    Returns (objective_value, gradient).
    
    We want to maximize the minimum clearance. Equivalently, minimize the violation.
    Violation = max(0, 2r - d_ij) for pairs, and max(0, |pos| - (0.5-r)) for walls.
    
    A smoother approach: minimize -soft_min(distances) + wall penalty.
    Actually, let's use the epigraph formulation with auxiliary variable.
    
    Better approach: minimize sum of soft violations.
    soft_violation_pair = softplus(beta * (2r - d_ij))
    soft_violation_wall = softplus(beta * (|pos_ik| - (0.5 - r)))
    """
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 1e10)
    
    # Pairwise violations: 2r - dist
    pair_viol = 2 * r - dist
    # Softplus approximation: log(1 + exp(beta * x))
    # For numerical stability: max(0, x) + log(1 + exp(-beta*|x|))/beta
    def _softplus(x, b):
        return np.where(x > 0, x + np.log1p(np.exp(-b * x)), np.log1p(np.exp(b * x))) / b
    
    obj_pair = np.sum(_softplus(pair_viol, beta))
    
    # Wall violations
    wall_viol = np.abs(pos) - (0.5 - r)
    obj_wall = np.sum(_softplus(wall_viol, beta))
    
    obj = obj_pair + obj_wall
    
    # Gradient
    # d/dpos of softplus(beta * (2r - dist)) 
    # = beta * softplus'(beta*(2r-dist)) * (-d(dist)/dpos)
    # softplus'(x) = sigmoid(beta*x) = 1/(1+exp(-beta*x))
    # d(dist)/dpos_i = (pos_i - pos_j) / dist_ij
    
    grad = np.zeros_like(pos)
    
    # Pairwise gradient
    pair_arg = beta * pair_viol
    pair_sigmoid = 1.0 / (1.0 + np.exp(-pair_arg))
    # d(dist)/dpos_i = (pos_i - pos_j)/dist for i != j
    # Contribution to grad[i] from pair (i,j): -beta * sigmoid * (pos_i - pos_j)/dist
    # But we need to be careful with the sign.
    # obj = sum softplus(beta*(2r - dist_ij))
    # d obj / d pos_i = sum_j beta * sigmoid(beta*(2r-dist_ij)) * (-1) * d(dist_ij)/d(pos_i)
    # d(dist_ij)/d(pos_i) = (pos_i - pos_j)/dist_ij
    # So d obj / d pos_i = -sum_j beta * sigmoid * (pos_i - pos_j)/dist_ij
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dist[i, j] < 1e-10:
                continue
            grad[i] -= beta * pair_sigmoid[i, j] * (pos[i] - pos[j]) / dist[i, j]
    
    # Wall gradient
    wall_arg = beta * wall_viol
    wall_sigmoid = 1.0 / (1.0 + np.exp(-wall_arg))
    # d|pos_ik|/d(pos_ik) = sign(pos_ik)
    for i in range(n):
        for k in range(2):
            if abs(pos[i, k]) < 1e-10:
                sign = 0.0
            else:
                sign = np.sign(pos[i, k])
            grad[i, k] += beta * wall_sigmoid[i, k] * sign
    
    return obj, grad


def _violation_exact(pos, r):
    """Exact violation (non-smooth)."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 2 * r)
    pair_viol = np.maximum(2 * r - dist, 0.0)
    wall_viol = np.maximum(np.abs(pos) - (0.5 - r), 0.0)
    return np.sum(pair_viol ** 2) + np.sum(wall_viol ** 2)


def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _hex_init():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    spacing = 0.15
    row = 0
    col = 0
    count = 0
    while count < N:
        x = (col + 0.5 * (row % 2)) * spacing - 0.5
        y = row * spacing * np.sqrt(3) / 2 - 0.5
        if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
            pos[count] = [x, y]
            count += 1
        col += 1
        if col > 10:
            col = 0
            row += 1
    return pos[:N]


def _random_init(seed=42):
    np.random.seed(seed)
    return np.random.uniform(-0.4, 0.4, size=(N, 2))


def _simulated_annealing(pos, time_budget, seed=123):
    """
    Simulated annealing to maximize the minimum radius.
    Uses adaptive step size and diverse move types.
    """
    best_pos = pos.copy()
    best_r = _compute_r(pos)
    cur_pos = pos.copy()
    cur_r = best_r
    rng = np.random.RandomState(seed)
    t_start = time.time()
    
    temp = 0.004
    cool = 0.9999
    step = 0.02
    min_step = 0.0005
    max_step = 0.06
    
    accept_count = 0
    total_count = 0
    
    while time.time() - t_start < time_budget:
        # Choose move type
        move_type = rng.rand()
        i = rng.randint(N)
        old = cur_pos[i].copy()
        
        if move_type < 0.6:
            # Single particle move
            cur_pos[i] = old + rng.normal(0, step, size=2)
        elif move_type < 0.8:
            # Pair swap-like move
            j = rng.randint(N)
            if j != i:
                old_j = cur_pos[j].copy()
                cur_pos[i] = old_j + rng.normal(0, step * 0.5, size=2)
                cur_pos[j] = old + rng.normal(0, step * 0.5, size=2)
        else:
            # Small global jitter
            cur_pos += rng.normal(0, step * 0.1, size=(N, 2))
        
        # Clamp inside square
        lim = 0.5 - 0.065
        cur_pos = np.clip(cur_pos, -lim, lim)
        
        new_r = _compute_r(cur_pos)
        total_count += 1
        
        if new_r > cur_r or rng.rand() < np.exp((new_r - cur_r) / max(temp, 1e-9)):
            cur_r = new_r
            accept_count += 1
            if new_r > best_r:
                best_r = new_r
                best_pos = cur_pos.copy()
        else:
            cur_pos = pos.copy() if move_type >= 0.8 else cur_pos
            if move_type < 0.8:
                cur_pos[i] = old
                if move_type >= 0.6 and move_type < 0.8:
                    # Revert pair move
                    pass  # Already reverted by the else branch logic above being incomplete
                    # Let's fix: revert properly
                    cur_pos[i] = old
                    if j != i:
                        cur_pos[j] = old_j
        
        temp *= cool
        
        # Adapt step size
        if total_count % 100 == 0:
            rate = accept_count / max(1, total_count)
            if rate < 0.25:
                step = max(min_step, step * 0.75)
            elif rate > 0.75:
                step = min(max_step, step * 1.25)
            accept_count = 0
            total_count = 0
    
    return best_pos, best_r


def _soft_min_polish(pos, r, time_budget, beta_start=50, beta_end=500):
    """
    Polish using soft-min objective with increasing beta.
    """
    t_start = time.time()
    cur_pos = pos.copy()
    cur_r = r
    
    n_betas = 8
    betas = np.geomspace(beta_start, beta_end, n_betas)
    
    for idx, beta in enumerate(betas):
        if time.time() - t_start > time_budget:
            break
        
        remaining = time_budget - (time.time() - t_start)
        if remaining < 2:
            break
        
        def obj_func(x):
            p = x.reshape(-1, 2)
            val, _ = _soft_min_obj_and_grad(p, cur_r, beta)
            return val
        
        def grad_func(x):
            p = x.reshape(-1, 2)
            _, g = _soft_min_obj_and_grad(p, cur_r, beta)
            return g.flatten()
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        res = minimize(
            obj_func,
            cur_pos.flatten(),
            jac=grad_func,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if res.success or res.fun < 1e-6:
            new_pos = res.x.reshape(-1, 2)
            new_r = _compute_r(new_pos)
            if new_r >= cur_r - 1e-8:
                cur_pos = new_pos
                cur_r = max(cur_r, new_r)
    
    return cur_pos, cur_r


def _final_tighten(pos, r, time_budget):
    """
    Final high-precision tightening using binary search on radius with L-BFGS.
    """
    t_start = time.time()
    cur_pos = pos.copy()
    cur_r = r
    
    lo = cur_r
    hi = cur_r * 1.02
    
    for _ in range(5):
        if time.time() - t_start > time_budget:
            break
        
        mid = (lo + hi) / 2
        remaining = time_budget - (time.time() - t_start)
        if remaining < 1:
            break
        
        def obj_func(x):
            p = x.reshape(-1, 2)
            return _violation_exact(p, mid)
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        res = minimize(
            obj_func,
            cur_pos.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        
        if res.success and res.fun < 1e-10:
            new_pos = res.x.reshape(-1, 2)
            new_r = _compute_r(new_pos)
            if new_r >= mid - 1e-9:
                lo = mid
                cur_pos = new_pos
                cur_r = new_r
            else:
                hi = mid
        else:
            hi = mid
    
    return cur_pos, cur_r


def construct():
    t_start = time.time()
    deadline = 145.0
    
    # Try multiple initializations
    inits = [
        ('grid', _grid_init()),
        ('hex', _hex_init()),
        ('rand1', _random_init(42)),
        ('rand2', _random_init(137)),
    ]
    
    best_pos = None
    best_r = 0.0
    
    for name, init_pos in inits:
        if time.time() - t_start > deadline - 40:
            break
        
        r_init = _compute_r(init_pos)
        if r_init > best_r:
            best_r = r_init
            best_pos = init_pos.copy()
    
    if best_pos is None:
        best_pos = _grid_init()
        best_r = _compute_r(best_pos)
    
    # Simulated annealing phase
    sa_budget = min(50, deadline - 40 - (time.time() - t_start))
    if sa_budget > 5:
        pos, r = _simulated_annealing(best_pos, sa_budget, seed=99)
        if r > best_r:
            best_r = r
            best_pos = pos
    
    # Soft-min polish phase
    polish_budget = min(30, deadline - 15 - (time.time() - t_start))
    if polish_budget > 3:
        pos, r = _soft_min_polish(best_pos, best_r, polish_budget)
        if r > best_r:
            best_r = r
            best_pos = pos
    
    # Final tighten phase
    tighten_budget = max(2, deadline - 3 - (time.time() - t_start))
    if tighten_budget > 1:
        pos, r = _final_tighten(best_pos, best_r, tighten_budget)
        if r > best_r:
            best_r = r
            best_pos = pos
    
    # Safety: ensure validity
    lim = 0.5 - best_r
    best_pos = np.clip(best_pos, -lim, lim)
    if not np.all(np.isfinite(best_pos)):
        return _grid_init()
    
    return best_pos
# EVOLVE-BLOCK-END
