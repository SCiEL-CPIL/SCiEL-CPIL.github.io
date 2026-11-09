"""Fifty equal circles in a square – optimised packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 145.0  # safety margin before 150s kill


def _score_from_centres(c):
    """Compute the radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    # Fast pairwise distance computation using matrix operations
    # Compute all pairwise squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    # Set diagonal to inf to avoid self-distance
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_sq = sq_dists.min()
    min_pair = np.sqrt(min_pair_sq) if min_pair_sq > 0 else 0.0
    
    # Distance to walls
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    r = min(min_pair / 2.0, min_wall)
    return r


def _direct_radius_maximize(c0, max_iter=500):
    """Directly maximize the minimum radius using SLSQP with hard constraints.
    
    Variables: x = [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y, r]
    Objective: maximize r
    Constraints:
      - For each pair (i,j): |c_i - c_j| >= 2r
      - For each point i and each wall: distance to wall >= r
    """
    c = c0.reshape(N, 2)
    r0 = _score_from_centres(c)
    x0 = np.concatenate([c.ravel(), [r0]])
    
    n_vars = 2 * N + 1
    
    def objective(x):
        return -x[-1]  # minimize -r
    
    def grad_objective(x):
        g = np.zeros(n_vars)
        g[-1] = -1.0
        return g
    
    # Generate constraints: pairwise distances and wall distances
    # Pairwise: |c_i - c_j|^2 >= (2r)^2 for all i < j
    # Wall: c_i[k] >= -HALF + r, c_i[k] <= HALF - r for all i, k
    
    cons = []
    
    # Pairwise distance constraints (only active pairs near the minimum)
    # To keep the problem tractable, we use inequality constraints
    # |c_i - c_j| >= 2r  =>  (c_i - c_j)^2 >= (2r)^2
    # This is nonlinear. We'll use a simplified approach:
    # For SLSQP, we need differentiable constraints.
    # Use: (c_i - c_j)^2 - (2r)^2 >= 0
    # But this is not convex. Instead, we'll use a penalty-free approach
    # by directly constraining the minimum distance.
    
    # Alternative: use the fact that at optimum, the binding constraints
    # are the ones achieving the minimum. We can use a sequential approach.
    
    # For now, use a simpler formulation with bounds and a penalty-free
    # approach: we'll use multiple SLSQP calls with different active sets.
    
    # Actually, let's use a different approach: 
    # We know the current best r. We can try to increase it by solving
    # a feasibility problem: can we place all circles with radius r + eps?
    
    # Simpler: just use Nelder-Mead on the direct (non-smooth) objective
    # since it handles kinks better.
    from scipy.optimize import minimize
    
    def direct_score(x):
        cc = x[:2*N].reshape(N, 2)
        return _score_from_centres(cc)
    
    def neg_direct_score(x):
        return -direct_score(x)
    
    bounds = [(-HALF + 1e-12, HALF - 1e-12)] * (2 * N)
    
    try:
        res = minimize(
            neg_direct_score,
            x0[:2*N],
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        c_new = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
        return c_new, _score_from_centres(c_new)
    except Exception:
        return c, r0


def _smooth_score(c_flat, beta=50.0):
    """Smooth surrogate for the radius score using log-sum-exp."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    d = np.sqrt(sq_dists)
    
    if d.size == 0:
        return 0.0
    
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    
    d_half = d / 2.0
    min_d_half = d_half.min()
    exp_d = np.exp(-beta * (d_half - min_d_half))
    soft_min_pair = min_d_half - np.log(exp_d.sum()) / beta
    
    min_wall = wall_dists.min()
    exp_w = np.exp(-beta * (wall_dists - min_wall))
    soft_min_wall = min_wall - np.log(exp_w.sum()) / beta
    
    all_dists = np.concatenate([soft_min_pair.reshape(1), soft_min_wall.reshape(1)])
    min_all = all_dists.min()
    exp_all = np.exp(-beta * (all_dists - min_all))
    r = min_all - np.log(exp_all.sum()) / beta
    
    return r


def _negative_score(c_flat, beta=50.0):
    """Objective to minimise (negative smooth radius score)."""
    return -_smooth_score(c_flat, beta)


def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_init():
    """Hexagonal packing initial layout."""
    # Hexagonal close packing for better initial configuration
    rows = int(np.ceil(N / 8))
    cols = 8
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= N:
                break
            x = (j - (cols - 1) / 2) * 0.22
            y = (i - (rows - 1) / 2) * 0.2
            if i % 2 == 1:
                x += 0.11
            points.append([x, y])
    return np.array(points[:N])


def _random_init(rng):
    """Random initial layout."""
    return rng.uniform(-HALF, HALF, size=(N, 2))


def construct():
    t0 = time.time()
    
    # Try multiple initial configurations
    inits = [_grid_init(), _hex_init()]
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    rng = np.random.default_rng(42)

    # Strategy 1: Direct radius maximization using Nelder-Mead on the non-smooth objective
    # Nelder-Mead handles the kinks in min-distance better than gradient methods
    for init in inits:
        s = _score_from_centres(init)
        if s > best_score:
            best_score = s
            best_centres = init

    n_starts = 15
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 40:
            break
        
        if i == 0:
            x0 = best_centres.copy()
        elif i == 1:
            x0 = _hex_init()
        elif i == 2:
            x0 = _grid_init()
        else:
            x0 = _random_init(rng)
            x0 = x0 + rng.normal(0, 0.03, size=x0.shape)
            x0 = np.clip(x0, -HALF + 1e-9, HALF - 1e-9)

        try:
            # First: smooth optimization to get close
            for beta in [30.0, 80.0]:
                res = minimize(
                    lambda cf, b=beta: _negative_score(cf, b), x0.ravel(),
                    method='L-BFGS-B',
                    bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                )
                c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                s = _score_from_centres(c)
                if s > best_score:
                    best_score = s
                    best_centres = c
                x0 = c.copy()
            
            # Then: Nelder-Mead polish on the direct objective
            c_nm, s_nm = _direct_radius_maximize(x0, max_iter=300)
            if s_nm > best_score:
                best_score = s_nm
                best_centres = c_nm
        except Exception:
            pass

    # Strategy 2: Simulated annealing with targeted local moves
    # This explores the global landscape more effectively than DE for this problem
    if time.time() - t0 < DEADLINE - 25:
        def _sa_optimize(c0, t_max=0.01, t_min=1e-5, n_iter=3000, rng_sa=None):
            """Simulated annealing with local refinement."""
            if rng_sa is None:
                rng_sa = np.random.default_rng()
            
            c = c0.copy()
            current_score = _score_from_centres(c)
            best_c = c.copy()
            best_s = current_score
            
            for it in range(n_iter):
                if time.time() - t0 > DEADLINE - 20:
                    break
                
                # Temperature schedule
                T = t_max * (t_min / t_max) ** (it / n_iter)
                
                # Generate move: perturb a random subset of points
                n_move = rng_sa.integers(1, 8)  # Larger moves for better exploration
                idx = rng_sa.choice(N, size=n_move, replace=False)
                
                c_new = c.copy()
                # Scale perturbation with temperature - larger scale for better exploration
                scale = 0.03 + 0.12 * (T / t_max)
                delta = rng_sa.normal(0, scale, size=(n_move, 2))
                c_new[idx] += delta
                c_new = np.clip(c_new, -HALF + 1e-9, HALF - 1e-9)
                
                new_score = _score_from_centres(c_new)
                delta_s = new_score - current_score
                
                # Accept if better or with probability exp(-delta_s/T) for worse
                if delta_s > 0 or (T > 0 and rng_sa.random() < np.exp(max(-100, delta_s / T))):
                    c = c_new
                    current_score = new_score
                    if current_score > best_s:
                        best_s = current_score
                        best_c = c.copy()
                
                # Every 20 iterations, do a local refinement on best with adaptive beta
                if it % 20 == 0 and it > 0:
                    try:
                        for beta in [50.0, 100.0, 200.0]:
                            res = minimize(
                                lambda cf, b=beta: _negative_score(cf, b), best_c.ravel(),
                                method='L-BFGS-B',
                                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                                options={'maxiter': 150, 'ftol': 1e-13, 'iprint': 0}
                            )
                            c_refined = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                            s_refined = _score_from_centres(c_refined)
                            if s_refined > best_s:
                                best_s = s_refined
                                best_c = c_refined
                                c = best_c.copy()
                                current_score = best_s
                    except Exception:
                        pass
            
            return best_c, best_s
        
        sa_c, sa_s = _sa_optimize(best_centres, t_max=0.03, t_min=1e-6, n_iter=8000, rng_sa=rng)
        if sa_s > best_score:
            best_score = sa_s
            best_centres = sa_c

    # Strategy 3: Nelder-Mead polish with small perturbations (handles non-smooth objective)
    if time.time() - t0 < DEADLINE - 10:
        try:
            for pert_scale in [0.0, 0.0005, 0.001, 0.002, 0.003]:
                if time.time() - t0 > DEADLINE - 6:
                    break
                x0 = best_centres.copy()
                if pert_scale > 0:
                    x0 = x0 + rng.normal(0, pert_scale, size=x0.shape)
                    x0 = np.clip(x0, -HALF + 1e-9, HALF - 1e-9)
                
                # Nelder-Mead on direct objective
                c_nm, s_nm = _direct_radius_maximize(x0, max_iter=500)
                if s_nm > best_score:
                    best_score = s_nm
                    best_centres = c_nm
        except Exception:
            pass

    # Strategy 4: Final Nelder-Mead polish with very tight tolerances
    if time.time() - t0 < DEADLINE - 5:
        try:
            from scipy.optimize import minimize as _minimize
            def neg_direct(x):
                cc = x.reshape(N, 2)
                return -_score_from_centres(cc)
            
            res = _minimize(
                neg_direct,
                best_centres.ravel(),
                method='Nelder-Mead',
                bounds=[(-HALF + 1e-12, HALF - 1e-12)] * (2 * N),
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            c_final = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
            s_final = _score_from_centres(c_final)
            if s_final > best_score:
                best_score = s_final
                best_centres = c_final
        except Exception:
            pass

    # Ensure valid output
    best_centres = np.clip(best_centres, -HALF + 1e-12, HALF - 1e-12)
    return best_centres
