"""Rewritten solver: smooth softmin max-min + vectorized repulsion annealing + local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _softmin_radius(c, t):
    """Smooth approximation of the true radius via log-sum-exp over all
    constraints (pair/2 and wall). Returns a smooth lower-bound-ish value
    that increases as the packing tightens. t = temperature (small = sharp)."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0                       # (N,N)
    wall = 0.5 - np.abs(c)               # (N,3)
    # gather all constraint values
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    # log-sum-exp for min: min ≈ -logsumexp(-vals/t)*t
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) / t))
    return m - t * np.log(s)


def _smooth_obj(x, t):
    """Objective to MINIMIZE: negative smooth radius plus a small wall-push."""
    c = x.reshape(N, 3)
    # keep strictly inside with a tiny margin to avoid boundary issues
    r = _softmin_radius(c, t)
    # small quadratic penalty if any coordinate escapes the cube
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _lloyd_step(c, n_iter=50):
    """Voronoi-Lloyd equalization: iteratively move each point to the
    centroid of its Voronoi cell (clipped to cube). This equalizes local
    density, which helps maximize the minimum pairwise distance."""
    from scipy.spatial import Voronoi
    c = c.copy()
    for _ in range(n_iter):
        try:
            vor = Voronoi(c)
        except Exception:
            break
        new_c = c.copy()
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            verts = vor.vertices[region]
            # Compute centroid of the Voronoi cell vertices
            centroid = verts.mean(axis=0)
            # Clip to cube
            centroid = np.clip(centroid, -0.49, 0.49)
            # Move point toward centroid
            new_c[i] = c[i] + 0.5 * (centroid - c[i])
        new_c = np.clip(new_c, -0.49, 0.49)
        # Check for improvement
        if _compute_radius(new_c) >= _compute_radius(c) - 1e-12:
            c = new_c
        else:
            break
    return c


def _lloyd_optimize(c, deadline, max_outer=5):
    """Outer loop: Lloyd equalization + smooth polish, repeated."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    for outer in range(max_outer):
        if time.time() > deadline:
            break
        # Lloyd step
        c_lloyd = _lloyd_step(c, n_iter=30)
        r_lloyd = _compute_radius(c_lloyd)
        if r_lloyd > best_r:
            best_r = r_lloyd
            best = c_lloyd.copy()
        # Smooth polish
        c_pol, r_pol = _smooth_polish(c_lloyd, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best = c_pol.copy()
        c = best.copy()
    return best, best_r


def _repulsion_anneal(c, deadline, steps=None):
    """Vectorized repulsion + wall-attraction gradient descent.
    Different search family from the smooth max-min: a physical relaxation
    that spreads points and pushes them toward walls, then tightens."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    if steps is None:
        steps = 4000
    # anneal temperature of the repulsion kernel and the wall strength
    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)                       # (N,N)
        # repulsive force: for each pair, push apart with strength ~ 1/(d^2)
        # but only meaningful when d is small relative to a scale
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        # force on i from j = (c_i - c_j) * inv_ij  (repulsion)
        diff = c[:, None, :] - c[None, :, :]           # (N,N,3)
        F = np.sum(diff * inv[:, :, None], axis=1)     # (N,3)
        # wall attraction: push toward nearest wall face
        # gradient of wall distance: for coordinate k, if c_ik>0 attract to +0.5
        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            # stronger attraction as the point gets closer to the centre
            # (we want to maximize min wall, so push toward walls)
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))
        # anneal: early = strong repulsion to spread, late = strong wall push
        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)
    return c


def _smooth_polish(c, deadline):
    """Polish a configuration with the smooth softmin max-min via L-BFGS-B,
    annealing the temperature down for a sharp min."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)
    for t in (0.05, 0.02, 0.008, 0.003, 0.001, 0.0005):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11,
                         'maxls': 50},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            x0 = c.flatten()
        except Exception:
            pass
    return best, best_r


def _metropolis_mc(c, deadline, n_iter=5000, t0=0.01, t1=0.001):
    """Metropolis Monte Carlo: random single-point moves with simulated
    annealing acceptance. Helps escape local optima that greedy local
    search gets stuck in."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_c = c.copy()
    best_r = _compute_radius(c)
    cur_r = best_r
    rng = np.random.default_rng(123)
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        # Anneal temperature
        frac = it / max(n_iter - 1, 1)
        t = t0 * (t1 / t0) ** frac
        
        # Pick a random point to move
        i = rng.integers(N)
        # Random displacement
        disp = rng.normal(scale=t, size=3)
        trial = c.copy()
        trial[i] = np.clip(c[i] + disp, -0.499, 0.499)
        r_trial = _compute_radius(trial)
        
        # Metropolis acceptance
        if r_trial >= cur_r:
            c = trial
            cur_r = r_trial
            if r_trial > best_r:
                best_r = r_trial
                best_c = c.copy()
        else:
            # Accept with probability exp((r_trial - cur_r) / (t * 0.01))
            # Scale the energy difference appropriately
            delta = (r_trial - cur_r) / (t * 0.01)
            if np.log(rng.random()) < delta:
                c = trial
                cur_r = r_trial
    
    return best_c, best_r


def _local_search(c, deadline, iters=3000):
    """Local search: perturb the closest pair and nearest wall, accept if radius improves."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # Find the binding constraints
        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]
        
        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]
        
        # Determine which constraint is binding
        if min_pair_dist / 2.0 < min_wall_val:
            # Binding pair
            i, j = min_pair_idx
            # Perturb these two points apart
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = rng.normal(size=3)
                norm = np.linalg.norm(direction)
                direction /= norm
            
            # Try moving i and j apart
            for step in (0.005, 0.002, 0.001, 0.0005):
                trial = c.copy()
                trial[i] += step * direction
                trial[j] -= step * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    break
        else:
            # Binding wall
            i, k = min_wall_idx
            # Push point i away from the nearest wall
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0
            direction = np.zeros(3)
            direction[k] = sgn
            
            for step in (0.005, 0.002, 0.001, 0.0005):
                trial = c.copy()
                trial[i] += step * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    break
        
        # Occasionally do a random perturbation to escape local optima
        if it % 100 == 0 and it > 0:
            for step in (0.01, 0.005, 0.002):
                trial = c.copy()
                trial += rng.normal(scale=step, size=(N, 3))
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    break
    
    return best_c, best_r


def _seed_fcc():
    """FCC lattice supercell, then farthest-point sample to N points."""
    pts = []
    a = 1.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts = pts - pts.mean(axis=0)
    me = np.max(np.abs(pts))
    if me > 0:
        pts = pts / me * 0.45
    # farthest point sampling
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = [i for i in range(len(pts)) if i != sel[0]]
    while len(sel) < N and rem:
        dmin = np.min(_pair_dist_matrix(pts[sel]), axis=1) if len(sel) > 1 \
            else np.full(len(pts), np.inf)
        # for each remaining, min dist to selected
        if len(sel) > 1:
            dm = np.min(_pair_dist_matrix(pts[sel + rem]), axis=0)
        else:
            dm = np.full(len(rem), np.inf)
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]


def _seed_grid():
    m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N]


def _seed_random(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _seed_shell_core():
    """Place points on a spherical shell (near the cube's inscribed sphere)
    plus a central cluster; gives diverse wall-near / interior mix."""
    rng = np.random.default_rng(7)
    # 35 on a shell of radius ~0.33, 15 in a small inner grid
    shell_n = 35
    core_n = N - shell_n
    # fibonacci sphere for even shell distribution
    idx = np.arange(shell_n)
    phi = np.arccos(1 - 2 * (idx + 0.5) / shell_n)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    R = 0.32
    shell = np.column_stack([
        R * np.sin(phi) * np.cos(theta),
        R * np.sin(phi) * np.sin(theta),
        R * np.cos(phi),
    ])
    shell = np.clip(shell, -0.45, 0.45)
    # inner grid
    ticks = (np.arange(3) + 0.5) / 3 - 0.5
    core = np.array([(x * 0.2, y * 0.2, z * 0.2)
                     for z in ticks for y in ticks for x in ticks])[:core_n]
    return np.vstack([shell, core])


def construct():
    """Maximize the common radius of 50 equal spheres in [-0.5,0.5]^3."""
    start = time.time()
    deadline = start + (150.0 - DEADLINE_MARGIN)

    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_shell_core(),
        _seed_random(1),
        _seed_random(2),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: repulsion annealing (cheap, parallelizable)
        c_an = _repulsion_anneal(s, deadline, steps=2000)
        r_an = _compute_radius(c_an)

        # Stage B: Lloyd equalization + smooth polish
        c_lloyd, r_lloyd = _lloyd_optimize(c_an, deadline, max_outer=3)
        r_lloyd = max(r_lloyd, r_an)
        c_lloyd = c_an if r_an > r_lloyd else c_lloyd

        # Stage C: smooth softmin polish
        c_pol, r_pol = _smooth_polish(c_lloyd, deadline)

        # Stage D: local search on the best
        c_cur = c_pol if r_pol >= r_lloyd else c_lloyd
        r_cur = max(r_pol, r_lloyd)
        
        if time.time() < deadline:
            c_ls, r_ls = _local_search(c_cur, deadline, iters=1500)
            r_cur = max(r_cur, r_ls)
            c_cur = c_ls if r_ls >= r_cur else c_cur

        if r_cur > best_r:
            best_r = r_cur
            best_c = c_cur.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and time.time() < deadline:
        # Metropolis MC to escape local optima
        c_mc, r_mc = _metropolis_mc(best_c, deadline, n_iter=3000)
        if r_mc > best_r:
            best_r = r_mc
            best_c = c_mc
        
        # Final smooth polish
        if time.time() < deadline:
            c_pol, r_pol = _smooth_polish(best_c, deadline)
            if r_pol > best_r:
                best_r = r_pol
                best_c = c_pol
        
        # Final local search
        if time.time() < deadline:
            c_ls, r_ls = _local_search(best_c, deadline, iters=1500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END
