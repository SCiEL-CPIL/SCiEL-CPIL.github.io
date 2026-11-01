"""Rewritten solver: fixed softmin gradient + Lloyd equalization + multi-stage polish.
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


def _softmin_radius_and_grad(c, t):
    """Smooth approximation of the true radius via log-sum-exp over all
    constraints (pair/2 and wall). Each pair counted ONCE.
    Returns (value, gradient w.r.t. c)."""
    diff = c[:, None, :] - c[None, :, :]          # (N,N,3)
    d = np.sqrt(np.sum(diff * diff, axis=-1))      # (N,N)
    np.fill_diagonal(d, np.inf)
    # Only upper triangle to count each pair once
    iu = np.triu_indices(N, k=1)
    pair_d = d[iu] / 2.0                          # (M,)
    wall = 0.5 - np.abs(c)                         # (N,3)

    vals = np.concatenate([pair_d, wall.ravel()])
    vals = np.maximum(vals, 1e-12)

    m = np.min(vals)
    exps = np.exp(-(vals - m) / t)
    s = np.sum(exps)
    r = m - t * np.log(s)

    # Gradient: dr/dc
    # For pair (i,j): d(pair_d)/dc_i = (c_i - c_j)/d_ij, d(pair_d)/dc_j = (c_j - c_i)/d_ij
    # For wall (i,k): d(wall)/dc_ik = -sign(c_ik)
    grad = np.zeros_like(c)
    # weights
    w = exps / s  # (M+N*3,)
    w_pair = w[:len(pair_d)]
    w_wall = w[len(pair_d):].reshape(N, 3)

    # pair contributions
    d_full = d.copy()
    np.fill_diagonal(d_full, 1.0)  # avoid div by 0 (won't be used on diag)
    # For each pair (i,j) in upper triangle:
    # grad_i += w_pair * (c_i - c_j) / d_ij
    # grad_j += w_pair * (c_j - c_i) / d_ij
    ii, jj = iu
    for idx in range(len(ii)):
        i, j = ii[idx], jj[idx]
        wv = w_pair[idx]
        dij = d_full[i, j]
        if dij < 1e-12:
            continue
        dv = (c[i] - c[j]) / dij
        grad[i] += wv * dv
        grad[j] -= wv * dv

    # wall contributions
    sgn = np.sign(c)
    sgn[sgn == 0] = 1.0
    grad -= w_wall * sgn

    return r, grad


def _smooth_obj(x, t):
    """Objective to MINIMIZE: negative smooth radius plus wall penalty."""
    c = x.reshape(N, 3)
    r, _ = _softmin_radius_and_grad(c, t)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _smooth_obj_grad(x, t):
    """Objective and gradient for L-BFGS-B."""
    c = x.reshape(N, 3)
    r, g = _softmin_radius_and_grad(c, t)
    pen = np.maximum(0.0, np.abs(c) - 0.5 + 1e-6)
    pen_grad = 2e3 * pen * np.sign(c)
    pen_grad[np.abs(c) <= 0.5 - 1e-6] = 0.0
    return -r, -g + pen_grad


def _lloyd_step(c, deadline):
    """One Lloyd iteration: move each point toward the centroid of its Voronoi cell
    (approximated by nearest-neighbor barycenter with radius limit)."""
    d = _pair_dist_matrix(c)
    # For each point, find its 4 nearest neighbors and move toward their centroid
    # but only if the move doesn't violate the current radius
    r_cur = _compute_radius(c)
    c_new = c.copy()
    for i in range(N):
        # 4 nearest neighbors
        order = np.argsort(d[i])
        nn = order[:4]
        centroid = c[nn].mean(axis=0)
        # Move toward centroid by a small fraction
        move = (centroid - c[i]) * 0.15
        c_try = c_new.copy()
        c_try[i] = c[i] + move
        c_try = np.clip(c_try, -0.499, 0.499)
        r_try = _compute_radius(c_try)
        if r_try >= r_cur - 1e-12:
            c_new[i] = c_try[i]
    return c_new


def _lloyd_optimize(c, deadline, iters=200):
    """Run Lloyd iterations, keeping the best configuration."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best_c = c.copy()
    best_r = _compute_radius(c)
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _lloyd_step(c, deadline)
        c = np.clip(c, -0.499, 0.499)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    return best_c, best_r


def _repulsion_anneal(c, deadline, steps=None):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    if steps is None:
        steps = 3000
    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)
        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))
        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)
    return c


def _smooth_polish(c, deadline, temps=None):
    """Polish with L-BFGS-B using the corrected softmin gradient."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)
    if temps is None:
        temps = (0.05, 0.02, 0.008, 0.003, 0.001)
    for t in temps:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12,
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


def _local_search(c, deadline, iters=1500):
    """Local search: perturb binding constraints, accept if radius improves."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()

    for it in range(iters):
        if time.time() > deadline:
            break

        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]

        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]

        improved = False
        if min_pair_dist / 2.0 < min_wall_val:
            i, j = min_pair_idx
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0, 0])
            else:
                direction = direction / norm
            for step in (0.005, 0.01, 0.02, 0.003, 0.008, 0.015):
                c_try = c.copy()
                c_try[i] += step * direction
                c_try[j] -= step * direction
                c_try = np.clip(c_try, -0.499, 0.499)
                r_try = _compute_radius(c_try)
                if r_try > best_r + 1e-13:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break
        else:
            i, k = min_wall_idx
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0
            for step in (0.005, 0.01, 0.02, 0.003, 0.008, 0.015):
                c_try = c.copy()
                c_try[i, k] += step * sgn
                c_try = np.clip(c_try, -0.499, 0.499)
                r_try = _compute_radius(c_try)
                if r_try > best_r + 1e-13:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break

        # Random perturbation to escape local minima
        if not improved and it % 80 == 40:
            idx = np.random.randint(N)
            c_try = c.copy()
            c_try[idx] += np.random.normal(0, 0.015, size=3)
            c_try = np.clip(c_try, -0.499, 0.499)
            r_try = _compute_radius(c_try)
            if r_try > best_r + 1e-13:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()

    return best_c, best_r


def _metropolis_mc(c, deadline, steps=5000, seed=42):
    """Metropolis Monte Carlo with annealing."""
    rng = np.random.default_rng(seed)
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    r0 = max(best_r, 1e-6)
    T0 = 0.02 * r0
    T_final = 1e-6 * r0

    for it in range(steps):
        if time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        T = T0 * (T_final / T0) ** frac
        if T < T_final:
            T = T_final

        move_type = rng.random()
        if move_type < 0.8:
            i = rng.integers(N)
            step_size = 0.05 * r0 * (1.0 - 0.5 * frac)
            delta = rng.normal(0, step_size, size=3)
            c_try = c.copy()
            c_try[i] += delta
            c_try = np.clip(c_try, -0.499, 0.499)
        else:
            i, j = rng.choice(N, size=2, replace=False)
            c_try = c.copy()
            c_try[i], c_try[j] = c_try[j].copy(), c_try[i].copy()
            c_try = np.clip(c_try, -0.499, 0.499)

        r_try = _compute_radius(c_try)
        delta_E = -(r_try - best_r)
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
            c = c_try.copy()
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()

    return best_c, best_r


def _seed_fcc():
    """FCC lattice supercell, farthest-point sample to N points."""
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
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = [i for i in range(len(pts)) if i != sel[0]]
    while len(sel) < N and rem:
        if len(sel) > 1:
            dm = np.min(_pair_dist_matrix(pts[sel + rem]), axis=0)
        else:
            dm = np.full(len(rem), np.inf)
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]


def _seed_bcc():
    """BCC lattice supercell, farthest-point sample to N points."""
    pts = []
    a = 1.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts = pts - pts.mean(axis=0)
    me = np.max(np.abs(pts))
    if me > 0:
        pts = pts / me * 0.45
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = [i for i in range(len(pts)) if i != sel[0]]
    while len(sel) < N and rem:
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
    rng = np.random.default_rng(7)
    shell_n = 35
    core_n = N - shell_n
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
        _seed_bcc(),
        _seed_grid(),
        _seed_shell_core(),
        _seed_random(1),
        _seed_random(2),
        _seed_random(3),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: repulsion annealing
        c_an = _repulsion_anneal(s, deadline, steps=2000)
        r_an = _compute_radius(c_an)

        # Stage B: Lloyd equalization
        c_lloyd, r_lloyd = _lloyd_optimize(c_an, deadline, iters=100)

        # Stage C: smooth softmin polish (corrected gradient)
        c_best_stage = c_lloyd if r_lloyd >= r_an else c_an
        r_best_stage = max(r_lloyd, r_an)
        c_pol, r_pol = _smooth_polish(c_best_stage, deadline)

        # Stage D: local search
        c_ls = c_pol if r_pol >= r_best_stage else c_best_stage
        r_ls = max(r_pol, r_best_stage)
        if time.time() < deadline:
            c_ls, r_ls = _local_search(c_ls, deadline, iters=600)

        # Stage E: Metropolis MC
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(c_ls, deadline, steps=2500, seed=si + 100)
            if r_mc > r_ls:
                r_ls = r_mc
                c_ls = c_mc
                if time.time() < deadline:
                    c_ls, r_ls = _local_search(c_ls, deadline, iters=300)

        r = max(r_an, r_lloyd, r_pol, r_ls)
        c = c_ls if r_ls >= max(r_an, r_lloyd, r_pol) else \
            (c_pol if r_pol >= max(r_an, r_lloyd) else \
             (c_lloyd if r_lloyd >= r_an else c_an))
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # Final polish on global best
    if best_c is not None and time.time() < deadline:
        # Extra smooth polish with tighter temps
        c_pol, r_pol = _smooth_polish(best_c, deadline,
                                       temps=(0.003, 0.001, 0.0005, 0.0002))
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol

        # Final local search
        if time.time() < deadline:
            c_ls, r_ls = _local_search(best_c, deadline, iters=800)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls

        # Final MC + local search
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(best_c, deadline, steps=3000, seed=999)
            if r_mc > best_r:
                best_r = r_mc
                best_c = c_mc
            if time.time() < deadline:
                c_ls, r_ls = _local_search(best_c, deadline, iters=500)
                if r_ls > best_r:
                    best_r = r_ls
                    best_c = c_ls

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END
