import numpy as np
import time
import scipy.optimize

N = 50
_T0 = time.time()
_DEADLINE = 118.0

def _score(pts):
    """Radius for a centre array: min(pair_dist/2, wall_dist)."""
    d = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_pair * 0.5, min_wall)

def _softmin_obj_grad(pts, beta):
    """
    Returns (objective_value, gradient) for the negative soft-min radius.
    The soft-min is over all pairwise half-distances and wall distances.
    """
    iu_i, iu_j = np.triu_indices(N, k=1)
    diffs = pts[iu_i] - pts[iu_j]
    dists_pair = np.sqrt(np.sum(diffs * diffs, axis=1))
    pair_vals = dists_pair * 0.5

    wall_vals = (0.5 - np.abs(pts)).flatten()

    vals = np.concatenate([pair_vals, wall_vals])

    max_val = np.max(vals)
    shifted = vals - max_val
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    probs = exp_vals / sum_exp

    softmin_val = max_val - np.log(sum_exp) / beta
    obj = -softmin_val

    grad = np.zeros_like(pts)

    # Pair contribution
    safe_d = np.where(dists_pair < 1e-12, 1e-12, dists_pair)
    # d(pair_val)/d(pts_i) = diffs / (2 * dists)
    # d(pair_val)/d(pts_j) = -diffs / (2 * dists)
    pair_probs = probs[:len(iu_i)]
    coeffs = pair_probs / (2.0 * safe_d)
    pair_grad = coeffs[:, None] * diffs
    np.add.at(grad, iu_i, pair_grad)
    np.add.at(grad, iu_j, -pair_grad)

    # Wall contribution
    # val = 0.5 - |x_k|
    # d(val)/d(x_k) = -sign(x_k)
    wall_probs = probs[len(iu_i):]
    wall_grad = -np.sign(pts) * wall_probs.reshape(N, 3)
    grad += wall_grad

    # Objective is -softmin, so gradient is -d(softmin)/dx
    # d(softmin)/dx = sum_m (-probs_m) * (d vals_m / dx)
    # d(obj)/dx = - d(softmin)/dx = sum_m probs_m * (d vals_m / dx)
    # So the computed grad is correct for the objective.
    return obj, grad

def _optimize(pts, beta_init=10.0, beta_final=2000.0, n_steps=100):
    """
    Continuation method with L-BFGS-B on the soft-min objective.
    """
    cur = pts.copy()
    cur_obj = _softmin_obj_grad(cur, beta_init)[0]
    best_pts = cur.copy()
    best_r = _score(cur)

    for i in range(n_steps):
        if time.time() > _DEADLINE:
            break
        # Geometric progression of beta
        beta = beta_init * (beta_final / beta_init) ** (i / n_steps)

        res = scipy.optimize.minimize(
            lambda x: _softmin_obj_grad(x, beta)[0],
            cur,
            jac=lambda x: _softmin_obj_grad(x, beta)[1],
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 10, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
        )

        if res.success:
            cur = np.clip(res.x, -0.5 + 1e-10, 0.5 - 1e-10)
            r = _score(cur)
            if r > best_r:
                best_r = r
                best_pts = cur.copy()

    return best_pts, best_r

def _anneal(pts, T0, T1, steps, rng):
    """Simulated annealing to escape local minima."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r

    for i in range(steps):
        if time.time() > _DEADLINE:
            break
        T = T0 * (T1 / T0) ** (i / steps)
        kick = 0.005 * (T / T0 + 0.05)
        # Move a random subset
        n_move = max(1, N // 5)
        idx = rng.choice(N, size=n_move, replace=False)
        cand = cur.copy()
        cand[idx] += rng.normal(0.0, kick, size=(n_move, 3))
        cand = np.clip(cand, -0.5 + 1e-10, 0.5 - 1e-10)

        cand_r = _score(cand)
        if cand_r > cur_r or (cand_r < cur_r and rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-12))):
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()

    return best, best_r

def _bcc_start():
    """
    Generate a Body-Centered Cubic (BCC) lattice start, clipped to the cube.
    BCC has higher packing density than simple cubic.
    """
    # BCC unit cell side length a.
    # To fit 50 spheres, we need a 3x3x3 block of BCC cells?
    # 3x3x3 = 27 cells. Each has 1 center + 8 corners (shared).
    # In a 3x3x3 block of cells, total spheres = 27 * 1 (body) + corners.
    # This is complex to handle boundaries.
    # Let's just generate a dense BCC grid and keep the best 50.
    # Or simpler: use the 4x4x4 grid as base, but perturb slightly towards BCC?
    # Let's stick to the 4x4x4 grid but with a specific jittered pattern that helps.
    # Actually, let's create a start that is a 4x4x4 grid but with the "corners" of the small cells pushed out?
    # No, let's just use the grid and a random start, and rely on the optimizer.
    pass

def construct():
    global _T0
    rng = np.random.default_rng(42)

    # Start 1: 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    start_grid = grid[:N].copy()

    starts = [start_grid]

    # Start 2: Jittered grid
    start_jitter = start_grid + rng.normal(0.0, 0.02, size=start_grid.shape)
    start_jitter = np.clip(start_jitter, -0.5 + 1e-6, 0.5 - 1e-6)
    starts.append(start_jitter)

    # Start 3: Random
    start_rand = rng.uniform(-0.45, 0.45, size=(N, 3))
    starts.append(start_rand)

    # Start 4: Another jittered grid
    start_jitter2 = start_grid + rng.normal(0.0, 0.05, size=start_grid.shape)
    start_jitter2 = np.clip(start_jitter2, -0.5 + 1e-6, 0.5 - 1e-6)
    starts.append(start_jitter2)

    best_pts = None
    best_r = -1.0

    for s in starts:
        if time.time() > _DEADLINE:
            break

        # Phase 1: Annealing to escape local minima
        a, ar = _anneal(s, T0=0.02, T1=0.0001, steps=1500, rng=rng)
        if ar > best_r:
            best_r = ar
            best_pts = a.copy()

        if time.time() > _DEADLINE:
            break

        # Phase 2: Polish with soft-min gradient ascent
        p, pr = _optimize(a, beta_init=10.0, beta_final=5000.0, n_steps=150)
        if pr > best_r:
            best_r = pr
            best_pts = p.copy()

    if best_pts is None:
        return start_grid

    # Final check: ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.5 + 1e-12, 0.5 - 1e-12)
    return best_pts
