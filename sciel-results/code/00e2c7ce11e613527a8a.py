"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 140.0
_start_time = time.time()

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _pair_dists(centers):
    """Return (i_idx, j_idx, dists) for i<j pairs, vectorized."""
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    diff = centers[i_idx] - centers[j_idx]
    d = np.sqrt(np.sum(diff**2, axis=1))
    return i_idx, j_idx, d

def _wall_dists(centers):
    """Distance from each center to the boundary of [-0.5,0.5]^2."""
    return 0.5 - np.max(np.abs(centers), axis=1)

def _compute_radius(centers):
    """Exact score: largest common radius."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-9):
        return 0.0
    _, _, d = _pair_dists(centers)
    if len(d) == 0:
        return 0.0
    min_pair = np.min(d) / 2.0
    min_wall = np.min(_wall_dists(centers))
    return min(min_pair, min_wall)

def _softmin_obj_and_grad(centers, beta):
    """
    Objective f = -softmin(D) where D = [pair_dists/2, wall_dists].
    softmin(D) = -1/beta * log(sum(exp(-beta*D)))  (approximates min(D))
    We minimize f = -softmin(D) = 1/beta * log(sum(exp(-beta*D))).
    Returns (f, grad_f) where grad_f is the gradient w.r.t. centers.
    """
    i_idx, j_idx, d = _pair_dists(centers)
    w = _wall_dists(centers)
    D = np.concatenate([d / 2.0, w])
    n_pairs = len(d)

    # stable log-sum-exp
    D_max = np.max(D)
    exp_terms = np.exp(-beta * (D - D_max))
    S = np.sum(exp_terms)
    f = D_max - (1.0 / beta) * np.log(S)
    # f = -softmin(D). We want to MINIMIZE f, i.e. MAXIMIZE softmin(D) ~ min(D) ~ radius.
    # grad f = - sum_k w_k * grad D_k  where w_k = exp(-beta D_k)/S
    weights = exp_terms / S
    w_pairs = weights[:n_pairs]
    w_walls = weights[n_pairs:]

    grad = np.zeros_like(centers)

    # Wall gradient: D_i = 0.5 - max(|x_i|,|y_i|)
    # grad w.r.t. x_i = -sign(x_i) if |x_i|>|y_i|, 0 if |y_i|>|x_i|, -0.5 sign if equal
    ax = np.abs(centers[:, 0])
    ay = np.abs(centers[:, 1])
    gx = np.where(ax > ay, -np.sign(centers[:, 0]),
                  np.where(ay > ax, 0.0, -0.5 * np.sign(centers[:, 0])))
    gy = np.where(ay > ax, -np.sign(centers[:, 1]),
                  np.where(ax > ay, 0.0, -0.5 * np.sign(centers[:, 1])))
    grad[:, 0] += w_walls * gx
    grad[:, 1] += w_walls * gy

    # Pair gradient: D_ij = |c_i - c_j| / 2
    # grad_{c_i} D_ij = (c_i - c_j) / (2 * |c_i - c_j|)
    safe_d = np.where(d < 1e-12, 1e-12, d)
    unit = (centers[i_idx] - centers[j_idx]) / (2.0 * safe_d)[:, None]
    wv = (w_pairs / 1.0)[:, None] * unit
    np.add.at(grad, i_idx, wv)
    np.add.at(grad, j_idx, -wv)

    # f = -softmin, grad f = -sum(w_k grad D_k). We computed sum(w_k grad D_k) above.
    return f, -grad

def _anneal_optimize(centers, betas, steps_per_beta, base_lr):
    """Anneal soft-min temperature from large (smooth) to small (sharp)."""
    centers = centers.copy()
    for beta in betas:
        if _time_left() < 5.0:
            break
        lr = base_lr
        for it in range(steps_per_beta):
            if _time_left() < 5.0:
                break
            f, g = _softmin_obj_and_grad(centers, beta)
            if not np.all(np.isfinite(g)):
                break
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-12:
                break
            g = g / gnorm
            # backtracking line search
            step = lr
            improved = False
            cur_r = _compute_radius(centers)
            for _ in range(8):
                cand = centers + step * g
                cand = np.clip(cand, -0.5, 0.5)
                if _compute_radius(cand) > cur_r + 1e-10:
                    centers = cand
                    improved = True
                    break
                step *= 0.5
            if not improved:
                # try a smaller base lr and continue
                lr *= 0.5
                if lr < 1e-9:
                    break
    return centers

def _grid_initial(m):
    centers = []
    n_rows = int(np.ceil(N / m))
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(n_rows) + 0.5) / n_rows - 0.5
    for y in ys:
        for x in xs:
            centers.append([x, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    while len(centers) < N:
        centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    return np.array(centers[:N])

def _hex_initial():
    centers = []
    spacing = 0.1
    y = -0.5 + spacing / 2
    row = 0
    while len(centers) < N and y <= 0.5:
        offset = (row % 2) * spacing * 0.5
        x = -0.5 + spacing / 2 + offset
        while len(centers) < N and x <= 0.5:
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                centers.append([x, y])
            x += spacing
        y += spacing * 0.8660254
        row += 1
    while len(centers) < N:
        centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    return np.array(centers[:N])

def _random_initial():
    return np.random.uniform(-0.45, 0.45, (N, 2))

def construct():
    global _start_time
    _start_time = time.time()
    np.random.seed(42)

    best_centers = None
    best_r = -1.0

    inits = []
    for m in [10, 11, 12, 13, 14]:
        inits.append(_grid_initial(m))
    inits.append(_hex_initial())
    for _ in range(6):
        inits.append(_random_initial())

    # Anneal schedule: large beta (smooth) -> small beta (sharp)
    betas = [200.0, 100.0, 50.0, 20.0, 10.0, 5.0]
    steps_per_beta = 60
    base_lr = 0.02

    for init in inits:
        if _time_left() < 12.0:
            break
        r = _compute_radius(init)
        if r > best_r:
            best_r = r
            best_centers = init.copy()
        try:
            opt = _anneal_optimize(init, betas, steps_per_beta, base_lr)
            r2 = _compute_radius(opt)
            if r2 > best_r:
                best_r = r2
                best_centers = opt.copy()
        except Exception:
            pass

    # Final polish with tighter betas on the best
    if best_centers is not None and _time_left() > 20.0:
        try:
            betas2 = [20.0, 10.0, 5.0, 3.0, 2.0]
            opt = _anneal_optimize(best_centers, betas2, 100, 0.01)
            r2 = _compute_radius(opt)
            if r2 > best_r:
                best_r = r2
                best_centers = opt.copy()
        except Exception:
            pass

    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers
# EVOLVE-BLOCK-END
