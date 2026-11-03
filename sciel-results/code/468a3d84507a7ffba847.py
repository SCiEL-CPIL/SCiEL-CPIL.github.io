import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 148.0  # Safety margin

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    # ------------------------------------------------------------------
    # Exact radius
    # ------------------------------------------------------------------
    def compute_radius(positions):
        positions = np.asarray(positions, dtype=np.float64).reshape(N, 3)
        if not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        diff = positions[:, None, :] - positions[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        walls = np.min(0.5 - np.abs(positions), axis=1)
        min_wall = np.min(walls)
        return float(min(min_pair, min_wall))

    # ------------------------------------------------------------------
    # Smooth surrogate with analytic gradient
    # ------------------------------------------------------------------
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def smooth_obj_grad(x_flat, beta):
        c = x_flat.reshape(N, 3)

        diff = c[tri_i] - c[tri_j]
        norms = np.linalg.norm(diff, axis=1)
        norms_safe = np.where(norms < 1e-12, 1e-12, norms)
        g_pairs = 0.5 * norms_safe

        g_walls = np.concatenate([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
            0.5 - c[:, 2], c[:, 2] + 0.5,
        ])

        g_all = np.concatenate([g_pairs, g_walls])

        log_sum_exp = sp.logsumexp(-beta * g_all)
        val = -log_sum_exp / beta

        log_w = -beta * g_all - log_sum_exp
        weights = np.exp(log_w)

        grad_c = np.zeros_like(c)

        wp = weights[:n_pairs]
        mask = wp > 1e-14
        if np.any(mask):
            factors = np.zeros(n_pairs)
            factors[mask] = 0.5 * wp[mask] / norms_safe[mask]
            contrib = factors[:, None] * diff
            np.add.at(grad_c, tri_i, -contrib)
            np.add.at(grad_c, tri_j, contrib)

        ws = weights[n_pairs:]
        grad_c[:, 0] -= ws[0:N]
        grad_c[:, 0] += ws[N:2*N]
        grad_c[:, 1] -= ws[2*N:3*N]
        grad_c[:, 1] += ws[3*N:4*N]
        grad_c[:, 2] -= ws[4*N:5*N]
        grad_c[:, 2] += ws[5*N:6*N]

        return -val, (-grad_c).flatten()

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    bounds = [(-0.499999, 0.499999)] * (N * 3)

    def local_smooth(c_init, beta, maxiter=200):
        if remaining() < 5:
            return c_init
        x0 = c_init.flatten()

        def f(x):
            v, _ = smooth_obj_grad(x, beta)
            return v

        def g(x):
            _, gr = smooth_obj_grad(x, beta)
            return gr

        try:
            res = opt.minimize(f, x0, jac=g, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def local_direct(c_init, maxiter=300):
        if remaining() < 5:
            return c_init
        x0 = c_init.flatten()

        def f(x):
            return -compute_radius(x.reshape(N, 3))

        try:
            res = opt.minimize(f, x0, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8})
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def subgradient_ascent(c_init, maxiter=200, lr_start=0.005, lr_end=0.0001):
        if remaining() < 5:
            return c_init
        pos = c_init.copy()
        best = pos.copy()
        best_r = compute_radius(pos)
        for it in range(maxiter):
            if remaining() < 3:
                break
            lr = lr_start * (lr_end / lr_start) ** (it / max(1, maxiter - 1))
            
            # Compute subgradient of the radius
            diff = pos[:, None, :] - pos[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            min_pair_val = np.min(d) / 2.0
            walls = 0.5 - np.abs(pos)
            min_wall_val = np.min(walls)

            grad = np.zeros_like(pos)

            if min_pair_val < min_wall_val:
                idx = np.argmin(d)
                i, j = np.unravel_index(idx, (N, N))
                if i == j:
                    d2 = d.copy()
                    np.fill_diagonal(d2, np.inf)
                    idx = np.argmin(d2)
                    i, j = np.unravel_index(idx, (N, N))
                dv = pos[i] - pos[j]
                dn = np.linalg.norm(dv)
                if dn > 1e-12:
                    u = dv / dn
                    grad[i] += 0.5 * u
                    grad[j] -= 0.5 * u
                r = min_pair_val
            else:
                idx = np.argmin(walls)
                i, dim = np.unravel_index(idx, (N, 3))
                if pos[i, dim] > 0:
                    grad[i, dim] -= 1.0
                else:
                    grad[i, dim] += 1.0
                r = min_wall_val

            if r > best_r:
                best_r = r
                best = pos.copy()
            
            pos = np.clip(pos + lr * grad, -0.499999, 0.499999)
            if it > 20 and np.linalg.norm(grad) < 1e-8:
                break
        return best

    def coordinate_descent(c_init, maxiter=60):
        if remaining() < 3:
            return c_init
        pos = c_init.copy()
        step = 0.006
        for _ in range(maxiter):
            if remaining() < 2:
                break
            improved = False
            for i in range(N):
                for dim in range(3):
                    old = pos[i, dim]
                    r_cur = compute_radius(pos)
                    pos[i, dim] = min(0.499999, old + step)
                    r_p = compute_radius(pos)
                    pos[i, dim] = max(-0.499999, old - step)
                    r_n = compute_radius(pos)
                    pos[i, dim] = old
                    if r_p > r_cur + 1e-13:
                        pos[i, dim] = min(0.499999, old + step)
                        improved = True
                    elif r_n > r_cur + 1e-13:
                        pos[i, dim] = max(-0.499999, old - step)
                        improved = True
            if not improved:
                step *= 0.85
                if step < 1e-6:
                    break
        return pos

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def make_bcc():
        pts = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    pts.append([x, y, z])
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    pts.append([x + 0.5, y + 0.5, z + 0.5])
        arr = np.array(pts) - np.array([1.5, 1.5, 1.5])
        d = np.linalg.norm(arr, axis=1)
        sel = arr[np.argsort(d)[:N]]
        mx = np.max(np.linalg.norm(sel, axis=1))
        if mx > 0:
            sel *= 0.45 / mx
        return sel

    def make_grid():
        m = 4
        t = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in t for y in t for x in t])
        return grid[:N].copy()

    def make_random(seed):
        return np.random.RandomState(seed).uniform(-0.45, 0.45, size=(N, 3))

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------
    best_pos = None
    best_r = -1.0

    def update(pos):
        nonlocal best_pos, best_r
        r = compute_radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        return r

    # Generate a diverse set of initializations
    inits = [make_bcc(), make_grid()]
    # Add many random inits
    for seed in range(20):
        inits.append(make_random(seed))

    for init in inits:
        if remaining() < 15:
            break
        pos = np.clip(init, -0.499999, 0.499999)

        # Beta-annealed smooth optimisation
        if remaining() > 25:
            # Aggressive annealing
            for beta in [1.0, 3.0, 10.0, 30.0, 100.0]:
                if remaining() < 6:
                    break
                pos = local_smooth(pos, beta, maxiter=150)

        # Direct L-BFGS-B polish
        if remaining() > 12:
            pos = local_direct(pos, maxiter=400)

        # Subgradient fine-tune
        if remaining() > 7:
            pos = subgradient_ascent(pos, maxiter=200, lr_start=0.004, lr_end=0.00005)

        # Coordinate descent
        if remaining() > 4:
            pos = coordinate_descent(pos, maxiter=60)

        update(pos)

    # --- Perturbation refinement around best ---
    if best_pos is not None and remaining() > 20:
        for k in range(10): # More perturbations
            if remaining() < 6:
                break
            rng = np.random.RandomState(2000 + k)
            scale = 0.015 * (1.0 - k / 11.0)
            pert = np.clip(best_pos + rng.normal(0, scale, size=best_pos.shape),
                           -0.499999, 0.499999)

            if remaining() > 14:
                for beta in [10.0, 50.0, 150.0]:
                    if remaining() < 5:
                        break
                    pert = local_smooth(pert, beta, maxiter=100)

            if remaining() > 8:
                pert = local_direct(pert, maxiter=300)

            if remaining() > 5:
                pert = subgradient_ascent(pert, maxiter=150, lr_start=0.002, lr_end=0.00003)

            if remaining() > 3:
                pert = coordinate_descent(pert, maxiter=45)

            update(pert)

    # --- Final high-precision polish ---
    if best_pos is not None and remaining() > 12:
        try:
            fp = local_direct(best_pos, maxiter=500)
            fp = subgradient_ascent(fp, maxiter=250, lr_start=0.0015, lr_end=0.000005)
            fp = coordinate_descent(fp, maxiter=70)
            update(fp)
        except Exception:
            pass

    # --- Fallback ---
    if best_pos is None:
        best_pos = np.clip(make_bcc(), -0.499999, 0.499999)

    return best_pos
