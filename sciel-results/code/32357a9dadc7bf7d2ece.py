"""Circle packing in unit square: pack 26 circles, maximize sum of radii.

Structural change vs. seed:
  - Global search: multi-start differential evolution (DE/rand/1) over the
    full (x, y, r) configuration vector, using a smooth soft-penalty objective.
  - Local polish: fast per-circle best-fit (vectorized) to restore validity.
  - No gradient polish; DE provides the global move, best-fit the local move.
"""
import time
import numpy as np

N = 26


def _eval_packing(pos):
    """Return (sum_radii, is_valid) for an (N,3) array [x, y, r]."""
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    total_r = float(np.sum(rs))
    if np.any(rs < 0) or np.any(xs < 0) or np.any(xs > 1) or np.any(ys < 0) or np.any(ys > 1):
        return total_r, False
    if np.any(rs > xs) or np.any(rs > 1 - xs) or np.any(rs > ys) or np.any(rs > 1 - ys):
        return total_r, False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
            if d2 < (rs[i] + rs[j]) ** 2 - 1e-9:
                return total_r, False
    return total_r, True


def _max_radius_vectorized(gx, gy, others_xy, others_r):
    """Max feasible radius at each grid point given other circles (vectorized)."""
    wall = np.minimum(np.minimum(gx, 1.0 - gx), np.minimum(gy, 1.0 - gy))
    if others_xy is not None and len(others_xy) > 0:
        dx = gx[:, None] - others_xy[None, :, 0]
        dy = gy[:, None] - others_xy[None, :, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        circle_clear = np.min(dists - others_r[None, :], axis=1)
        return np.maximum(0.0, np.minimum(wall, circle_clear))
    return np.maximum(0.0, wall)


def _best_fit(pos, rng, deadline, max_rounds=60):
    """Per-circle best-fit local search (fast, vectorized grid + random)."""
    pos = pos.copy()
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline:
                break
            others_idx = [j for j in range(N) if j != i]
            others_xy = pos[others_idx, :2]
            others_r = pos[others_idx, 2]
            best_r = pos[i, 2]
            bx, by = pos[i, 0], pos[i, 1]
            cx, cy = pos[i, 0], pos[i, 1]
            sr = max(0.12, 2 * pos[i, 2] + 0.04)
            for scale in (1.0, 0.5, 0.25):
                n_pts = max(10, int(28 * scale))
                gx = np.linspace(bx - sr * scale, bx + sr * scale, n_pts)
                gy = np.linspace(by - sr * scale, by + sr * scale, n_pts)
                GX, GY = np.meshgrid(gx, gy)
                GX = GX.ravel()
                GY = GY.ravel()
                m = (GX >= 0) & (GX <= 1) & (GY >= 0) & (GY <= 1)
                GX, GY = GX[m], GY[m]
                if len(GX) == 0:
                    continue
                radii = _max_radius_vectorized(GX, GY, others_xy, others_r)
                k = int(np.argmax(radii))
                if radii[k] > best_r + 1e-11:
                    best_r = float(radii[k])
                    bx, by = float(GX[k]), float(GY[k])
            # a few random probes
            n_rand = 30
            rx = rng.uniform(0, 1, n_rand)
            ry = rng.uniform(0, 1, n_rand)
            radii = _max_radius_vectorized(rx, ry, others_xy, others_r)
            k = int(np.argmax(radii))
            if radii[k] > best_r + 1e-11:
                best_r = float(radii[k])
                bx, by = float(rx[k]), float(ry[k])
            if best_r > pos[i, 2] + 1e-11:
                pos[i, 0] = bx
                pos[i, 1] = by
                pos[i, 2] = best_r
                improved = True
        if not improved:
            break
    # clip to validity
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], 0, 1)
        pos[i, 1] = np.clip(pos[i, 1], 0, 1)
        pos[i, 2] = min(pos[i, 2], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        pos[i, 2] = max(0.0, pos[i, 2])
    return pos


def _smooth_obj(x, lam=80.0):
    """Smooth objective: minimize -sum(r) + penalty for boundary & overlap violations."""
    p = x.reshape(N, 3)
    xs, ys, rs = p[:, 0], p[:, 1], p[:, 2]
    # boundary violation (r beyond wall clearance), squared
    b = np.maximum(0.0, rs - xs) ** 2 + np.maximum(0.0, rs - (1 - xs)) ** 2
    b += np.maximum(0.0, rs - ys) ** 2 + np.maximum(0.0, rs - (1 - ys)) ** 2
    # overlap violation (smooth): (r_i + r_j - dist)^2 when positive
    ov = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 + 1e-12)
            v = np.maximum(0.0, (rs[i] + rs[j]) - d)
            ov += v * v
    # negative radius
    nr = np.maximum(0.0, -rs) ** 2
    return -float(np.sum(rs)) + lam * (float(np.sum(b)) + ov + float(np.sum(nr)))


def _de_evolve(pos0, rng, deadline, pop=36, F=0.6, CR=0.6, max_gen=400):
    """Differential evolution over the flattened config, smooth objective."""
    pos0 = np.clip(pos0, 0, 1).copy()
    for i in range(N):
        pos0[i, 2] = min(pos0[i, 2], pos0[i, 0], 1 - pos0[i, 0], pos0[i, 1], 1 - pos0[i, 1])
    pop = max(pop, 4)
    # initial population: pos0 + random perturbations
    P = np.zeros((pop, N, 3))
    P[0] = pos0
    for k in range(1, pop):
        P[k] = pos0 + rng.normal(0, 0.06, (N, 3))
        P[k, :2] = np.clip(P[k, :2], 0, 1)
        P[k, 2] = np.clip(P[k, 2], 0, 0.5)
    # bounds per coordinate
    lo = np.zeros(N * 3)
    hi = np.zeros(N * 3)
    hi[0::3] = 1.0
    hi[1::3] = 1.0
    hi[2::3] = 0.5

    def eval_pop():
        f = np.empty(pop)
        for k in range(pop):
            f[k] = _smooth_obj(P[k].flatten())
        return f

    f = eval_pop()
    best_idx = int(np.argmin(f))
    best_x = P[best_idx].copy()
    best_f = f[best_idx]

    for gen in range(max_gen):
        if time.time() > deadline:
            break
        for i in range(pop):
            # pick 3 distinct indices different from i
            a, b, c = rng.choice(pop, size=3, replace=False)
            while a == i or b == i or c == i:
                a, b, c = rng.choice(pop, size=3, replace=False)
            v = P[a] + F * (P[b] - P[c])
            # crossover
            trial = P[i].copy()
            j0 = rng.randint(0, N * 3)
            for j in range(N * 3):
                if (j == j0) or (rng.random() < CR):
                    trial.flat[j] = v.flat[j]
            # clamp to bounds
            trial = np.clip(trial, lo, hi).reshape(N, 3)
            ft = _smooth_obj(trial.flatten())
            if ft <= f[i]:
                P[i] = trial
                f[i] = ft
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
    return best_x.reshape(N, 3)


def _grid_init(rng):
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 1] = (row + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 2] = 0.5 / side - 0.01
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1 - pos[i, 2])
        pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1 - pos[i, 2])
    return pos


def construct():
    deadline = time.time() + 135  # safety margin before 150s
    rng = np.random.default_rng(20240607)

    best_pos = None
    best_val = -np.inf

    def consider(pos):
        nonlocal best_pos, best_val
        pos = np.asarray(pos, dtype=float)
        total, valid = _eval_packing(pos)
        if valid and total > best_val:
            best_val = total
            best_pos = pos.copy()

    # Multi-start: several diverse seeds -> DE -> best-fit polish
    n_starts = 5
    per_start = (deadline - (time.time() + 5)) / float(n_starts)

    for s in range(n_starts):
        t0 = time.time()
        if t0 > deadline - 12:
            break
        # build a diverse seed
        if s == 0:
            seed = _grid_init(rng)
        elif s == 1:
            # ring seed
            seed = np.zeros((N, 3))
            seed[0] = [0.5, 0.5, 0.12]
            idx = 1
            for k in range(6):
                a = 2 * np.pi * k / 6
                seed[idx] = [0.5 + 0.28 * np.cos(a), 0.5 + 0.28 * np.sin(a), 0.07]
                idx += 1
            for k in range(12):
                a = 2 * np.pi * k / 12 + np.pi / 12
                seed[idx] = [0.5 + 0.40 * np.cos(a), 0.5 + 0.40 * np.sin(a), 0.045]
                idx += 1
            corners = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
                       (0.5, 0.06), (0.5, 0.94), (0.06, 0.5)]
            for k in range(7):
                seed[idx] = [corners[k][0], corners[k][1], 0.04]
                idx += 1
            for i in range(N):
                seed[i, 0] = np.clip(seed[i, 0], seed[i, 2], 1 - seed[i, 2])
                seed[i, 1] = np.clip(seed[i, 1], seed[i, 2], 1 - seed[i, 2])
        else:
            seed = _grid_init(rng)
            seed[:, :2] += rng.normal(0, 0.04, (N, 2))
            seed[:, :2] = np.clip(seed[:, :2], 0, 1)
            for i in range(N):
                seed[i, 2] = min(seed[i, 2], seed[i, 0], 1 - seed[i, 0], seed[i, 1], 1 - seed[i, 1])

        # quick best-fit to give DE a sane starting point
        seed = _best_fit(seed, rng, min(deadline - 10, t0 + per_start * 0.25), max_rounds=25)

        # global DE
        de_deadline = min(deadline - 12, t0 + per_start * 0.75)
        evolved = _de_evolve(seed, rng, de_deadline, pop=34, F=0.6, CR=0.6, max_gen=400)

        # polish to validity
        polished = _best_fit(evolved, rng, min(deadline - 8, t0 + per_start * 0.95), max_rounds=50)
        consider(polished)

    # Final global polish pass on the best found, using remaining time
    if best_pos is not None and time.time() < deadline - 6:
        final = _best_fit(best_pos, rng, deadline - 4, max_rounds=80)
        consider(final)

    if best_pos is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_pos
