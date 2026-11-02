"""Circle packing in unit square: sequential greedy + local refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
EPS = 1e-9


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _is_valid(pos, rad):
    """Check all circles are inside [0,1]^2 and non-overlapping."""
    n = len(rad)
    for i in range(n):
        if rad[i] <= 0:
            return False
        if pos[i, 0] < rad[i] + EPS or pos[i, 0] > 1 - rad[i] - EPS:
            return False
        if pos[i, 1] < rad[i] + EPS or pos[i, 1] > 1 - rad[i] - EPS:
            return False
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < rad[i] + rad[j] - 1e-9:
                return False
    return True


def _greedy_place(pos, rad, n_placed):
    """Place circle n_placed greedily: find position maximizing min clearance."""
    # Candidate positions: grid + random
    candidates = []
    # Coarse grid
    for gx in range(21):
        for gy in range(21):
            x = (gx + 0.5) / 21
            y = (gy + 0.5) / 21
            candidates.append((x, y))
    # Random
    rng_local = np.random.default_rng(hash((n_placed, tuple(pos[:, :2].round(4).flatten().tolist()))) % (2**32))
    for _ in range(200):
        x = rng_local.random() * 0.9 + 0.05
        y = rng_local.random() * 0.9 + 0.05
        candidates.append((x, y))

    best_score = -1.0
    best_xy = None
    for x, y in candidates:
        # Max radius at this position
        dmin = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(n_placed):
            d = np.hypot(x - pos[j, 0], y - pos[j, 1])
            dmin = min(dmin, d - rad[j])
        if dmin > best_score:
            best_score = dmin
            best_xy = (x, y)

    if best_xy is None:
        return None, 0.0
    return best_xy, best_score


def _local_refine(pos, rad, deadline):
    """Refine positions via coordinate descent + radius updates."""
    n = len(rad)
    pos = pos.copy()
    rad = rad.copy()

    for iteration in range(200):
        if time.time() > deadline - 2:
            break
        # Update radii
        rad = _max_radius(pos, rad)

        # Coordinate descent: for each circle, try small moves
        improved = False
        for i in range(n):
            if time.time() > deadline - 2:
                break
            best_pos = pos[i].copy()
            best_rad = rad[i]

            # Try radial and angular perturbations
            for angle in range(0, 360, 15):
                for dist in [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]:
                    nx = pos[i, 0] + dist * np.cos(np.radians(angle))
                    ny = pos[i, 1] + dist * np.sin(np.radians(angle))
                    if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                        continue
                    pos[i] = [nx, ny]
                    new_rad = _max_radius(pos, rad)[i]
                    # Check no overlaps
                    ok = True
                    for j in range(n):
                        if i != j:
                            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                            if d < rad[i] + rad[j] - 1e-9:
                                ok = False
                                break
                    if ok and new_rad > best_rad + 1e-6:
                        best_rad = new_rad
                        best_pos = [nx, ny]
                    pos[i] = [nx, ny]  # restore for next trial
            pos[i] = best_pos
            rad[i] = best_rad
            if not np.allclose(pos[i], best_pos):
                improved = True

        if not improved:
            break

    return pos, rad


def construct():
    t_start = time.time()
    DEADLINE = t_start + 145.0  # 145s to leave margin

    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        if len(rad) < N:
            return
        # Validate
        for i in range(N):
            if rad[i] <= 0:
                return
            if pos[i, 0] < rad[i] + EPS or pos[i, 0] > 1 - rad[i] - EPS:
                return
            if pos[i, 1] < rad[i] + EPS or pos[i, 1] > 1 - rad[i] - EPS:
                return
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if d < rad[i] + rad[j] - 1e-9:
                    return
        s = float(np.sum(rad[:N]))
        if s > best_score:
            best_score = s
            best = np.column_stack([pos[:N], rad[:N]]).copy()

    # ---- Phase 1: Multiple greedy sequential placements with different orderings ----
    # Strategy: place circles in different orders (by size preference, random, etc.)
    orderings = []

    # Ordering 1: Place in grid order (systematic)
    for side in range(4, 8):
        grid_pos = []
        for i in range(side):
            for j in range(side):
                grid_pos.append(((i + 0.5) / side, (j + 0.5) / side))
        grid_pos = grid_pos[:N]
        orderings.append(('grid', grid_pos))

    # Ordering 2: Random permutations of a base grid
    for _ in range(10):
        base = []
        for i in range(6):
            for j in range(6):
                base.append(((i + 0.5) / 6, (j + 0.5) / 6))
        perm = list(range(36))
        rng.shuffle(perm)
        shuffled = [base[k] for k in perm[:N]]
        orderings.append(('rand', shuffled))

    # Ordering 3: Corner-to-center spiral
    for _ in range(5):
        spiral = []
        cx, cy = 0.5, 0.5
        for r in np.linspace(0.05, 0.45, N):
            angle = rng.random() * 2 * np.pi
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            spiral.append((x, y))
        orderings.append(('spiral', spiral))

    # Run greedy placement for each ordering
    for order_name, order_points in orderings:
        if time.time() > DEADLINE - 30:
            break

        pos = np.zeros((N, 2))
        rad = np.zeros(N)

        # Use the ordering as hints for initial positions, but greedily optimize
        # Place circles one by one, using the hint position as a starting candidate
        for k in range(N):
            if time.time() > DEADLINE - 30:
                break
            hint_x, hint_y = order_points[k]

            # Greedy: find best position for circle k
            best_r = 0.0
            best_x, best_y = hint_x, hint_y

            # Try hint position
            dmin = min(hint_x, 1.0 - hint_x, hint_y, 1.0 - hint_y)
            for j in range(k):
                d = np.hypot(hint_x - pos[j, 0], hint_y - pos[j, 1])
                dmin = min(dmin, d - rad[j])
            best_r = max(0.0, dmin)

            # Try grid candidates
            for gx in range(15):
                for gy in range(15):
                    x = (gx + 0.5) / 15
                    y = (gy + 0.5) / 15
                    dmin2 = min(x, 1.0 - x, y, 1.0 - y)
                    for j in range(k):
                        d = np.hypot(x - pos[j, 0], y - pos[j, 1])
                        dmin2 = min(dmin2, d - rad[j])
                    if dmin2 > best_r:
                        best_r = dmin2
                        best_x, best_y = x, y

            # Try random candidates
            for _ in range(100):
                x = rng.random()
                y = rng.random()
                dmin2 = min(x, 1.0 - x, y, 1.0 - y)
                for j in range(k):
                    d = np.hypot(x - pos[j, 0], y - pos[j, 1])
                    dmin2 = min(dmin2, d - rad[j])
                if dmin2 > best_r:
                    best_r = dmin2
                    best_x, best_y = x, y

            pos[k] = [best_x, best_y]
            rad[k] = best_r

        # Local refinement
        if time.time() < DEADLINE - 15:
            pos, rad = _local_refine(pos, rad, DEADLINE)

        consider(pos, rad)

    # ---- Phase 2: Penalty-based optimization from best greedy solution ----
    if best is not None:
        x0 = best.flatten().copy()

        def make_obj(x, lam):
            p = x[:2 * N].reshape(N, 2)
            r = x[2 * N:]
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
            viol = np.maximum(r[:, None] + r[None, :] - d, 0.0)
            pair = np.sum(viol ** 2)
            w = np.maximum(r - p, 0.0) + np.maximum(r - (1.0 - p), 0.0)
            wall = np.sum(w ** 2)
            return -np.sum(r) + lam * (pair + wall)

        def make_grad(x, lam):
            p = x[:2 * N].reshape(N, 2)
            r = x[2 * N:]
            g = np.empty_like(x)
            g[:2 * N] = 0.0
            g[2 * N:] = -1.0

            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
            d = np.maximum(d, EPS)
            diff = p[:, None, :] - p[None, :, :]
            viol = np.maximum(r[:, None] + r[None, :] - d, 0.0)
            d_pair = 2.0 * viol
            d_dij = -np.einsum('ijk,ijk->ij', diff, d)
            d_ra = d_pair * (1.0 + d_dij)
            d_rb = d_pair * (1.0 - d_dij)
            np.fill_diagonal(d_ra, 0.0)
            np.fill_diagonal(d_rb, 0.0)
            g[2 * N:] += lam * (d_ra.sum(axis=1) + d_rb.sum(axis=1))
            dp = (d_ra - d_rb)[:, :, None] * diff / d[:, :, None]
            np.fill_diagonal(dp.sum(axis=0), 0.0)
            g[:2 * N] += lam * dp.sum(axis=1).reshape(-1)

            a = np.maximum(r - p, 0.0)
            b = np.maximum(r - (1.0 - p), 0.0)
            g[2 * N:] += lam * 2.0 * (a + b)
            gp = lam * (-2.0 * a + 2.0 * b)
            g[:2 * N] += gp.reshape(-1)
            return g

        # Try multiple lambda values
        for lam in [100.0, 30.0, 10.0, 3.0, 1.0, 0.3]:
            if time.time() > DEADLINE - 10:
                break
            try:
                maxiter = 800 if lam < 2.0 else 400
                res = minimize(make_obj, x0, jac=make_grad, method='L-BFGS-B',
                               args=(lam,),
                               options={'maxiter': maxiter, 'maxfun': 8000,
                                        'gtol': 1e-8, 'maxls': 30, 'maxcor': 15})
                p_opt = res.x[:2 * N].reshape(N, 2)
                r_opt = np.maximum(res.x[2 * N:], 0.0)
                # Polish radii
                r_opt = _max_radius(p_opt, r_opt)
                consider(p_opt, r_opt)
                # Update x0 for next lambda
                x0 = np.concatenate([p_opt.flatten(), r_opt])
            except Exception:
                pass

        # Final polish: iterative radius update + small position nudge
        if best is not None:
            for _ in range(20):
                if time.time() > DEADLINE - 5:
                    break
                p = best[:, :2].copy()
                r = best[:, 2].copy()
                r = _max_radius(p, r)
                # Tiny random perturbations to escape local minima
                p += 0.002 * (rng.random((N, 2)) * 2 - 1)
                p = np.clip(p, 0.0, 1.0)
                r = _max_radius(p, r)
                if _is_valid(p, r):
                    s = float(np.sum(r))
                    if s > best_score:
                        best_score = s
                        best = np.column_stack([p, r]).copy()
                else:
                    # Revert
                    p = best[:, :2].copy()
                    r = _max_radius(p, r)

    if best is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
