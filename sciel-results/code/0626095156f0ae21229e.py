"""Circle packing in unit square: physics-based relaxation with simulated annealing."""
import numpy as np

N = 26


def _valid(arr, eps=1e-9):
    """Check that all circles are inside [0,1]^2 and have disjoint interiors."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0) or np.any(x < -eps) or np.any(y < -eps) or np.any(x + r > 1 + eps) or np.any(y + r > 1 + eps):
        return False
    # pairwise distance check
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - eps:
                return False
    return True


def _score(arr):
    """Sum of radii, only if valid; else -inf."""
    if not _valid(arr):
        return -np.inf
    return float(np.sum(arr[:, 2]))


def _relax(x, y, r, iters=1500, dt=0.01, wall_margin=0.0):
    """
    Physics-based relaxation. x, y, r are 1D arrays.
    Returns (x, y, r) after relaxation.
    """
    n = len(r)
    x = x.copy()
    y = y.copy()
    r = r.copy()

    for it in range(iters):
        # slower cooling schedule for better convergence
        t = it / iters
        cool = 1.0 - 0.2 * t
        # grow radii slightly each step (annealing: allow packing tighter)
        growth_rate = 0.0003 + 0.0002 * (1.0 - t)
        r *= (1.0 + growth_rate * cool)

        # compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)

        # pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    dx = 1e-8
                    dy = 0.0
                    d = np.hypot(dx, dy)
                overlap = (r[i] + r[j]) - d
                if overlap > 0:
                    # force proportional to overlap, push apart
                    f = overlap * 2.5 * cool
                    fx[i] += f * dx / d
                    fy[i] += f * dy / d
                    fx[j] -= f * dx / d
                    fy[j] -= f * dy / d

        # wall repulsion
        for i in range(n):
            # left
            ov = r[i] - x[i]
            if ov > 0:
                fx[i] += ov * 4.0 * cool
            # right
            ov = r[i] - (1.0 - x[i])
            if ov > 0:
                fx[i] -= ov * 4.0 * cool
            # bottom
            ov = r[i] - y[i]
            if ov > 0:
                fy[i] += ov * 4.0 * cool
            # top
            ov = r[i] - (1.0 - y[i])
            if ov > 0:
                fy[i] -= ov * 4.0 * cool

        # update positions with adaptive dt
        effective_dt = dt * (0.5 + 0.5 * cool)
        x += effective_dt * fx
        y += effective_dt * fy

        # clamp to [0,1] (soft, keep inside)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        # keep radii reasonable
        r = np.clip(r, 0.001, 0.5)

    return x, y, r


def _maximize_radii(x, y, r, max_iters=500):
    """
    After positions are stable, try to maximize radii by growing them
    and resolving any resulting overlaps.
    """
    n = len(r)
    x = x.copy()
    y = y.copy()
    r = r.copy()

    for it in range(max_iters):
        # Try to grow all radii uniformly
        old_r = r.copy()
        r *= 1.001
        
        # Check if we need to resolve overlaps
        needs_resolve = False
        for i in range(n):
            # Wall check
            if x[i] < r[i] or x[i] > 1.0 - r[i] or y[i] < r[i] or y[i] > 1.0 - r[i]:
                needs_resolve = True
                break
            for j in range(i + 1, n):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < r[i] + r[j]:
                    needs_resolve = True
                    break
            if needs_resolve:
                break

        if not needs_resolve:
            continue

        # Resolve overlaps: push circles apart and adjust positions
        for _ in range(20):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    if d < 1e-12:
                        dx, dy, d = 1e-8, 0.0, 1e-8
                    need = r[i] + r[j]
                    if d < need:
                        push = (need - d) / 2.0 + 1e-7
                        ux, uy = dx / d, dy / d
                        x[i] += push * ux
                        y[i] += push * uy
                        x[j] -= push * ux
                        y[j] -= push * uy
                        moved = True

            # Wall constraints
            for i in range(n):
                if x[i] < r[i]:
                    x[i] = r[i]
                    moved = True
                if x[i] > 1.0 - r[i]:
                    x[i] = 1.0 - r[i]
                    moved = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    moved = True
                if y[i] > 1.0 - r[i]:
                    y[i] = 1.0 - r[i]
                    moved = True

            if not moved:
                break

        # If we couldn't accommodate the growth, revert
        if not _valid(np.column_stack([x, y, r]), eps=1e-8):
            r = old_r

    return x, y, r


def _final_polish(arr, iters=500):
    """
    Final local polish: resolve overlaps and push toward walls to maximize radii.
    arr is (n,3). Returns improved arr.
    """
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    n = len(r)

    for it in range(iters):
        changed = False
        # resolve pairwise overlaps by moving circles apart
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    dx, dy, d = 1e-8, 0.0, 1e-8
                need = r[i] + r[j]
                if d < need:
                    # push apart along the line
                    push = (need - d) / 2.0 + 1e-6
                    ux, uy = dx / d, dy / d
                    x[i] += push * ux
                    y[i] += push * uy
                    x[j] -= push * ux
                    y[j] -= push * uy
                    changed = True

        # resolve wall violations
        for i in range(n):
            if x[i] < r[i]:
                x[i] = r[i]
                changed = True
            if x[i] > 1.0 - r[i]:
                x[i] = 1.0 - r[i]
                changed = True
            if y[i] < r[i]:
                y[i] = r[i]
                changed = True
            if y[i] > 1.0 - r[i]:
                y[i] = 1.0 - r[i]
                changed = True

        # try to grow radii a tiny bit if there's room
        if not changed:
            # check if we can grow all radii slightly
            can_grow = True
            for i in range(n):
                # check walls
                if x[i] - r[i] < -1e-9 or 1.0 - x[i] - r[i] < -1e-9 or \
                   y[i] - r[i] < -1e-9 or 1.0 - y[i] - r[i] < -1e-9:
                    can_grow = False
                    break
            if can_grow:
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.hypot(x[i] - x[j], y[i] - y[j])
                        if d < r[i] + r[j] + 1e-9:
                            can_grow = False
                            break
                    if not can_grow:
                        break
                if can_grow:
                    r += 0.0003
                    changed = True

        if not changed:
            break

    arr[:, 0] = x
    arr[:, 1] = y
    arr[:, 2] = r
    return arr


def construct():
    best_arr = None
    best_score = -np.inf

    rng = np.random.default_rng(42)

    n_restarts = 15
    for restart in range(n_restarts):
        # Varied initial conditions
        if restart % 3 == 0:
            # Uniform random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)
            r = np.full(N, 0.02)
        elif restart % 3 == 1:
            # Grid-like initialization
            side = int(np.ceil(np.sqrt(N)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        positions.append(((i + 0.5) / side, (j + 0.5) / side))
            x = np.array([p[0] for p in positions])
            y = np.array([p[1] for p in positions])
            r = np.full(N, 0.02)
        else:
            # Clustered initialization
            x = rng.uniform(0.2, 0.8, N)
            y = rng.uniform(0.2, 0.8, N)
            r = np.full(N, 0.015)

        # Initial de-overlap: simple relaxation with small dt
        x, y, r = _relax(x, y, r, iters=300, dt=0.02)

        # Main annealing relaxation
        x, y, r = _relax(x, y, r, iters=2000, dt=0.012)

        # Maximize radii after positions stabilize
        x, y, r = _maximize_radii(x, y, r, max_iters=300)

        arr = np.column_stack([x, y, r])

        # Final polish
        arr = _final_polish(arr, iters=500)

        # clamp to be safe
        arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
        arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
        arr[:, 2] = np.clip(arr[:, 2], 0.0, 0.5)

        sc = _score(arr)
        if sc > best_score:
            best_score = sc
            best_arr = arr.copy()

    if best_arr is None:
        # fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])

    return best_arr
