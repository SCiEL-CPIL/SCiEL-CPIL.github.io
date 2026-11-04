"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time

N = 26
START = time.time()
DEADLINE = START + 130.0  # safety margin before 150s kill


def _valid(x, y, r):
    # circles inside [0,1]^2 and pairwise disjoint interiors
    if np.any(x < -1e-6) or np.any(y < -1e-6) or np.any(x > 1 + 1e-6) or np.any(y > 1 + 1e-6):
        return False
    if np.any(r < -1e-6):
        return False
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    if np.any(np.diag(d) < 0):
        return False
    np.fill_diagonal(d, np.inf)
    if np.any(d < r[:, None] + r[None, :] - 1e-6):
        return False
    return True


def _score(x, y, r):
    if _valid(x, y, r):
        return float(np.sum(r))
    return -np.inf


def construct():
    # --- initial grid seed ---
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    x = np.array([c[0] for c in cells], dtype=float)
    y = np.array([c[1] for c in cells], dtype=float)
    r = np.array([c[2] for c in cells], dtype=float)

    best = (x.copy(), y.copy(), r.copy(), _score(x, y, r))

    def save():
        nonlocal best
        s = _score(x, y, r)
        if s > best[3]:
            best = (x.copy(), y.copy(), r.copy(), s)

    save()

    # --- physics relaxation: pairwise repulsion, wall repulsion, gravity toward center ---
    def relax():
        nonlocal x, y, r
        dt = 0.02
        for _ in range(400):
            if time.time() > DEADLINE:
                return
            fx = np.zeros(N)
            fy = np.zeros(N)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-9:
                        dx = 1e-9
                        dy = 0.0
                        d = 1e-9
                    # repel if closer than sum of radii + margin
                    gap = r[i] + r[j] + 0.02 - d
                    if gap > 0:
                        f = 8.0 * gap
                        fx[i] += f * dx / d
                        fy[i] += f * dy / d
                        fx[j] -= f * dx / d
                        fy[j] -= f * dy / d
                # walls
                if x[i] - r[i] < 0:
                    fx[i] += 8.0 * (r[i] - x[i])
                if 1.0 - x[i] - r[i] < 0:
                    fx[i] -= 8.0 * (x[i] - 1.0 + r[i])
                if y[i] - r[i] < 0:
                    fy[i] += 8.0 * (r[i] - y[i])
                if 1.0 - y[i] - r[i] < 0:
                    fy[i] -= 8.0 * (y[i] - 1.0 + r[i])
                # gentle centering
                fx[i] += 0.05 * (0.5 - x[i])
                fy[i] += 0.05 * (0.5 - y[i])
            x += dt * fx
            y += dt * fy
            # clamp inside
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            # grow radii a little each step
            r += 0.0004
            # keep radii feasible relative to position (shrink if a circle pokes out)
            for i in range(N):
                mx = min(x[i], 1.0 - x[i])
                my = min(y[i], 1.0 - y[i])
                m = max(mx, my)
                if r[i] > m:
                    r[i] = m
            save()

    relax()
    save()

    # --- final polish: iterative pairwise separation + wall projection + radius grow ---
    for _ in range(300):
        if time.time() > DEADLINE:
            break
        # separate overlapping pairs
        for _ in range(3):
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    req = r[i] + r[j]
                    if d < req:
                        if d < 1e-9:
                            dx = 1e-9
                            d = 1e-9
                        push = (req - d) / 2.0 + 1e-5
                        x[i] += push * dx / d
                        y[i] += push * dy / d
                        x[j] -= push * dx / d
                        y[j] -= push * dy / d
        # project walls
        for i in range(N):
            if x[i] < r[i]:
                x[i] = r[i]
            if x[i] > 1.0 - r[i]:
                x[i] = 1.0 - r[i]
            if y[i] < r[i]:
                y[i] = r[i]
            if y[i] > 1.0 - r[i]:
                y[i] = 1.0 - r[i]
        # try to grow radii
        for i in range(N):
            # max radius from walls
            wall = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            # max radius from neighbors
            neigh = np.inf
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                neigh = min(neigh, d - r[j])
            newr = min(wall, neigh)
            if newr > r[i]:
                r[i] = newr
        save()

    # return best found
    bx, by, br, _ = best
    return np.column_stack([bx, by, br])
