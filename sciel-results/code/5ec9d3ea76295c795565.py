import numpy as np
import time

N = 32
_DEADLINE = 135.0  # wall-clock safety margin (evaluator kills at 150s)

def _validity_check(arr, eps=1e-9):
    """Return True if all circles are disjoint (interiors) and inside [0,1]^2."""
    if arr is None or len(arr) != N:
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r <= 0):
        return False
    # inside square
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    # disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - eps:
                return False
    return True

def _score(arr):
    """Sum of radii if valid, else -inf."""
    if _validity_check(arr):
        return float(np.sum(arr[:, 2]))
    return -np.inf

def _hex_seed():
    """Staggered hexagonal-ish lattice trimmed to N circles."""
    pts = []
    # Use a hex packing: rows offset, vertical spacing = r*sqrt(3), horizontal = 2r
    # We'll lay out a grid of candidate centers and pick N, then compute a base radius.
    # Simpler: place on a hex lattice with spacing 1.0, scale later.
    row_h = 0.8660254  # sqrt(3)/2
    cols = 0
    centers = []
    while len(centers) < N + 10:
        y = row_h * cols
        xoff = 0.5 if (cols % 2) else 0.0
        for i in range(0, 20):
            centers.append((xoff + i, y))
            if len(centers) >= N + 10:
                break
        cols += 1
    centers = centers[:N]
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    # normalize into [0,1] with margin
    cx = (cx - cx.min()) / (cx.max() - cx.min() + 1e-12)
    cy = (cy - cy.min()) / (cy.max() - cy.min() + 1e-12)
    # base radius: min distance to boundary /2, and half min pair distance
    r = np.full(N, 0.0)
    for i in range(N):
        dmin = min(cx[i], 1 - cx[i], cy[i], 1 - cy[i])
        r[i] = dmin * 0.5
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            r[i] = min(r[i], d * 0.5)
    return np.column_stack([cx, cy, r])

def _repulsion_optimize(x0, y0, r0, max_iter=4000, dt=0.02, grow=0.0008,
                        wall_k=40.0, pair_k=60.0, max_r=0.25, seed=0,
                        deadline=None):
    """
    Physics-style optimization: repel overlapping circles and grow radii.
    Returns best valid (x,y,r) found during the run.
    """
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = np.minimum(r0.copy().astype(float), max_r)

    best = None
    best_score = -np.inf

    # Precompute index pairs for vectorized distance computation
    ii, jj = np.triu_indices(N, k=1)
    n_pairs = len(ii)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Compute pairwise distances and forces (vectorized)
        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        d = np.sqrt(dx * dx + dy * dy) + 1e-12
        s = r[ii] + r[jj]
        overlap = np.maximum(0, s - d)
        
        # Pair forces: f = pair_k * overlap / d * direction
        f_mag = pair_k * overlap / d
        fx = np.zeros(N)
        fy = np.zeros(N)
        np.add.at(fx, ii, f_mag * dx)
        np.add.at(fx, jj, -f_mag * dx)
        np.add.at(fy, ii, f_mag * dy)
        np.add.at(fy, jj, -f_mag * dy)

        # Wall forces
        wall_viol_x = np.maximum(0, r - x) + np.maximum(0, x + r - 1)
        wall_viol_y = np.maximum(0, r - y) + np.maximum(0, y + r - 1)
        # Force pushes away from wall
        push_x = np.where(x < r, wall_k * (r - x), 0) - np.where(x > 1 - r, wall_k * (x + r - 1), 0)
        push_y = np.where(y < r, wall_k * (r - y), 0) - np.where(y > 1 - r, wall_k * (y + r - 1), 0)
        fx += push_x
        fy += push_y

        # Update positions
        x += dt * fx
        y += dt * fy
        # Grow radii
        r = np.minimum(r + grow, max_r)
        # Clamp to square (centers)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        if it % 15 == 0 or it == max_iter - 1:
            arr = np.column_stack([x, y, r])
            sc = _score(arr)
            if sc > best_score:
                best_score = sc
                best = arr.copy()

    if best is None:
        best = np.column_stack([x, y, r])
    return best, best_score

def _shrink_to_valid(arr):
    """Given a candidate, shrink radii minimally to guarantee validity."""
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = arr[:, 2].copy()
    # ensure inside
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
    # ensure disjoint: iteratively shrink
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if d < s:
                    # shrink the larger one (or both half)
                    need = s - d
                    # reduce r[i] and r[j] so that r[i]+r[j] <= d
                    # keep ratio
                    tot = r[i] + r[j]
                    if tot <= 0:
                        continue
                    new_tot = max(0.0, d)
                    r[i] = r[i] * new_tot / tot
                    r[j] = r[j] * new_tot / tot
                    changed = True
        if not changed:
            break
    return np.column_stack([x, y, r])

def _polish(arr, max_iter=2000, deadline=None):
    """High-precision polish: fine-tune positions to maximize sum of radii."""
    x = arr[:, 0].copy().astype(float)
    y = arr[:, 1].copy().astype(float)
    r = arr[:, 2].copy().astype(float)
    
    ii, jj = np.triu_indices(N, k=1)
    
    best = arr.copy()
    best_score = _score(best)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current min distance for each pair
        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        d = np.sqrt(dx * dx + dy * dy) + 1e-12
        s = r[ii] + r[jj]
        
        # Try to grow radii slightly by finding slack
        # For each circle, compute the max radius it can have given neighbors
        new_r = r.copy()
        for i in range(N):
            # Distance to walls
            max_r_wall = min(x[i], 1 - x[i], y[i], 1 - y[i])
            new_r[i] = min(new_r[i], max_r_wall)
            # Distance to other circles
            mask = (ii == i) | (jj == i)
            if np.any(mask):
                for k in np.where(mask)[0]:
                    other = jj[k] if ii[k] == i else ii[k]
                    d_ik = np.hypot(x[i] - x[other], y[i] - y[other])
                    available = (d_ik - r[other]) * 0.5
                    new_r[i] = min(new_r[i], max(0, available))
        
        # Only accept if it improves
        if np.sum(new_r) > np.sum(r) * 1.0001:
            r = new_r
        
        # Small position adjustments to reduce overlaps
        dx_all = x[ii] - x[jj]
        dy_all = y[ii] - y[jj]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all) + 1e-12
        s_all = r[ii] + r[jj]
        overlap = np.maximum(0, s_all - d_all)
        
        if np.sum(overlap) > 1e-10:
            f_mag = 50.0 * overlap / d_all
            fx = np.zeros(N)
            fy = np.zeros(N)
            np.add.at(fx, ii, f_mag * dx_all)
            np.add.at(fx, jj, -f_mag * dx_all)
            np.add.at(fy, ii, f_mag * dy_all)
            np.add.at(fy, jj, -f_mag * dy_all)
            
            # Wall push
            push_x = np.where(x < r, 50.0 * (r - x), 0) - np.where(x > 1 - r, 50.0 * (x + r - 1), 0)
            push_y = np.where(y < r, 50.0 * (r - y), 0) - np.where(y > 1 - r, 50.0 * (y + r - 1), 0)
            fx += push_x
            fy += push_y
            
            step = 0.005
            x += step * fx
            y += step * fy
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
        
        if it % 50 == 0 or it == max_iter - 1:
            arr2 = np.column_stack([x, y, r])
            arr2 = _shrink_to_valid(arr2)
            sc = _score(arr2)
            if sc > best_score:
                best_score = sc
                best = arr2.copy()
    
    return best, best_score

def construct():
    t0 = time.time()
    global _DEADLINE

    # baseline from hex seed
    base = _hex_seed()
    base = _shrink_to_valid(base)
    best = base.copy()
    best_score = _score(best)

    rng = np.random.default_rng(12345)

    restart = 0
    while time.time() < _DEADLINE - 20:
        # build a random-ish initial configuration
        x0 = base[:, 0].copy()
        y0 = base[:, 1].copy()
        r0 = base[:, 2].copy()
        # random jitter
        jit = 0.05
        x0 += rng.uniform(-jit, jit, N)
        y0 += rng.uniform(-jit, jit, N)
        x0 = np.clip(x0, 0.0, 1.0)
        y0 = np.clip(y0, 0.0, 1.0)
        # random initial radius scale
        scale = rng.uniform(0.7, 1.3)
        r0 = r0 * scale

        # randomize order / positions for diversity occasionally
        if restart % 5 == 0:
            perm = rng.permutation(N)
            x0 = x0[perm]; y0 = y0[perm]; r0 = r0[perm]

        arr, sc = _repulsion_optimize(
            x0, y0, r0,
            max_iter=3000,
            dt=0.02,
            grow=0.0006,
            wall_k=40.0,
            pair_k=80.0,
            max_r=0.25,
            seed=restart,
            deadline=_DEADLINE - 25
        )
        # shrink to valid
        arr = _shrink_to_valid(arr)
        sc = _score(arr)
        if sc > best_score:
            best_score = sc
            best = arr.copy()
        restart += 1

    # Final polish pass
    if time.time() < _DEADLINE - 5:
        polished, pol_score = _polish(best, max_iter=3000, deadline=_DEADLINE - 3)
        if pol_score > best_score:
            best_score = pol_score
            best = polished.copy()

    # final safety: ensure validity
    best = _shrink_to_valid(best)
    if not _validity_check(best):
        # fall back to a guaranteed-valid trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])

    return best
