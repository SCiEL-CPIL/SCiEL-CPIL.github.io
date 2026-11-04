"""Circle packing in unit square: force-directed repulsion strategy."""
import numpy as np
import time

N = 26


def _eval_packing(xy, r):
    """Return (sum_radii, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9:
            return 0.0, False
        if y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            if np.sqrt(dx * dx + dy * dy) < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _hex_seed():
    """Generate a hexagonal close-packed seed for 26 circles."""
    pts = []
    # Hexagonal lattice with spacing ~ 1/5
    a = 0.19  # half-spacing
    rows = 7
    for row in range(rows):
        y = 0.08 + row * 0.135
        if y > 0.95:
            break
        offset = a if row % 2 == 1 else 0.0
        x = 0.10 + offset
        while x < 0.95 and len(pts) < N:
            pts.append([x, y])
            x += 2 * a
    # Pad if needed
    while len(pts) < N:
        pts.append([0.5, 0.5])
    pts = np.array(pts[:N], dtype=np.float64)
    # Initial radii: small
    r = np.full(N, 0.02)
    return pts, r


def _force_directed(xy, r, deadline, n_iters=20000):
    """
    Force-directed optimization:
    - Pressure force pushes each radius up.
    - Repulsive force between circles: F = k / d^2 (capped), along the separation vector.
    - Wall repulsion: push away from walls when circle gets close.
    - Damping / velocity Verlet-like update.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    vel_xy = np.zeros_like(xy)
    vel_r = np.zeros(n)

    pressure = 0.0003       # upward force on each radius
    repulse_k = 0.008       # repulsion strength
    wall_k = 0.05           # wall repulsion strength
    damping = 0.85          # velocity damping
    max_force = 0.05        # cap per-pair force
    min_d = 0.005           # minimum distance to avoid singularity

    best_total = 0.0
    best_xy = xy.copy()
    best_r = r.copy()

    for it in range(n_iters):
        if time.time() > deadline:
            break

        # Reset forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)

        # Pressure: push radii up
        fr += pressure

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < min_d:
                    d = min_d
                    # Random nudge
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-6:
                        d = 1e-6

                # Repulsion magnitude (stronger when closer than r_i + r_j)
                overlap_thresh = r[i] + r[j]
                if d < overlap_thresh * 1.5:
                    # Force ~ k / d^2, directed apart
                    f_mag = repulse_k / (d * d)
                    if f_mag > max_force:
                        f_mag = max_force
                    ux = dx / d
                    uy = dy / d
                    fx[i] += f_mag * ux
                    fy[i] += f_mag * uy
                    fx[j] -= f_mag * ux
                    fy[j] -= f_mag * uy
                    # Also push radii apart (reduce both slightly)
                    fr[i] -= 0.5 * f_mag * 0.1
                    fr[j] -= 0.5 * f_mag * 0.1

        # Wall repulsion
        for i in range(n):
            ri = r[i]
            # Left wall
            if xy[i, 0] < ri:
                fx[i] += wall_k * (ri - xy[i, 0])
            # Right wall
            if 1.0 - xy[i, 0] < ri:
                fx[i] -= wall_k * (ri - (1.0 - xy[i, 0]))
            # Bottom
            if xy[i, 1] < ri:
                fy[i] += wall_k * (ri - xy[i, 1])
            # Top
            if 1.0 - xy[i, 1] < ri:
                fy[i] -= wall_k * (ri - (1.0 - xy[i, 1]))

        # Update velocities with damping
        vel_xy[:, 0] = damping * vel_xy[:, 0] + fx
        vel_xy[:, 1] = damping * vel_xy[:, 1] + fy
        vel_r = damping * vel_r + fr

        # Update positions
        xy[:, 0] += vel_xy[:, 0]
        xy[:, 1] += vel_xy[:, 1]
        r += vel_r

        # Enforce bounds
        for i in range(n):
            r[i] = max(r[i], 0.005)
            r[i] = min(r[i], 0.45)
            xy[i, 0] = np.clip(xy[i, 0], 0.0, 1.0)
            xy[i, 1] = np.clip(xy[i, 1], 0.0, 1.0)
            # Keep inside
            r[i] = min(r[i], xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            r[i] = max(r[i], 0.001)

        # Periodically check validity
        if it % 200 == 0 or it == n_iters - 1:
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total:
                best_total = total
                best_xy = xy.copy()
                best_r = r.copy()

    # Final check
    total, valid = _eval_packing(xy, r)
    if valid and total > best_total:
        best_total = total
        best_xy = xy.copy()
        best_r = r.copy()

    return best_xy, best_r, best_total


def _coordinate_descent_polish(xy, r, deadline):
    """
    Lightweight coordinate descent: for each circle, try to increase its radius
    by checking if it can grow without overlapping neighbors or walls.
    Also try small position shifts to allow growth.
    """
    n = len(xy)
    best_total, _ = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()

    for sweep in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Try to grow radius
            # Max possible from walls
            max_r_wall = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            # Max possible from neighbors
            max_r_neighbor = max_r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                available = d - r[j]
                if available < max_r_neighbor:
                    max_r_neighbor = available
            max_r_neighbor = max(max_r_neighbor, 0.001)

            new_r = min(max_r_neighbor, max_r_wall)
            if new_r > r[i] + 1e-8:
                r[i] = new_r
                improved = True

        if not improved:
            # Try small random perturbations to escape local optima
            for i in range(n):
                if time.time() > deadline:
                    break
                for _ in range(3):
                    old_x, old_y, old_r = xy[i, 0], xy[i, 1], r[i]
                    xy[i, 0] += np.random.uniform(-0.02, 0.02)
                    xy[i, 1] += np.random.uniform(-0.02, 0.02)
                    xy[i, 0] = np.clip(xy[i, 0], 0.0, 1.0)
                    xy[i, 1] = np.clip(xy[i, 1], 0.0, 1.0)
                    # Recompute max radius
                    max_r_wall = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                    max_r_neighbor = max_r_wall
                    for j in range(n):
                        if i == j:
                            continue
                        dx = xy[i, 0] - xy[j, 0]
                        dy = xy[i, 1] - xy[j, 1]
                        d = np.sqrt(dx * dx + dy * dy)
                        available = d - r[j]
                        if available < max_r_neighbor:
                            max_r_neighbor = available
                    max_r_neighbor = max(max_r_neighbor, 0.001)
                    new_r = min(max_r_neighbor, max_r_wall)
                    if new_r > old_r + 1e-8:
                        r[i] = new_r
                        improved = True
                        break
                    else:
                        # Revert
                        xy[i, 0], xy[i, 1], r[i] = old_x, old_y, old_r

        # Check validity
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total


def _scipy_polish(xy, r, deadline):
    """
    Final polish using scipy minimize with a smooth penalty,
    starting from the force-directed result.
    """
    from scipy.optimize import minimize

    n = len(xy)

    def objective(params):
        p = np.asarray(params)
        pxy = p[0::3].reshape(-1, 2)
        pr = np.maximum(p[1::3], 0.001)
        # Wall penalty
        pen = 0.0
        for i in range(n):
            pen += max(0.0, pr[i] - pxy[i, 0]) ** 2
            pen += max(0.0, pr[i] - (1.0 - pxy[i, 0])) ** 2
            pen += max(0.0, pr[i] - pxy[i, 1]) ** 2
            pen += max(0.0, pr[i] - (1.0 - pxy[i, 1])) ** 2
        # Overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                dx = pxy[i, 0] - pxy[j, 0]
                dy = pxy[i, 1] - pxy[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < pr[i] + pr[j]:
                    pen += (pr[i] + pr[j] - d) ** 2
        return -np.sum(pr) + 1000.0 * pen

    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.45))

    best_total, _ = _eval_packing(xy, r)
    best_params = params.copy()

    try:
        res = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10})
        opt_params = res.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_params = opt_params.copy()
    except Exception:
        pass

    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    return final_xy, final_r, best_total


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin

    # Phase 1: Force-directed relaxation from hex seed
    xy, r = _hex_seed()
    fd_xy, fd_r, fd_total = _force_directed(xy, r, deadline - 30.0, n_iters=15000)

    # Phase 2: Coordinate descent polish
    cd_xy, cd_r, cd_total = _coordinate_descent_polish(fd_xy, fd_r, deadline - 15.0)

    # Phase 3: SciPy polish if time remains
    if time.time() < deadline - 10.0:
        sp_xy, sp_r, sp_total = _scipy_polish(cd_xy, cd_r, deadline - 3.0)
        if sp_total > cd_total:
            cd_xy, cd_r, cd_total = sp_xy, sp_r, sp_total

    # Final validation and clipping
    n = len(cd_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = cd_xy[i, 0], cd_xy[i, 1], cd_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        result[i, 2] = ri

    # Final validity check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
