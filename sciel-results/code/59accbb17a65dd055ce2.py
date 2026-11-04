"""Circle packing in unit square: pack 26 circles, maximize sum of radii.

New strategy: continuous physics-based force-directed packing.
- Robust greedy initialization using largest-empty-circle (Voronoi + boundary sampling)
- Force-directed evolution: repulsive disk forces + wall forces, symplectic-like integrator
  with adaptive velocity damping, evolving until forces vanish (equilibrium packing)
- Multi-start L-BFGS-B on a smooth objective (penalized overlap + wall violations) to polish
- Multiple random restarts to escape local minima
"""

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _grid_init(side):
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(rng, n=N):
    """Random initial positions and small radii."""
    xy = rng.uniform(0.05, 0.95, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _largest_empty_circle_greedy(rng, n=N, n_candidates=2000):
    """
    Greedy largest-empty-circle initialization.
    Repeatedly place a circle at the point with maximum clearance to existing
    circles and walls. Uses a candidate set of grid points + random points +
    Voronoi-like vertices (midpoints between pairs).
    """
    xy = np.zeros((n, 2))
    r = np.zeros(n)

    # Build candidate set
    grid = np.linspace(0.0, 1.0, 60)
    GX, GY = np.meshgrid(grid, grid, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    # Add random candidates
    rand_pts = rng.uniform(0.0, 1.0, size=(n_candidates, 2))
    candidates = np.vstack([candidates, rand_pts])
    # Add midpoints between random pairs (Voronoi-like)
    for _ in range(500):
        a = rng.randint(0, len(candidates))
        b = rng.randint(0, len(candidates))
        mid = 0.5 * (candidates[a] + candidates[b])
        candidates = np.vstack([candidates, mid.reshape(1, 2)])

    M = len(candidates)
    # Precompute wall clearance for each candidate
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    placed_xy = []
    placed_r = []

    for k in range(n):
        if len(placed_xy) == 0:
            # First circle: max wall clearance
            idx = int(np.argmax(wall_clear))
            best_r = wall_clear[idx]
            best_xy = candidates[idx]
        else:
            placed_arr = np.array(placed_xy)
            placed_r_arr = np.array(placed_r)
            # Distance from each candidate to each placed circle
            dx = candidates[:, 0:1] - placed_arr[:, 0][np.newaxis, :]
            dy = candidates[:, 1:2] - placed_arr[:, 1][np.newaxis, :]
            dists = np.sqrt(dx * dx + dy * dy)
            circle_clear = np.min(dists - placed_r_arr[np.newaxis, :], axis=1)
            r_candidates = np.minimum(wall_clear, circle_clear)
            # Only consider positive clearance
            valid_mask = r_candidates > 1e-6
            if not np.any(valid_mask):
                # Fallback: place at random point with tiny radius
                idx = rng.randint(M)
                best_r = max(0.005, wall_clear[idx])
                best_xy = candidates[idx]
            else:
                r_candidates[~valid_mask] = -1.0
                idx = int(np.argmax(r_candidates))
                best_r = r_candidates[idx]
                best_xy = candidates[idx]

        xy[k] = best_xy
        r[k] = best_r
        placed_xy.append(best_xy)
        placed_r.append(best_r)

    return xy, r


def _force_directed(xy, r, deadline, dt=0.005, max_iter=5000, damping=0.95):
    """
    Force-directed evolution: circles repel each other (disk overlap penalty)
    and are attracted to stay inside the square. Uses a symplectic-like
    integrator with adaptive velocity damping.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    vel = np.zeros_like(xy)

    # Force parameters
    k_repel = 50.0    # repulsion strength for overlapping circles
    k_wall = 100.0    # wall penalty strength
    k_center = 0.5    # weak attraction to center to prevent drift

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Compute forces
        forces = np.zeros_like(xy)

        # Circle-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    # Random nudge
                    dx = rng.uniform(-1e-6, 1e-6)
                    dy = rng.uniform(-1e-6, 1e-6)
                    dist = math.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    fx = k_repel * overlap * dx / dist
                    fy = k_repel * overlap * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Wall forces
        for i in range(n):
            # Left wall
            if xy[i, 0] - r[i] < 0:
                forces[i, 0] += k_wall * (r[i] - xy[i, 0])
            # Right wall
            if 1.0 - xy[i, 0] - r[i] < 0:
                forces[i, 0] -= k_wall * (xy[i, 0] + r[i] - 1.0)
            # Bottom wall
            if xy[i, 1] - r[i] < 0:
                forces[i, 1] += k_wall * (r[i] - xy[i, 1])
            # Top wall
            if 1.0 - xy[i, 1] - r[i] < 0:
                forces[i, 1] -= k_wall * (xy[i, 1] + r[i] - 1.0)

            # Weak center attraction
            forces[i, 0] -= k_center * (xy[i, 0] - 0.5)
            forces[i, 1] -= k_center * (xy[i, 1] - 0.5)

        # Update velocities with damping
        vel = vel * damping + forces * dt
        # Limit velocity
        speed = np.sqrt(np.sum(vel ** 2, axis=1, keepdims=True))
        max_speed = 0.05
        too_fast = speed > max_speed
        if np.any(too_fast):
            vel[too_fast] *= max_speed / speed[too_fast]

        # Update positions
        xy += vel * dt

        # Project back into square (hard constraint)
        for i in range(n):
            ri = r[i]
            xy[i, 0] = np.clip(xy[i, 0], ri, 1.0 - ri)
            xy[i, 1] = np.clip(xy[i, 1], ri, 1.0 - ri)

        # Check convergence: max force small
        max_force = np.max(np.sqrt(np.sum(forces ** 2, axis=1)))
        if max_force < 1e-5:
            break

    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=80, max_sweeps=15):
    """
    Best-fit local search: for each circle, find the grid position that
    maximizes its radius given all other circles fixed. Iterate sweeps.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _smooth_objective(params, n=N):
    """
    Smooth objective for L-BFGS-B: minimize penalty for overlaps and wall violations.
    params: flat array [x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}, r0, r1, ..., r_{n-1}]
    Returns negative sum of radii minus penalties (so minimizing = maximizing radii).
    """
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]

    penalty = 0.0
    # Wall violations
    for i in range(n):
        ri = r[i]
        if xy[i, 0] - ri < 0:
            penalty += 1000.0 * (xy[i, 0] - ri) ** 2
        if 1.0 - xy[i, 0] - ri < 0:
            penalty += 1000.0 * (1.0 - xy[i, 0] - ri) ** 2
        if xy[i, 1] - ri < 0:
            penalty += 1000.0 * (xy[i, 1] - ri) ** 2
        if 1.0 - xy[i, 1] - ri < 0:
            penalty += 1000.0 * (1.0 - xy[i, 1] - ri) ** 2

    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += 1000.0 * overlap ** 2

    # Small penalty for very small radii (encourage growth)
    penalty += 0.01 * np.sum(np.maximum(0.0, 0.01 - r) ** 2)

    return penalty - np.sum(r)


def _smooth_grad(params, n=N):
    """Gradient of _smooth_objective."""
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]
    grad = np.zeros_like(params)

    # Wall gradient contributions
    for i in range(n):
        ri = r[i]
        # Left wall
        viol = xy[i, 0] - ri
        if viol < 0:
            grad[2*i] += 2 * 1000.0 * viol
            grad[2*n+i] -= 2 * 1000.0 * viol
        # Right wall
        viol = 1.0 - xy[i, 0] - ri
        if viol < 0:
            grad[2*i] -= 2 * 1000.0 * viol
            grad[2*n+i] -= 2 * 1000.0 * viol
        # Bottom wall
        viol = xy[i, 1] - ri
        if viol < 0:
            grad[2*i+1] += 2 * 1000.0 * viol
            grad[2*n+i] -= 2 * 1000.0 * viol
        # Top wall
        viol = 1.0 - xy[i, 1] - ri
        if viol < 0:
            grad[2*i+1] -= 2 * 1000.0 * viol
            grad[2*n+i] -= 2 * 1000.0 * viol

    # Overlap gradient contributions
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                continue
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                inv_d = 1.0 / dist
                # d(overlap^2)/dx_i = 2*overlap * d(overlap)/dx_i = 2*overlap * (dx/dist)
                g = 2 * 1000.0 * overlap
                grad[2*i] += g * dx * inv_d
                grad[2*i+1] += g * dy * inv_d
                grad[2*j] -= g * dx * inv_d
                grad[2*j+1] -= g * dy * inv_d
                grad[2*n+i] -= g
                grad[2*n+j] -= g

    # Radius gradient: -1 for each radius (from -sum(r))
    grad[2*n:] -= 1.0
    # Small radius penalty gradient
    for i in range(n):
        if r[i] < 0.01:
            grad[2*n+i] += 2 * 0.01 * (0.01 - r[i])

    return grad


def _gradient_polish(xy, r, deadline, n_starts=3):
    """
    Multi-start L-BFGS-B on the smooth objective to polish the packing.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_val, best_valid = _eval_packing(best_xy, best_r)
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    rng = np.random.RandomState(123)

    for start in range(n_starts):
        if time.time() > deadline:
            break

        # Perturb initial guess
        if start == 0:
            xy0 = xy.copy()
            r0 = r.copy()
        else:
            xy0 = xy + rng.uniform(-0.02, 0.02, size=xy.shape)
            r0 = np.maximum(0.005, r + rng.uniform(-0.01, 0.01, size=r.shape))

        # Project into square
        for i in range(n):
            ri = r0[i]
            xy0[i, 0] = np.clip(xy0[i, 0], ri, 1.0 - ri)
            xy0[i, 1] = np.clip(xy0[i, 1], ri, 1.0 - ri)

        params0 = np.concatenate([xy0.ravel(), r0])

        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
        bounds += [(0.001, 0.5)] * n  # radii

        try:
            res = minimize(_smooth_objective, params0,
                           method='L-BFGS-B',
                           jac=_smooth_grad,
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            continue

        if res.success or res.fun < 1e-6:
            params_sol = res.x
            xy_sol = params_sol[:2*n].reshape(n, 2)
            r_sol = params_sol[2*n:]

            # Validate
            total, valid = _eval_packing(xy_sol, r_sol)
            if valid and total > best_total + 1e-10:
                best_total = total
                best_valid = True
                best_xy = xy_sol.copy()
                best_r = r_sol.copy()

    return best_xy, best_r, best_total, best_valid


def _high_precision_polish(xy, r, deadline, grid_size=150, max_sweeps=10):
    """High-resolution best-fit local search for final precision."""
    return _best_fit_local_search(xy, r, deadline, grid_size=grid_size, max_sweeps=max_sweeps)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    rng = np.random.RandomState(42)

    # Phase 1: Greedy largest-empty-circle initialization + force-directed evolution
    xy_greedy, r_greedy = _largest_empty_circle_greedy(rng, n=N)
    xy_fd, r_fd = _force_directed(xy_greedy, r_greedy, deadline - 100.0)
    total, valid = _eval_packing(xy_fd, r_fd)
    if valid and total > best_total:
        best_total = total
        best_valid = True
        best_xy = xy_fd.copy()
        best_r = r_fd.copy()

    # Phase 2: Gradient polish (L-BFGS-B multi-start)
    if best_valid and time.time() < deadline - 30:
        gp_xy, gp_r, gp_total, gp_valid = _gradient_polish(
            best_xy.copy(), best_r.copy(), deadline - 20, n_starts=3)
        if gp_valid and gp_total > best_total + 1e-10:
            best_total = gp_total
            best_valid = True
            best_xy = gp_xy
            best_r = gp_r

    # Phase 3: Best-fit local search
    if best_valid and time.time() < deadline - 15:
        ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(
            best_xy.copy(), best_r.copy(), deadline - 10, grid_size=100, max_sweeps=10)
        if ls_valid and ls_total > best_total + 1e-10:
            best_total = ls_total
            best_valid = True
            best_xy = ls_xy
            best_r = ls_r

    # Phase 4: More random restarts with force-directed + polish
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break

        # Random init
        xy_rand, r_rand = _random_init(rng, n=N)

        # Force-directed evolution
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        xy_fd_r, r_fd_r = _force_directed(xy_rand, r_rand, deadline - 20)

        # Quick best-fit
        ls_xy_r, ls_r_r, ls_total_r, ls_valid_r = _best_fit_local_search(
            xy_fd_r, r_fd_r, deadline - 20, grid_size=60, max_sweeps=5)

        if ls_valid_r and ls_total_r > best_total + 1e-10:
            best_total = ls_total_r
            best_valid = True
            best_xy = ls_xy_r
            best_r = ls_r_r

            # Gradient polish on improved solution
            if time.time() < deadline - 10:
                gp_xy_r, gp_r_r, gp_total_r, gp_valid_r = _gradient_polish(
                    best_xy.copy(), best_r.copy(), deadline - 10, n_starts=2)
                if gp_valid_r and gp_total_r > best_total + 1e-10:
                    best_total = gp_total_r
                    best_valid = True
                    best_xy = gp_xy_r
                    best_r = gp_r_r

    # Phase 5: High-precision polish
    if best_valid and time.time() < deadline - 5:
        hp_xy, hp_r, hp_total, hp_valid = _high_precision_polish(
            best_xy.copy(), best_r.copy(), deadline, grid_size=200, max_sweeps=5)
        if hp_valid and hp_total > best_total + 1e-10:
            best_total = hp_total
            best_valid = True
            best_xy = hp_xy
            best_r = hp_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
