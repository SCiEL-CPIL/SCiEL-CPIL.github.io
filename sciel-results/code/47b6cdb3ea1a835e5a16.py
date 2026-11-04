"""Circle packing in unit square: pack 26 circles, maximize sum of radii.

Strategy: combinatorial search over contact-graph topologies.
Each candidate topology assigns each circle a subset of wall contacts and a
set of circle-to-circle contacts.  The radii and positions are then the unique
solution of a system of nonlinear equations (distance = r_i + r_j for contacts,
x = r_i for left wall, etc.).  We enumerate many topologies (from structured
patterns + random perturbations) and solve them with a vectorized Newton method.
"""

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
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
            dist = np.sqrt(dx * dx + dy * dy)
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


def _newton_solve(walls, contacts, x0, max_iter=300, tol=1e-10):
    """
    Vectorized Newton solver for the contact-constraint system.

    walls: list of (i, wall) where wall in {0:L, 1:R, 2:B, 3:T}
    contacts: list of (i, j) circle-circle contacts
    x0: initial guess (2N + N) = (x, y, r)

    Returns (solution, converged, residual).
    """
    n = N
    m = 3 * n
    x = x0.copy()

    # Build sparse structure: which variables each equation involves
    # We'll just use dense Jacobian for simplicity (n=26, m<=~80)
    # Precompute equation list
    eqs = []
    for (i, w) in walls:
        eqs.append(('wall', i, w))
    for (i, j) in contacts:
        eqs.append(('contact', i, j))

    n_eq = len(eqs)
    if n_eq == 0:
        return x, True, 0.0

    for it in range(max_iter):
        # Evaluate residuals
        res = np.zeros(n_eq)
        for k, eq in enumerate(eqs):
            if eq[0] == 'wall':
                i, w = eq[1], eq[2]
                xi, yi, ri = x[2*i], x[2*i+1], x[2*i+n]
                if w == 0:   # left
                    res[k] = xi - ri
                elif w == 1: # right
                    res[k] = 1.0 - xi - ri
                elif w == 2: # bottom
                    res[k] = yi - ri
                else:        # top
                    res[k] = 1.0 - yi - ri
            else:
                i, j = eq[1], eq[2]
                xi, yi, ri = x[2*i], x[2*i+1], x[2*i+n]
                xj, yj, rj = x[2*j], x[2*j+1], x[2*j+n]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                res[k] = dist - ri - rj

        norm_res = np.linalg.norm(res)
        if norm_res < tol:
            return x, True, norm_res

        # Build Jacobian (dense, n_eq x m)
        J = np.zeros((n_eq, m))
        for k, eq in enumerate(eqs):
            if eq[0] == 'wall':
                i, w = eq[1], eq[2]
                if w == 0:   # x - r = 0
                    J[k, 2*i] = 1.0
                    J[k, 2*i+n] = -1.0
                elif w == 1: # 1 - x - r = 0
                    J[k, 2*i] = -1.0
                    J[k, 2*i+n] = -1.0
                elif w == 2: # y - r = 0
                    J[k, 2*i+1] = 1.0
                    J[k, 2*i+n] = -1.0
                else:        # 1 - y - r = 0
                    J[k, 2*i+1] = -1.0
                    J[k, 2*i+n] = -1.0
            else:
                i, j = eq[1], eq[2]
                xi, yi, ri = x[2*i], x[2*i+1], x[2*i+n]
                xj, yj, rj = x[2*j], x[2*j+1], x[2*j+n]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-15:
                    continue
                inv_d = 1.0 / dist
                # d(dist)/dx_i = dx/dist, d(dist)/dy_i = dy/dist
                # d(dist)/dx_j = -dx/dist, d(dist)/dy_j = -dy/dist
                # d(dist - r_i - r_j)/dr_i = -1, /dr_j = -1
                J[k, 2*i] = dx * inv_d
                J[k, 2*i+1] = dy * inv_d
                J[k, 2*i+n] = -1.0
                J[k, 2*j] = -dx * inv_d
                J[k, 2*j+1] = -dy * inv_d
                J[k, 2*j+n] = -1.0

        # Solve J dx = -res (least squares if overdetermined)
        try:
            dx_step, residuals, rank, sv = np.linalg.lstsq(J, -res, rcond=None)
        except np.linalg.LinAlgError:
            break

        # Damping to ensure progress
        step_scale = 1.0
        for _ in range(10):
            x_new = x + step_scale * dx_step
            # Quick check: all radii positive
            if np.all(x_new[2*n:] > 1e-6):
                x = x_new
                break
            step_scale *= 0.5
        else:
            break

    norm_res = np.linalg.norm(res)
    return x, norm_res < 1e-6, norm_res


def _check_packing_solution(x):
    """Check if a Newton solution is a valid packing. Returns (valid, total_radius)."""
    n = N
    xy = np.column_stack([x[:2*n:2], x[1:2*n:2]])
    r = x[2*n:]
    # Check radii positive
    if np.any(r < 1e-6):
        return False, 0.0
    # Check walls
    for i in range(n):
        xi, yi, ri = xy[i, 0], xy[i, 1], r[i]
        if xi - ri < -1e-7 or 1.0 - xi - ri < -1e-7 or yi - ri < -1e-7 or 1.0 - yi - ri < -1e-7:
            return False, 0.0
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-7:
                return False, 0.0
    return True, float(np.sum(r))


def _generate_topology_grid(side):
    """Generate a topology from a regular grid pattern."""
    walls = []
    contacts = []
    # Assign grid positions
    pos = {}
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx >= N:
                break
            pos[idx] = (i, j)

    # Wall contacts: leftmost column -> L, rightmost -> R, bottom row -> B, top -> T
    for idx, (i, j) in pos.items():
        if i == 0:
            walls.append((idx, 0))
        if i == side - 1:
            walls.append((idx, 1))
        if j == 0:
            walls.append((idx, 2))
        if j == side - 1:
            walls.append((idx, 3))

    # Circle contacts: neighbors in grid
    for idx, (i, j) in pos.items():
        # Right neighbor
        if (i+1, j) in pos:
            contacts.append((idx, pos[(i+1, j)]))
        # Up neighbor
        if (i, j+1) in pos:
            contacts.append((idx, pos[(i, j+1)]))

    return walls, contacts


def _generate_topology_hex(rng, n_circles=N):
    """Generate a hexagonal-ish topology with random contacts."""
    # Random positions for initial guess
    xy0 = rng.uniform(0.1, 0.9, size=(n_circles, 2))
    r0 = rng.uniform(0.03, 0.06, size=n_circles)

    # Assign wall contacts based on proximity to walls
    walls = []
    for i in range(n_circles):
        x, y = xy0[i]
        if x < 0.2:
            walls.append((i, 0))
        elif x > 0.8:
            walls.append((i, 1))
        if y < 0.2:
            walls.append((i, 2))
        elif y > 0.8:
            walls.append((i, 3))

    # Circle contacts: connect nearby pairs (k-nearest neighbors)
    contacts = []
    k = 3
    for i in range(n_circles):
        dists = np.sqrt((xy0[i,0]-xy0[:,0])**2 + (xy0[i,1]-xy0[:,1])**2)
        dists[i] = 1e9
        nn = np.argsort(dists)[:k]
        for j in nn:
            if i < j:
                contacts.append((i, j))
    # Deduplicate
    contacts = list(set(contacts))

    return walls, contacts, np.concatenate([xy0.ravel(), r0])


def _generate_topology_random(rng):
    """Fully random topology with random wall and circle contacts."""
    walls = []
    for i in range(N):
        # Each circle gets 1-2 wall contacts
        n_walls = rng.randint(1, 3)
        chosen = rng.choice(4, size=n_walls, replace=False)
        for w in chosen:
            walls.append((i, w))

    contacts = []
    # Random circle contacts: each circle connects to 2-4 others
    for i in range(N):
        n_contacts = rng.randint(2, 5)
        others = [j for j in range(N) if j != i]
        chosen = rng.choice(others, size=min(n_contacts, len(others)), replace=False)
        for j in chosen:
            if i < j:
                contacts.append((i, j))
    contacts = list(set(contacts))

    # Random initial guess
    xy0 = rng.uniform(0.1, 0.9, size=(N, 2))
    r0 = rng.uniform(0.03, 0.06, size=N)
    x0 = np.concatenate([xy0.ravel(), r0])

    return walls, contacts, x0


def _local_search_polish(xy, r, deadline, grid_size=80):
    """Best-fit local search to polish a solution."""
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

    max_sweeps = 20
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


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    rng = np.random.RandomState(42)

    # Phase 1: Structured grid topologies (side 4, 5, 6)
    for side in [4, 5, 6]:
        if time.time() > deadline:
            break
        walls, contacts = _generate_topology_grid(side)

        # Initial guess: uniform grid
        x0 = np.zeros(3 * N)
        for i in range(N):
            gi = i % side
            gj = i // side
            x0[2*i] = (gi + 0.5) / side
            x0[2*i+1] = (gj + 0.5) / side
            x0[2*i+N] = 0.45 / side

        sol, converged, res = _newton_solve(walls, contacts, x0, max_iter=200)
        valid, total = _check_packing_solution(sol)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = np.column_stack([sol[:2*N:2], sol[1:2*N:2]])
            best_r = sol[2*N:].copy()

    # Phase 2: Hex-like topologies with random perturbations
    n_hex_attempts = 15
    for attempt in range(n_hex_attempts):
        if time.time() > deadline:
            break
        walls, contacts, x0 = _generate_topology_hex(rng)
        sol, converged, res = _newton_solve(walls, contacts, x0, max_iter=200)
        valid, total = _check_packing_solution(sol)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = np.column_stack([sol[:2*N:2], sol[1:2*N:2]])
            best_r = sol[2*N:].copy()

    # Phase 3: Random topologies
    n_random_attempts = 20
    for attempt in range(n_random_attempts):
        if time.time() > deadline:
            break
        walls, contacts, x0 = _generate_topology_random(rng)
        sol, converged, res = _newton_solve(walls, contacts, x0, max_iter=150)
        valid, total = _check_packing_solution(sol)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = np.column_stack([sol[:2*N:2], sol[1:2*N:2]])
            best_r = sol[2*N:].copy()

    # Phase 4: Local search polish on best result
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        ls_xy, ls_r, ls_total, ls_valid = _local_search_polish(
            best_xy.copy(), best_r.copy(), deadline, grid_size=80)
        if ls_valid and ls_total > best_total + 1e-10:
            best_total = ls_total
            best_valid = True
            best_xy = ls_xy
            best_r = ls_r

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
