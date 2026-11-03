"""Circle packing in unit square: 26 circles, maximize sum of radii.
Uses a Hertzian-contact (soft-potential) rigid-body simulation with symplectic
integration to find jammed configurations, combined with multi-start and a
final analytic polish. The physical representation naturally enforces non-overlap
through energy minimization and momentum conservation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _validate(circles, tol=1e-7):
    """Check that circles are valid: inside unit square, non-overlapping, positive radii."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _fix_boundaries(circles):
    """Project circles back into valid configuration by shrinking radii and clipping positions."""
    circles = circles.copy()
    for i in range(N):
        r = max(0.0, circles[i, 2])
        r = min(r, 0.5)
        circles[i, 2] = r
        circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)
    for _iter in range(30):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2.0 + 1e-6
                    circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                    circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
                    circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], 1.0 - circles[j, 2])
                    circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], 1.0 - circles[j, 2])
                    changed = True
        if not changed:
            break
    return circles


def _hertzian_forces(circles, k_contact=100.0, k_boundary=200.0, power=1.5):
    """
    Compute Hertzian contact forces. For overlapping circles, the repulsive force
    is proportional to delta^(3/2) where delta is the penetration depth.
    For boundary violations, similar spring forces push circles back inside.
    """
    n = len(circles)
    forces = np.zeros((n, 2))
    radii_forces = np.zeros(n)

    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Boundary forces
    for i in range(n):
        # Left wall
        pen = r[i] - x[i]
        if pen > 0:
            f = k_boundary * (pen ** power)
            forces[i, 0] += f
        # Right wall
        pen = x[i] + r[i] - 1.0
        if pen > 0:
            f = k_boundary * (pen ** power)
            forces[i, 0] -= f
        # Bottom wall
        pen = r[i] - y[i]
        if pen > 0:
            f = k_boundary * (pen ** power)
            forces[i, 1] += f
        # Top wall
        pen = y[i] + r[i] - 1.0
        if pen > 0:
            f = k_boundary * (pen ** power)
            forces[i, 1] -= f

    # Circle-circle contact forces
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-10:
                # Perturb slightly to avoid singularity
                dx = np.random.uniform(-1e-6, 1e-6)
                dy = np.random.uniform(-1e-6, 1e-6)
                dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            pen = min_dist - dist
            if pen > 0:
                # Hertzian force: F = k * pen^(3/2), directed along center line
                f_mag = k_contact * (pen ** power)
                ux = dx / dist
                uy = dy / dist
                forces[i, 0] += f_mag * ux
                forces[i, 1] += f_mag * uy
                forces[j, 0] -= f_mag * ux
                forces[j, 1] -= f_mag * uy
                # Also apply a radial growth pressure
                radii_forces[i] += 0.1
                radii_forces[j] += 0.1

    return forces, radii_forces


def _simulate(circles, n_steps=5000, dt=0.002, damping=0.98, growth_rate=0.0005, deadline=None):
    """
    Symplectic (velocity-Verlet) simulation with Hertzian contacts.
    Circles have unit mass. Damping removes energy to help convergence.
    A small growth pressure on radii drives the system to expand circles.
    """
    n = len(circles)
    pos = circles[:, :2].copy()
    vel = np.zeros((n, 2))
    radii = circles[:, 2].copy()
    rad_vel = np.zeros(n)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 2:
            break

        # Compute forces
        circles_tmp = np.column_stack([pos, radii])
        forces, radii_forces = _hertzian_forces(circles_tmp)

        # Add growth pressure to radii (small constant push to increase radii)
        radii_forces += growth_rate

        # Velocity Verlet integration
        # Half-step velocity update
        vel += 0.5 * dt * forces
        rad_vel += 0.5 * dt * radii_forces

        # Position update
        pos += dt * vel
        radii += dt * rad_vel

        # Damping
        vel *= damping
        rad_vel *= damping

        # Second half-step velocity update (recompute forces at new position)
        circles_tmp2 = np.column_stack([pos, radii])
        forces2, radii_forces2 = _hertzian_forces(circles_tmp2)
        radii_forces2 += growth_rate
        vel += 0.5 * dt * forces2
        rad_vel += 0.5 * dt * radii_forces2

        # Clip to boundaries
        for i in range(n):
            radii[i] = np.clip(radii[i], 0.01, 0.5)
            pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
            pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])

        # Check validity and update best
        if step % 50 == 0:
            circles_check = np.column_stack([pos, radii])
            if _validate(circles_check):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()

    return np.column_stack([best_pos, best_radii]), best_score


def _polish_analytic(circles, deadline):
    """
    Final analytic polish using L-BFGS-B with a penalty-based objective.
    This refines the configuration found by the simulation.
    """
    n = len(circles)
    params = circles.flatten().copy()

    def _objective(p):
        x = p[0::3]
        y = p[1::3]
        r = p[2::3]
        penalty = 0.0
        # Boundary penalties
        penalty += np.sum(np.maximum(r - x, 0)**2)
        penalty += np.sum(np.maximum(x + r - 1.0, 0)**2)
        penalty += np.sum(np.maximum(r - y, 0)**2)
        penalty += np.sum(np.maximum(y + r - 1.0, 0)**2)
        penalty += np.sum(np.maximum(-r, 0)**2)
        # Overlap penalties
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist = np.sqrt(np.maximum(dx*dx + dy*dy, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(np.maximum(overlap, 0)**2)
        return -np.sum(r) + 500.0 * penalty

    def _grad(p):
        x = p[0::3]
        y = p[1::3]
        r = p[2::3]
        grad = np.zeros_like(p)
        grad[2::3] = -1.0

        # Boundary gradients
        mask_lx = (r - x) > 0
        grad[0::3][mask_lx] += 2.0 * 500.0 * (r[mask_lx] - x[mask_lx]) * (-1)
        grad[2::3][mask_lx] += 2.0 * 500.0 * (r[mask_lx] - x[mask_lx])
        mask_ux = (x + r - 1.0) > 0
        grad[0::3][mask_ux] += 2.0 * 500.0 * (x[mask_ux] + r[mask_ux] - 1.0)
        grad[2::3][mask_ux] += 2.0 * 500.0 * (x[mask_ux] + r[mask_ux] - 1.0)
        mask_ly = (r - y) > 0
        grad[1::3][mask_ly] += 2.0 * 500.0 * (r[mask_ly] - y[mask_ly]) * (-1)
        grad[2::3][mask_ly] += 2.0 * 500.0 * (r[mask_ly] - y[mask_ly])
        mask_uy = (y + r - 1.0) > 0
        grad[1::3][mask_uy] += 2.0 * 500.0 * (y[mask_uy] + r[mask_uy] - 1.0)
        grad[2::3][mask_uy] += 2.0 * 500.0 * (y[mask_uy] + r[mask_uy] - 1.0)
        mask_nr = (-r) > 0
        grad[2::3][mask_nr] += 2.0 * 500.0 * (-r[mask_nr]) * (-1)

        # Overlap gradients
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist = np.sqrt(np.maximum(dx*dx + dy*dy, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        mask_ov = overlap > 0
        if np.any(mask_ov):
            inv_dist = 1.0 / dist[mask_ov]
            f = 2.0 * 500.0 * overlap[mask_ov]
            ii = idx_i[mask_ov]
            jj = idx_j[mask_ov]
            np.add.at(grad[0::3], ii, -f * dx[mask_ov] * inv_dist)
            np.add.at(grad[0::3], jj, f * dx[mask_ov] * inv_dist)
            np.add.at(grad[1::3], ii, -f * dy[mask_ov] * inv_dist)
            np.add.at(grad[1::3], jj, f * dy[mask_ov] * inv_dist)
            np.add.at(grad[2::3], ii, f)
            np.add.at(grad[2::3], jj, f)

        return grad

    bounds = [(0, 1), (0, 1), (0, 0.5)] * N

    try:
        res = minimize(_objective, params, jac=_grad, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.fun < _objective(params):
            circles_out = np.column_stack([res.x[0::3], res.x[1::3], res.x[2::3]])
            if _validate(circles_out):
                return circles_out
    except Exception:
        pass

    return circles


def _initial_hex():
    """Hexagonal packing initial guess."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial guess."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.85
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_corner_weighted(seed=0):
    """Circles near corners and edges."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])


def _initial_voronoi(seed=0):
    """Voronoi-based initial guess with radii from nearest neighbors."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    radii = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([points[i, 0], 1 - points[i, 0], points[i, 1], 1 - points[i, 1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
    return np.column_stack([points[:, 0], points[:, 1], radii])


def _initial_random(seed=0):
    """Pure random initial guess."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return np.column_stack([positions[:, 0], positions[:, 1], radii])


def construct():
    start_time = time.time()
    deadline = _start_time + _TIME_LIMIT

    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_corner_weighted(seed=123),
        _initial_voronoi(seed=7),
        _initial_voronoi(seed=1234),
        _initial_random(seed=99),
        _initial_random(seed=2024),
    ]

    best_overall = None
    best_score_overall = -np.inf

    n_inits = len(inits)
    # Allocate time: first 60% to simulation, last 40% to polish
    sim_time_per_init = 0.6 * (_TIME_LIMIT - 10) / n_inits
    polish_time_per_init = 0.4 * (_TIME_LIMIT - 10) / n_inits

    for i, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 5:
            break

        # Phase 1: Hertzian simulation
        sim_deadline = time.time() + min(sim_time_per_init, remaining - 3)
        n_steps = max(1000, int(sim_time_per_init / 0.0001))  # estimate steps
        n_steps = min(n_steps, 8000)  # cap

        try:
            circles_sim, score_sim = _simulate(xyz0, n_steps=n_steps, dt=0.002,
                                               damping=0.98, growth_rate=0.0005,
                                               deadline=sim_deadline)
        except Exception:
            circles_sim = xyz0.copy()
            score_sim = np.sum(xyz0[:, 2])

        # Fix boundaries
        circles_sim = _fix_boundaries(circles_sim)
        if not _validate(circles_sim):
            circles_sim = xyz0.copy()

        # Phase 2: Analytic polish
        remaining = deadline - time.time()
        if remaining > 3:
            polish_deadline = time.time() + min(polish_time_per_init, remaining - 1)
            try:
                circles_polished = _polish_analytic(circles_sim, polish_deadline)
                if _validate(circles_polished):
                    score_polished = np.sum(circles_polished[:, 2])
                    if score_polished > score_sim:
                        circles_sim = circles_polished
                        score_sim = score_polished
            except Exception:
                pass

        # Update best
        if _validate(circles_sim):
            score = np.sum(circles_sim[:, 2])
            if score > best_score_overall:
                best_score_overall = score
                best_overall = circles_sim.copy()

    # Final polish on the best result
    if best_overall is not None:
        remaining = deadline - time.time()
        if remaining > 3:
            try:
                circles_final = _polish_analytic(best_overall, deadline - 1)
                if _validate(circles_final):
                    score_final = np.sum(circles_final[:, 2])
                    if score_final > best_score_overall:
                        best_overall = circles_final.copy()
                        best_score_overall = score_final
            except Exception:
                pass

    if best_overall is not None:
        repaired = _fix_boundaries(best_overall)
        if _validate(repaired):
            best_overall = repaired
        else:
            best_overall = None

    if best_overall is None:
        result = np.zeros((N, 3))
        side = 5
        for idx in range(N):
            i = idx % side
            j = idx // side
            result[idx, 0] = (i + 0.5) / side
            result[idx, 1] = (j + 0.5) / side
            result[idx, 2] = 0.5 / side * 0.8
        return result

    return best_overall

# EVOLVE-BLOCK-END
