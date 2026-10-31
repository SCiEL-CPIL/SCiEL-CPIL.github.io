"""Circle packing in unit square - 26 circles maximizing sum of radii.
Uses Voronoi-based radius assignment, hexagonal close-packing initialization,
and multi-stage local search with force-directed annealing."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth penalties."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    eps = 1e-8

    # Boundary violations - smooth soft-max penalty
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bot_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)

    def _softmax(v):
        return (v + np.sqrt(v * v + eps)) * 0.5

    bpen = (np.sum(_softmax(left_v) ** 2) + np.sum(_softmax(right_v) ** 2) +
            np.sum(_softmax(bot_v) ** 2) + np.sum(_softmax(top_v) ** 2))

    # Overlap violations
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)
    viol_s = _softmax(viol)
    np.fill_diagonal(viol_s, 0.0)
    open_pen = 0.5 * np.sum(viol_s ** 2)

    return -np.sum(radii) + 100.0 * (bpen + open_pen)


def _voronoi_radii(pos, deadline):
    """Compute maximum radii using Voronoi diagram."""
    pos = pos.copy()
    n = len(pos)

    # Add boundary points to ensure Voronoi cells are bounded
    boundary_pts = np.array([
        [-0.5, -0.5], [1.5, -0.5], [-0.5, 1.5], [1.5, 1.5],
        [0.5, -0.5], [0.5, 1.5], [-0.5, 0.5], [1.5, 0.5],
        [-0.25, -0.25], [1.25, -0.25], [-0.25, 1.25], [1.25, 1.25],
        [0.5, -0.25], [0.5, 1.25], [-0.25, 0.5], [1.25, 0.5],
    ])
    all_pts = np.vstack([pos, boundary_pts])

    try:
        vor = Voronoi(all_pts)
    except Exception:
        # Fallback: simple distance-based radii
        radii = np.zeros(n)
        for i in range(n):
            # Distance to boundary
            db = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            # Distance to nearest other circle
            dn = np.inf
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(pos[i] - pos[j])
                    dn = min(dn, d)
            radii[i] = 0.49 * min(db, dn / 2.0)
        return radii

    radii = np.zeros(n)
    for i in range(n):
        # Find the Voronoi region for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            # Unbounded region - use distance to boundary
            db = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            radii[i] = 0.49 * db
            continue

        # Get vertices of this Voronoi cell
        vertices = vor.vertices[region]

        # Maximum inscribed circle in the Voronoi cell
        # Approximate by finding the minimum distance from pos[i] to each edge
        max_r = np.inf

        # Distance to boundary of unit square
        db = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        max_r = min(max_r, db)

        # Distance to each Voronoi edge (line segment between vertices)
        for k in range(len(vertices)):
            v1 = vertices[k]
            v2 = vertices[(k + 1) % len(vertices)]
            # Distance from pos[i] to line segment v1-v2
            v = v2 - v1
            w = pos[i] - v1
            if np.dot(v, v) < 1e-12:
                d = np.linalg.norm(w)
            else:
                t = np.dot(w, v) / np.dot(v, v)
                t = np.clip(t, 0, 1)
                closest = v1 + t * v
                d = np.linalg.norm(pos[i] - closest)
            max_r = min(max_r, d)

        radii[i] = 0.49 * max_r

    return radii


def _make_hexagonal(n, seed=42):
    """Hexagonal close-packing initialization."""
    pos = np.zeros((n, 2))
    radii = np.zeros(n)

    # Hexagonal packing: rows offset by half spacing
    r = 0.11
    spacing = 2.0 * r
    row_spacing = spacing * np.sqrt(3) / 2.0

    # Start position to center the pattern
    x_start = 0.5 - 2.5 * spacing
    y_start = 0.5 - 2.0 * row_spacing

    idx = 0
    for row in range(6):
        y = y_start + row * row_spacing
        if row % 2 == 0:
            xs = np.arange(6) * spacing + x_start
        else:
            xs = np.arange(6) * spacing + x_start + spacing / 2.0

        for col, x in enumerate(xs):
            if idx >= n:
                break
            # Clamp to valid range
            pos[idx] = [np.clip(x, r, 1.0 - r), np.clip(y, r, 1.0 - r)]
            radii[idx] = r
            idx += 1
            if idx >= n:
                break

    return pos, radii


def _make_grid(n, seed=42):
    """Grid initialization."""
    side = 5
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def _make_center_ring(n, seed=42):
    """Center + ring initialization."""
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.14
    r_inner = 0.11
    dist_inner = 0.28
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos[i + 1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i + 1] = r_inner
    r_outer = 0.07
    dist_outer = 0.42
    for i in range(17):
        angle = 2 * np.pi * i / 17
        pos[i + 9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i + 9] = r_outer
    return pos, radii


def _make_random(n, seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(n, 2))
    radii = rng.uniform(0.04, 0.10, size=n)
    return pos, radii


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(1000):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if radii[i] <= radii[j]:
                        needed = (min_dist - dist) / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.5, 1.0 - needed * 0.999)
                    else:
                        needed = (min_dist - dist) / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.5, 1.0 - needed * 0.999)
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision L-BFGS-B polish."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            if time.time() < deadline - 5:
                try:
                    x2 = np.concatenate([pos_pp.flatten(), radii_pp])
                    bounds2 = []
                    for i in range(N):
                        bounds2.append((max(0.0, radii_pp[i] * 0.7), min(1.0, 1.0 - radii_pp[i] * 0.7)))
                        bounds2.append((max(0.0, radii_pp[i] * 0.7), min(1.0, 1.0 - radii_pp[i] * 0.7)))
                    for i in range(N):
                        bounds2.append((max(1e-4, radii_pp[i] * 0.8), min(0.35, radii_pp[i] * 1.2 + 0.02)))
                    res2 = minimize(lambda x: _objective(x, deadline), x2, method='SLSQP',
                                    bounds=bounds2,
                                    options={'maxiter': 500, 'ftol': 1e-15, 'disp': False})
                    pos_opt2 = res2.x[:2 * N].reshape(N, 2)
                    radii_opt2 = res2.x[2 * N:]
                    pos_pp2, radii_pp2 = _post_process(pos_opt2, radii_opt2, deadline)
                    if _is_valid(pos_pp2, radii_pp2) and np.sum(radii_pp2) > np.sum(radii_pp):
                        return pos_pp2, radii_pp2
                except Exception:
                    pass
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _force_directed_anneal(pos, radii, deadline, max_iter=3000, rng_seed=123):
    """Force-directed annealing to explore topological rearrangements."""
    pos = pos.copy()
    radii = radii.copy()
    rng = np.random.RandomState(rng_seed)

    T = 0.04
    cooling = 0.9995

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if T < 1e-6:
            break

        forces = np.zeros_like(pos)

        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 10.0
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i])) * 10.0
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 10.0
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i])) * 10.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    f = overlap * 50.0 / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
                elif dist > min_dist and dist < min_dist * 1.5:
                    f = (min_dist * 1.5 - dist) * 0.1 / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy

        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            local_step = 0.006 * T / (1.0 + radii[i] * 5.0)
            if f_norm > 1e-12:
                delta = local_step * (forces[i] / f_norm + rng.randn(2) * 0.4)
            else:
                delta = local_step * 0.3 * rng.randn(2)
            pos[i] += delta
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)

        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
            for i in range(N):
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                growth = 1.0012 if radii[i] < 0.08 else 1.0006
                radii[i] = min(radii[i] * growth, max_r * 0.999)

        T *= cooling

    return best_pos, best_radii


def _voronoi_refine(pos, radii, deadline, max_iter=200):
    """Iteratively recompute Voronoi radii and adjust positions."""
    pos = pos.copy()
    radii = radii.copy()
    rng = np.random.RandomState(789)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Recompute radii from Voronoi
        new_radii = _voronoi_radii(pos, deadline)

        # Use the minimum of current and Voronoi-computed radii
        # to avoid sudden jumps
        radii = np.minimum(radii * 1.005, new_radii * 1.0)

        # Check validity
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

        # Small random perturbation to escape local minima
        for i in range(N):
            pos[i] += rng.randn(2) * 0.005 * (1.0 + it / 100.0)
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)

    return best_pos, best_radii


def _homothetic_scale(pos, radii, deadline):
    """Try to find a global scale factor that improves the packing.
    This is a simplified version of the homothetic scaling approach."""
    pos = pos.copy()
    radii = radii.copy()

    # Try different scale factors around 1.0
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for scale in [1.0, 1.005, 1.01, 1.015, 1.02, 0.995, 0.99, 0.985, 0.98]:
        if time.time() > deadline:
            break
        scaled_pos = pos * scale
        scaled_radii = radii * scale

        # Clamp positions to valid range
        for i in range(N):
            max_r = min(scaled_pos[i, 0], 1.0 - scaled_pos[i, 0],
                       scaled_pos[i, 1], 1.0 - scaled_pos[i, 1])
            scaled_radii[i] = min(scaled_radii[i], max_r * 0.999)

        pos_pp, radii_pp = _post_process(scaled_pos, scaled_radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    # ---------- Stage 1: Multiple initializations with Voronoi radii ----------
    configs = [
        _make_hexagonal(N),
        _make_grid(N),
        _make_center_ring(N),
        _make_random(N, seed=1),
        _make_random(N, seed=2),
        _make_random(N, seed=3),
    ]

    for pos, radii in configs:
        if time.time() > deadline - 80:
            break
        # Use Voronoi to get better initial radii
        vor_radii = _voronoi_radii(pos, deadline)
        radii = np.maximum(radii, vor_radii * 0.8)  # Blend
        update_best(pos, radii)

    # ---------- Stage 2: Polish best configuration ----------
    if best_pos is not None and time.time() < deadline - 50:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

    # ---------- Stage 3: Voronoi refinement ----------
    if best_pos is not None and time.time() < deadline - 35:
        pos_v, radii_v = _voronoi_refine(best_pos, best_radii, deadline, max_iter=150)
        if _is_valid(pos_v, radii_v):
            s = np.sum(radii_v)
            if s > best_score:
                best_score = s
                best_pos = pos_v.copy()
                best_radii = radii_v.copy()

    # ---------- Stage 4: Force-directed annealing ----------
    if best_pos is not None and time.time() < deadline - 20:
        pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, deadline,
                                                   max_iter=2000, rng_seed=456)
        if _is_valid(pos_fd, radii_fd):
            s = np.sum(radii_fd)
            if s > best_score:
                best_score = s
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()

    # ---------- Stage 5: Homothetic scaling ----------
    if best_pos is not None and time.time() < deadline - 10:
        pos_h, radii_h = _homothetic_scale(best_pos, best_radii, deadline)
        if _is_valid(pos_h, radii_h):
            s = np.sum(radii_h)
            if s > best_score:
                best_score = s
                best_pos = pos_h.copy()
                best_radii = radii_h.copy()

    # ---------- Stage 6: Final polish ----------
    if best_pos is not None and time.time() < deadline - 5:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

        # Extra micro-polish: coordinate descent on radii
        if time.time() < deadline - 2:
            for _ in range(5):
                improved = False
                for i in range(N):
                    if time.time() > deadline - 1:
                        break
                    for growth in [1.003, 1.0015, 1.0008, 1.0004]:
                        new_r = best_radii[i] * growth
                        max_r = min(best_pos[i, 0], 1.0 - best_pos[i, 0],
                                    best_pos[i, 1], 1.0 - best_pos[i, 1])
                        if new_r > max_r:
                            new_r = max_r * 0.999
                        old_r = best_radii[i]
                        best_radii[i] = new_r
                        if _is_valid(best_pos, best_radii):
                            improved = True
                            break
                        else:
                            best_radii[i] = old_r
                if not improved:
                    break
            if _is_valid(best_pos, best_radii):
                s = np.sum(best_radii)
                if s > best_score:
                    best_score = s

    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
