"""Circle packing in unit square: Voronoi-tessellation dual representation.

The unit square is partitioned into N cells by the centers; each circle is the
inscribed circle (inradius) of its cell, so disjointness is automatic.  We
optimize the 2N center coordinates by gradient ascent on the sum of inradii,
using Voronoi vertices for radii and subgradients.
"""
import numpy as np
import time
from scipy.spatial import Voronoi

N = 26
EPS = 1e-9

# EVOLVE-BLOCK-START


def _inradii_and_grads(centers, n=N):
    """Compute inradius of each Voronoi cell and a subgradient of sum of inradii.

    Returns (radii, grad) where radii has shape (n,) and grad has shape (n, 2).
    """
    # Voronoi needs points strictly in the interior of the hull; add boundary
    # points so the diagram tiles the whole unit square and cells are bounded.
    pts = np.vstack([centers,
                     np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])])
    vor = Voronoi(pts)
    radii = np.zeros(n)
    grad = np.zeros((n, 2))

    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if len(region) == 0:
            radii[i] = 0.0
            continue
        verts = vor.vertices[region]
        if len(verts) < 3:
            radii[i] = 0.0
            continue
        # Inradius = min distance from center to each edge (edge between two
        # consecutive vertices, which lie on a single Voronoi edge line).
        m = len(verts)
        best = np.inf
        best_grad = np.zeros(2)
        for k in range(m):
            a = verts[k]
            b = verts[(k + 1) % m]
            d = b - a
            dlen = np.hypot(d[0], d[1])
            if dlen < 1e-14:
                continue
            # distance from center c to line through a,b
            # signed: (c - a) x d / |d|
            num = (centers[i, 0] - a[0]) * d[1] - (centers[i, 1] - a[1]) * d[0]
            dist = num / dlen
            if dist < best:
                best = dist
                # unit normal pointing from the line toward the center:
                # n = (-dy, dx)/|d| ; if num<0 the center is on the -n side,
                # so the normal toward center is sign(num)*n.
                nx = -d[1] / dlen
                ny = d[0] / dlen
                if num < 0:
                    nx, ny = -nx, -ny
                best_grad = np.array([nx, ny])
        if not np.isfinite(best):
            radii[i] = 0.0
        else:
            radii[i] = best
            grad[i] = best_grad
    return radii, grad


def _project_centers(centers):
    """Clip centers into the unit square (with a tiny margin)."""
    m = 1e-6
    centers[:, 0] = np.clip(centers[:, 0], m, 1.0 - m)
    centers[:, 1] = np.clip(centers[:, 1], m, 1.0 - m)
    return centers


def _voronoi_gradient_ascent(centers, deadline, lr0=0.01, max_iter=4000,
                             min_time_per_iter=0.004):
    """Gradient ascent on sum of inradii with adaptive step size."""
    centers = _project_centers(centers.copy())
    radii, _ = _inradii_and_grads(centers)
    best_val = float(np.sum(radii))
    best_centers = centers.copy()
    lr = lr0
    stall = 0
    n = len(centers)

    for it in range(max_iter):
        if time.time() > deadline - 0.5:
            break
        radii, grad = _inradii_and_grads(centers)
        cur_val = float(np.sum(radii))
        if cur_val > best_val + 1e-12:
            best_val = cur_val
            best_centers = centers.copy()
            stall = 0
        else:
            stall += 1

        # Gradient ascent: move each center along its inradius subgradient.
        # The sum of inradii is concave in the sense that moving a center toward
        # the incenter increases its radius; subgradient ascent is appropriate.
        step = lr * grad
        # Limit step size to avoid jumping over cells (which changes the diagram
        # non-smoothly). Cap per-coordinate step.
        max_step = 0.02
        step = np.clip(step, -max_step, max_step)
        new_centers = _project_centers(centers + step)

        new_radii, _ = _inradii_and_grads(new_centers)
        new_val = float(np.sum(new_radii))
        if new_val > cur_val + 1e-12:
            centers = new_centers
            lr = min(lr * 1.05, 0.05)
        else:
            # Reduce step
            lr = max(lr * 0.5, 1e-4)
            if stall > 60:
                break

    return best_centers, best_val


def _voronoi_lbfgs(centers, deadline, max_iter=2000):
    """Use L-BFGS-B to maximize sum of inradii (smooth-ish surrogate)."""
    n = len(centers)
    flat = centers.ravel().copy()

    def neg_sum(v):
        c = v.reshape(n, 2)
        c = _project_centers(c)
        radii, _ = _inradii_and_grads(c)
        return -float(np.sum(radii))

    def neg_grad(v):
        c = v.reshape(n, 2)
        c = _project_centers(c)
        radii, grad = _inradii_and_grads(c)
        return -grad.ravel()

    from scipy.optimize import minimize
    bounds = [(0.0, 1.0)] * (2 * n)
    try:
        res = minimize(neg_sum, flat, jac=neg_grad, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12,
                                'gtol': 1e-8, 'maxfun': max_iter * 5})
        new_centers = res.x.reshape(n, 2)
        new_centers = _project_centers(new_centers)
    except Exception:
        new_centers = centers.copy()
    return new_centers


def _final_radii(centers):
    """Compute the final (clipped) radii from the Voronoi inradii, ensuring
    feasibility against walls and neighbors."""
    radii, _ = _inradii_and_grads(centers)
    # Clip to wall clearance
    wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.minimum(radii, wall)
    radii = np.maximum(radii, 0.0)
    # Enforce pairwise disjointness conservatively: shrink any overlapping pair.
    for _ in range(3):
        changed = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.hypot(centers[i, 0] - centers[j, 0],
                             centers[i, 1] - centers[j, 1])
                s = radii[i] + radii[j]
                if s > d + EPS:
                    # split the deficit equally
                    deficit = s - d
                    half = deficit / 2.0
                    if radii[i] > half:
                        radii[i] -= half
                        radii[j] -= half
                    else:
                        radii[j] -= max(deficit, 0.0)
                    radii[i] = max(radii[i], 0.0)
                    radii[j] = max(radii[j], 0.0)
                    changed = True
        if not changed:
            break
    # Final wall clip
    radii = np.minimum(radii, wall)
    radii = np.maximum(radii, 0.0)
    return radii


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
                positions.append([x, y])
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(0.05, 0.95, size=(n, 2))


def _spiral_init(n, seed=0):
    """Archimedean spiral of centers for diverse initial geometry."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 6 * np.pi, n)
    r = 0.03 + 0.42 * t / (6 * np.pi)
    x = 0.5 + r * np.cos(t)
    y = 0.5 + r * np.sin(t)
    x = np.clip(x + rng.normal(0, 0.01, n), 0.02, 0.98)
    y = np.clip(y + rng.normal(0, 0.01, n), 0.02, 0.98)
    return np.column_stack([x, y])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    best_centers = None
    best_total = 0.0

    # Phase 1: diverse initializations + Voronoi gradient ascent
    inits = []
    for side in [5, 6, 7]:
        inits.append(_grid_init(side))
    inits.append(_spiral_init(N, seed=1))
    for s in range(6):
        inits.append(_random_init(N, seed=s))

    for idx, c0 in enumerate(inits):
        if time.time() > deadline - 5:
            break
        centers, val = _voronoi_gradient_ascent(c0, deadline)
        radii = _final_radii(centers)
        total, valid = _eval_packing(centers, radii)
        if valid and total > best_total:
            best_total = total
            best_centers = centers.copy()

    # Phase 2: L-BFGS polish on the best centers (and a couple of others)
    candidates = [best_centers]
    if time.time() < deadline - 30:
        # polish the top-2 initializations too
        for c0 in inits[:3]:
            if time.time() > deadline - 20:
                break
            centers, val = _voronoi_gradient_ascent(c0, deadline)
            pc = _voronoi_lbfgs(centers, deadline, max_iter=500)
            radii = _final_radii(pc)
            total, valid = _eval_packing(pc, radii)
            if valid and total > best_total:
                best_total = total
                best_centers = pc.copy()
                candidates = [best_centers]

    # L-BFGS on the overall best
    if best_centers is not None and time.time() < deadline - 10:
        polished = _voronoi_lbfgs(best_centers, deadline, max_iter=1500)
        radii = _final_radii(polished)
        total, valid = _eval_packing(polished, radii)
        if valid and total > best_total:
            best_total = total
            best_centers = polished.copy()

    # Phase 3: final Voronoi gradient ascent from polished result
    if best_centers is not None and time.time() < deadline - 5:
        centers, val = _voronoi_gradient_ascent(best_centers, deadline)
        radii = _final_radii(centers)
        total, valid = _eval_packing(centers, radii)
        if valid and total > best_total:
            best_total = total
            best_centers = centers.copy()

    # Fallback
    if best_centers is None:
        best_centers = _grid_init(5)

    radii = _final_radii(best_centers)
    result = np.column_stack([best_centers, radii])

    # Final validation; if invalid, fall back to a safe grid.
    total, valid = _eval_packing(result[:, :2], result[:, 2])
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
