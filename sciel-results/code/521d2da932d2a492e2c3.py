"""Circle packing in unit square – interior-point / active-set NLP with Delaunay contact graph."""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import cdist
import sys

N = 26

# EVOLVE-BLOCK-START

def _check_validity(x, y, r):
    """Return True if all circles are inside [0,1]^2 and disjoint."""
    if np.any(r <= 0):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # pairwise distances
    d = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
    for i in range(N):
        for j in range(i + 1, N):
            if d[i, j] < r[i] + r[j] - 1e-9:
                return False
    return True


def _voronoi_init(n, seed=0):
    """Generate initial positions via jittered grid, then estimate radii from Voronoi cells."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            pts.append([cx, cy])
    pts = np.array(pts[:n])
    # add jitter
    pts += rng.uniform(-0.02, 0.02, size=pts.shape)
    pts = np.clip(pts, 0.05, 0.95)

    # Estimate max radius from Voronoi cells
    try:
        vor = Voronoi(pts)
        radii = np.zeros(n)
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 in region:
                radii[i] = 0.02
                continue
            vertices = vor.vertices[region]
            # Find closest boundary point
            min_dist = 1.0
            for v in vertices:
                d = np.linalg.norm(pts[i] - v)
                min_dist = min(min_dist, d)
            # Also check against box boundaries
            min_dist = min(min_dist, pts[i][0], 1 - pts[i][0], pts[i][1], 1 - pts[i][1])
            radii[i] = max(0.01, min_dist * 0.95)
    except Exception:
        radii = np.full(n, 0.03)

    return pts, radii


def _solve_packing(pts0, radii0, max_time=30.0):
    """
    Solve the circle packing problem using SLSQP with dynamic contact graph.
    Variables: [x_1..x_n, y_1..y_n, r_1..r_n]
    Objective: maximize sum(r_i)
    Constraints:
      - r_i >= 0
      - x_i - r_i >= 0, x_i + r_i <= 1
      - y_i - r_i >= 0, y_i + r_i <= 1
      - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2  for all i < j
    """
    n = N
    x0 = np.concatenate([pts0[:, 0], pts0[:, 1], radii0])

    def objective(v):
        r = v[2*n:]
        return -np.sum(r)

    def objective_grad(v):
        g = np.zeros(3*n)
        g[2*n:] = -1.0
        return g

    # Inequality constraints: each returns >= 0
    constraints = []

    # Boundary constraints (4 per circle)
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[i] - v[2*n+i],
            'jac': lambda v, i=i: np.concatenate([
                np.eye(3*n)[i] - np.eye(3*n)[2*n+i]
            ])
        })
        # 1 - (x_i + r_i) >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[i] - v[2*n+i],
            'jac': lambda v, i=i: np.concatenate([
                -np.eye(3*n)[i] - np.eye(3*n)[2*n+i]
            ])
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[n+i] - v[2*n+i],
            'jac': lambda v, i=i: np.concatenate([
                np.eye(3*n)[n+i] - np.eye(3*n)[2*n+i]
            ])
        })
        # 1 - (y_i + r_i) >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[n+i] - v[2*n+i],
            'jac': lambda v, i=i: np.concatenate([
                -np.eye(3*n)[n+i] - np.eye(3*n)[2*n+i]
            ])
        })

    # Pairwise non-overlap: dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            def fun(v, i=i, j=j):
                dx = v[i] - v[j]
                dy = v[n+i] - v[n+j]
                ri = v[2*n+i]
                rj = v[2*n+j]
                return dx*dx + dy*dy - (ri + rj)**2

            def jac(v, i=i, j=j):
                dx = v[i] - v[j]
                dy = v[n+i] - v[n+j]
                ri = v[2*n+i]
                rj = v[2*n+j]
                g = np.zeros(3*n)
                g[i] = 2*dx
                g[j] = -2*dx
                g[n+i] = 2*dy
                g[n+j] = -2*dy
                g[2*n+i] = -2*(ri + rj)
                g[2*n+j] = -2*(ri + rj)
                return g

            constraints.append({'type': 'ineq', 'fun': fun, 'jac': jac})

    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0.001, 0.5)] * n

    t_start = time.time()
    res = minimize(
        objective, x0,
        method='SLSQP',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-12}
    )

    elapsed = time.time() - t_start
    return res, elapsed


def _refine_with_contact_graph(x, y, r, max_time=20.0):
    """
    Refine packing by identifying the contact graph via Delaunay and re-solving
    with only the active constraints (contact pairs + boundary).
    """
    n = N
    t_start = time.time()
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_val = np.sum(r)

    for iteration in range(5):
        if time.time() - t_start > max_time:
            break

        # Find contact pairs: circles whose distance is close to sum of radii
        pts = np.column_stack([x, y])
        d = cdist(pts, pts)
        contact_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < r[i] + r[j] + 1e-6:
                    contact_pairs.append((i, j))

        # Also check boundary contacts
        boundary_contacts = []
        for i in range(n):
            if x[i] - r[i] < 1e-6:
                boundary_contacts.append(('left', i))
            if x[i] + r[i] > 1 - 1e-6:
                boundary_contacts.append(('right', i))
            if y[i] - r[i] < 1e-6:
                boundary_contacts.append(('bottom', i))
            if y[i] + r[i] > 1 - 1e-6:
                boundary_contacts.append(('top', i))

        # Build reduced constraint set
        constraints = []
        for i in range(n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[i] - v[2*n+i],
                'jac': lambda v, i=i: np.concatenate([np.eye(3*n)[i] - np.eye(3*n)[2*n+i]])
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[i] - v[2*n+i],
                'jac': lambda v, i=i: np.concatenate([-np.eye(3*n)[i] - np.eye(3*n)[2*n+i]])
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[n+i] - v[2*n+i],
                'jac': lambda v, i=i: np.concatenate([np.eye(3*n)[n+i] - np.eye(3*n)[2*n+i]])
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[n+i] - v[2*n+i],
                'jac': lambda v, i=i: np.concatenate([-np.eye(3*n)[n+i] - np.eye(3*n)[2*n+i]])
            })

        for (i, j) in contact_pairs:
            def fun(v, i=i, j=j):
                dx = v[i] - v[j]
                dy = v[n+i] - v[n+j]
                ri = v[2*n+i]
                rj = v[2*n+j]
                return dx*dx + dy*dy - (ri + rj)**2
            def jac(v, i=i, j=j):
                dx = v[i] - v[j]
                dy = v[n+i] - v[n+j]
                ri = v[2*n+i]
                rj = v[2*n+j]
                g = np.zeros(3*n)
                g[i] = 2*dx
                g[j] = -2*dx
                g[n+i] = 2*dy
                g[n+j] = -2*dy
                g[2*n+i] = -2*(ri + rj)
                g[2*n+j] = -2*(ri + rj)
                return g
            constraints.append({'type': 'ineq', 'fun': fun, 'jac': jac})

        v0 = np.concatenate([x, y, r])
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(0.001, 0.5)] * n

        def obj(v):
            return -np.sum(v[2*n:])
        def obj_grad(v):
            g = np.zeros(3*n)
            g[2*n:] = -1.0
            return g

        res = minimize(obj, v0, method='SLSQP', jac=obj_grad,
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 100, 'ftol': 1e-12})

        if res.success or (hasattr(res, 'fun') and res.fun < -best_val + 1e-10):
            new_v = res.x
            new_x = new_v[:n]
            new_y = new_v[n:2*n]
            new_r = new_v[2*n:]
            if _check_validity(new_x, new_y, new_r):
                if np.sum(new_r) > best_val:
                    best_x, best_y, best_r = new_x, new_y, new_r
                    best_val = np.sum(new_r)
                x, y, r = new_x, new_y, new_r
            else:
                # Slightly shrink to make feasible
                new_r *= 0.999
                if _check_validity(new_x, new_y, new_r):
                    x, y, r = new_x, new_y, new_r
                    best_val = np.sum(r)
        else:
            break

    return best_x, best_y, best_r, best_val


def _multi_start(n_starts=8, total_time=110.0):
    """Run multiple starts with Voronoi-based initialization."""
    t_start = time.time()
    best_result = None
    best_value = -1.0

    for s in range(n_starts):
        if time.time() - t_start > total_time:
            break

        pts, radii = _voronoi_init(N, seed=s)
        # Scale radii slightly down for feasibility
        radii = np.clip(radii * 0.9, 0.01, 0.3)

        res, elapsed = _solve_packing(pts, radii, max_time=8.0)

        if res.success or res.fun is not None:
            v = res.x
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]

            # Check validity
            if _check_validity(x, y, r):
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_result = (x, y, r)

            # Refine with contact graph
            if time.time() - t_start < total_time - 15:
                rx, ry, rr, rval = _refine_with_contact_graph(x, y, r, max_time=10.0)
                if rval > best_value:
                    best_value = rval
                    best_result = (rx, ry, rr)

    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        arr = np.array(cells[:N])
        return arr

    x, y, r = best_result
    return np.column_stack([x, y, r])


def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s

    try:
        result = _multi_start(n_starts=10, total_time=100.0)
    except Exception:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        result = np.array(cells[:N])

    # Final validity check
    x = result[:, 0]
    y = result[:, 1]
    r = result[:, 2]
    if not _check_validity(x, y, r):
        # Shrink radii to make feasible
        r = r * 0.95
        result = np.column_stack([x, y, r])
        if not _check_validity(x, y, r):
            # Grid fallback
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            result = np.array(cells[:N])

    return result

# EVOLVE-BLOCK-END
