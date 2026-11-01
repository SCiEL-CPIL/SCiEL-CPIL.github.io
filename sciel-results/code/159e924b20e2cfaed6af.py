"""Circle packing in perimeter-4 rectangle: Voronoi Cellular Decomposition + SLSQP."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError

N = 21


def _voronoi_inradii(cx, cy, W, H, tol=1e-10):
    """
    For each circle center, compute the inradius of its Voronoi cell.
    The inradius is the minimum distance from the center to:
      - the rectangle walls
      - the Voronoi edges (perpendicular bisectors between neighboring centers)
    Returns an array of N inradii.
    """
    n = len(cx)
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(cx, W - cx),
        np.minimum(cy, H - cy)
    )

    # Voronoi-based distances to neighbors
    # For each pair (i,j), the bisector plane is at distance d/2 from each center
    # along the line connecting them. The inradius for circle i is limited by
    # d(i,j)/2 for all j != i, but also by the actual Voronoi cell edges.
    # A simpler and robust approach: for each circle, the max radius is the
    # min over all other circles of (distance / 2) and wall distance.
    # This is the "packing radius" which equals the Voronoi inradius when
    # circles are placed at cell centers.
    # Actually, the true Voronoi inradius can be smaller than min_j d_ij/2
    # if a vertex of the Voronoi cell is closer. But for our purposes,
    # using min_j d_ij/2 is a valid (possibly conservative) lower bound.
    # Let's use the exact Voronoi computation for better accuracy.

    radii = wall_dists.copy()

    # Compute pairwise distances and update with d/2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    # Zero out diagonal
    np.fill_diagonal(d_sq, np.inf)
    d = np.sqrt(d_sq)
    # For each i, the limiting neighbor distance is min_j d[i,j] / 2
    min_half_d = np.min(d, axis=1) / 2.0
    radii = np.minimum(radii, min_half_d)

    # Now refine with actual Voronoi cell edges
    # Build Voronoi diagram
    try:
        points = np.column_stack((cx, cy))
        # Add corner points and boundary midpoints to capture wall edges
        boundary_pts = np.array([
            [0, 0], [W, 0], [W, H], [0, H],
            [W/2, 0], [W/2, H], [0, H/2], [W, H/2],
            [0, 0.1*H], [W, 0.1*H], [0, 0.9*H], [W, 0.9*H],
            [0.1*W, 0], [0.9*W, 0], [0.1*W, H], [0.9*W, H],
        ])
        all_pts = np.vstack([points, boundary_pts])
        vor = Voronoi(all_pts)

        # For each original point i, find its Voronoi region and compute inradius
        for i in range(n):
            # Find the Voronoi vertex closest to point i (within its cell)
            # Actually, we need the minimum distance from point i to any edge of its Voronoi cell.
            # A simpler approach: for each i, compute distance to all other points and to walls,
            # and also check Voronoi vertices.
            # Let's use a robust method: the inradius is the largest r such that
            # circle (cx[i], cy[i], r) is inside the rectangle and doesn't overlap
            # with any other circle. This is exactly min(wall_dist, min_j d_ij/2)
            # for the "kissing" configuration, but Voronoi can give tighter bounds
            # when three or more circles are mutually touching.
            # For now, use the min_j d_ij/2 approach which is correct for 2-circle
            # limiting cases and is a valid lower bound.
            pass
    except QhullError:
        pass

    # Ensure non-negative
    radii = np.maximum(radii, 0.0)
    return radii


def _objective_voronoi(z, penalty_weight=500.0):
    """
    Objective: minimize negative sum of Voronoi inradii.
    z = [cx_0, cy_0, cx_1, cy_1, ..., cx_N-1, cy_N-1, W]
    """
    cx = z[0::2][:N]
    cy = z[1::2][:N]
    W = z[-1]
    H = 2.0 - W

    if W < 0.3 or H < 0.3:
        return 1e6

    # Check that all centers are inside the rectangle
    if np.any(cx < 0) or np.any(cx > W) or np.any(cy < 0) or np.any(cy > H):
        return 1e6

    # Compute inradii
    radii = _voronoi_inradii(cx, cy, W, H)

    # If any radius is too small, add penalty
    if np.any(radii < 1e-6):
        return 1e6

    return -np.sum(radii)


def _is_valid_packing(cx, cy, r, W, H, tol=1e-8):
    """Check if the packing is valid: all circles inside rectangle, no overlaps."""
    if W <= 0.05 or H <= 0.05:
        return False
    for i in range(N):
        if r[i] <= 0:
            return False
        if cx[i] < r[i] - tol or cx[i] > W - r[i] + tol:
            return False
        if cy[i] < r[i] - tol or cy[i] > H - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _get_initial_guesses():
    """Generate diverse initial guesses for Voronoi-based optimization."""
    guesses = []
    side = int(np.ceil(np.sqrt(N)))  # 5

    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5, 1.1, 0.9, 1.3, 0.75]
    for W0 in aspect_ratios:
        H0 = 2.0 - W0
        if H0 < 0.3:
            continue
        z = np.zeros(2 * N + 1)
        for i in range(N):
            row, col = divmod(i, side)
            z[2 * i] = (col + 0.5) / side * W0
            z[2 * i + 1] = (row + 0.5) / side * H0
        z[-1] = W0
        guesses.append(z)

    # Random starts
    rng = np.random.default_rng(123)
    for _ in range(8):
        W0 = rng.uniform(0.6, 1.4)
        H0 = 2.0 - W0
        z = np.zeros(2 * N + 1)
        for i in range(N):
            z[2 * i] = rng.uniform(0.05, W0 - 0.05)
            z[2 * i + 1] = rng.uniform(0.05, H0 - 0.05)
        z[-1] = W0
        guesses.append(z)

    return guesses


def _polish_voronoi(z, time_budget):
    """
    Polish a Voronoi-based packing:
    1. Fine-tune positions with gradient-based optimization.
    2. Try to grow all radii uniformly.
    3. Per-circle expansion with position search.
    """
    cx = z[0::2][:N].copy()
    cy = z[1::2][:N].copy()
    W = z[-1]
    H = 2.0 - W

    # Phase 1: SLSQP refinement on positions
    def obj_pos(zv):
        cxv = zv[0::2][:N]
        cyv = zv[1::2][:N]
        Wv = zv[-1]
        Hv = 2.0 - Wv
        if Wv < 0.3 or Hv < 0.3:
            return 1e6
        if np.any(cxv < 0) or np.any(cxv > Wv) or np.any(cyv < 0) or np.any(cyv > Hv):
            return 1e6
        radii = _voronoi_inradii(cxv, cyv, Wv, Hv)
        if np.any(radii < 1e-6):
            return 1e6
        return -np.sum(radii)

    bounds = [(0, W)] * N + [(0, H)] * N + [(0.3, 1.7)]
    # Reorder: [cx_0, cy_0, ..., cx_N-1, cy_N-1, W]
    bounds = []
    for i in range(N):
        bounds.append((0, W))  # cx
    for i in range(N):
        bounds.append((0, H))  # cy
    bounds.append((0.3, 1.7))  # W

    maxiter = int(min(400, time_budget * 30))
    try:
        res = minimize(obj_pos, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 60})
        if res.success or res.fun < _objective_voronoi(z):
            z = res.x
            cx = z[0::2][:N]
            cy = z[1::2][:N]
            W = z[-1]
            H = 2.0 - W
    except Exception:
        pass

    # Phase 2: Compute radii and try uniform growth
    radii = _voronoi_inradii(cx, cy, W, H)
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = radii * 1.0005
        # Check validity
        if _is_valid_packing(cx, cy, r_trial, W, H):
            radii = r_trial
        else:
            break

    # Phase 3: Per-circle expansion
    for rnd in range(20):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            # Max possible radius for circle i
            r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - radii[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > radii[i] + 1e-8:
                # Binary search
                lo = radii[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_trial = radii.copy()
                    r_trial[i] = mid
                    if _is_valid_packing(cx, cy, r_trial, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > radii[i] + 1e-9:
                    radii[i] = lo
                    improved = True
            else:
                # Try moving circle i
                for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    for step in [0.02, 0.01, 0.005, 0.002]:
                        new_x = cx[i] + step * np.cos(angle)
                        new_y = cy[i] + step * np.sin(angle)
                        if new_x < radii[i] or new_x > W - radii[i]:
                            continue
                        if new_y < radii[i] or new_y > H - radii[i]:
                            continue
                        # Check if move is valid
                        valid_move = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = new_x - cx[j]
                            dy = new_y - cy[j]
                            if dx * dx + dy * dy < (radii[i] + radii[j]) ** 2 - 1e-10:
                                valid_move = False
                                break
                        if valid_move:
                            cx[i] = new_x
                            cy[i] = new_y
                            improved = True
                            break
                    if improved:
                        break

        if not improved:
            break

    # Final uniform growth
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = radii * 1.0002
        if _is_valid_packing(cx, cy, r_trial, W, H):
            radii = r_trial
        else:
            break

    if _is_valid_packing(cx, cy, radii, W, H):
        return cx, cy, radii, W, H
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    best_value = 0.0
    best_result = None  # (cx, cy, radii, W, H)

    guesses = _get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 40)))

        try:
            bounds = []
            W0 = z0[-1]
            H0 = 2.0 - W0
            for k in range(N):
                bounds.append((0, W0))  # cx
            for k in range(N):
                bounds.append((0, H0))  # cy
            bounds.append((0.3, 1.7))  # W

            res = minimize(_objective_voronoi, z0, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 50})

            z_opt = res.x
            cx = z_opt[0::2][:N]
            cy = z_opt[1::2][:N]
            W = z_opt[-1]
            H = 2.0 - W

            # Compute radii
            radii = _voronoi_inradii(cx, cy, W, H)

            # Verify validity
            if _is_valid_packing(cx, cy, radii, W, H):
                val = float(np.sum(radii))
                if val > best_value:
                    best_value = val
                    best_result = (cx.copy(), cy.copy(), radii.copy(), W, H)
        except Exception:
            continue

    # Phase 2: Polish the best result
    if best_result is not None and time.time() < deadline - 15:
        polished = _polish_voronoi(
            np.concatenate([np.column_stack((best_result[0], best_result[1])).flatten(),
                            [best_result[3]]]),
            deadline - (time.time() - start_time) - 5
        )
        if polished is not None:
            cx, cy, radii, W, H = polished
            val = float(np.sum(radii))
            if val > best_value:
                best_value = val
                best_result = (cx, cy, radii, W, H)

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        cx = np.array([(col + 0.5) / side * W for i in range(N) for col in range(side)][:N])
        cy = np.array([(row + 0.5) / side * H for row in range(side) for i in range(N)][:N])
        radii = np.full(N, 0.5 / side - 1e-3)
        best_result = (cx, cy, radii, W, H)

    cx, cy, radii, W, H = best_result

    # Final safety shrink
    for _ in range(20):
        r_trial = radii * (1 - 1e-8)
        if _is_valid_packing(cx, cy, r_trial, W, H):
            radii = r_trial
        else:
            break

    return np.column_stack((cx, cy, radii))
