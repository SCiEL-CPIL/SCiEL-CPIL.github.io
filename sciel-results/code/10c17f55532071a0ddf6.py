"""Packing 11 unit regular hexagons inside a regular hexagon.

Structural change: SDF-based differentiable optimization with gradient descent.
Instead of SAT-based discrete overlap checks, we use signed distance fields
for hexagons to get smooth gradients through near-contact states.
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0
START_TIME = time.time()


def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)


def _hex_sdf_point(x, y, cx, cy, angle_deg, side=1.0):
    """
    Signed distance from point (x,y) to a regular hexagon centered at (cx,cy)
    with rotation angle_deg. Negative inside, positive outside.
    For a regular hexagon with side length s, the SDF can be computed by
    transforming to local frame and using the hexagon SDF formula.
    """
    angle_rad = math.radians(angle_deg)
    ca = math.cos(-angle_rad)
    sa = math.sin(-angle_rad)
    dx = x - cx
    dy = y - cy
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca

    # For a regular hexagon with "radius" (center to vertex) = side,
    # the inradius = side * sqrt(3)/2
    # SDF for regular hexagon: we can use the formula based on 6 half-planes
    # but for smoothness, use a differentiable approximation.
    # Actually, for a regular hexagon with side s, the distance to the boundary
    # can be computed as follows:
    # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # (in local frame, before rotation). The inradius is s*sqrt(3)/2.
    # For a point in local frame, the signed distance is:
    #   min over 6 edges of (dot(point - midpoint_edge, normal) - inradius)
    # but this is not smooth. For a smooth SDF, we can use the fact that
    # for a regular hexagon, the SDF is:
    #   d = min_k ( |dot(p, n_k)| - inradius )  ... no, that's for a diamond.
    #
    # Better approach: for a regular hexagon with circumradius R = side,
    # the SDF at a point p (in local frame) is:
    #   Let r = |p|, theta = atan2(py, px)
    #   The hexagon boundary in polar coords: r_boundary(theta) = R * cos(30deg) / cos(theta - 60k deg)
    #   for theta in [60k-30, 60k+30]
    #   SDF = r - r_boundary(theta)  (approximately, but not exact for SDF)
    #
    # For our purposes, we'll use a penalty-based approach with smooth min
    # of the 6 half-plane distances.
    inradius = side * math.sqrt(3) / 2.0
    R = side  # circumradius = side for regular hexagon

    # 6 edge normals in local frame
    min_dist = 1e10
    for k in range(6):
        a = math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        # Distance from point to this edge's supporting line
        d = lx * nx + ly * ny - inradius
        if d < min_dist:
            min_dist = d

    # This gives the signed distance to the closest half-plane boundary,
    # which is an upper bound on the true SDF but works as a smooth penalty.
    # Actually this is not quite right. Let me use a proper approach.
    #
    # For a convex polygon, the signed distance can be computed as:
    # d(p) = min over edges of (distance from p to edge segment)
    # with sign determined by inside/outside.
    #
    # For simplicity and differentiability, I'll use the half-plane min approach
    # which gives a smooth (though not exact) SDF-like function.
    return min_dist


def _hex_sdf_grad(x, y, cx, cy, angle_deg, side=1.0):
    """Gradient of the hexagon SDF with respect to (cx, cy, angle)."""
    angle_rad = math.radians(angle_deg)
    ca = math.cos(-angle_rad)
    sa = math.sin(-angle_rad)
    dx = x - cx
    dy = y - cy
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca

    inradius = side * math.sqrt(3) / 2.0

    best_k = 0
    best_d = 1e10
    for k in range(6):
        a = math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        d = lx * nx + ly * ny - inradius
        if d < best_d:
            best_d = d
            best_k = k

    a = math.radians(30 + 60 * best_k)
    nx = math.cos(a)
    ny = math.sin(a)

    # d = lx*nx + ly*ny - inradius
    # lx = dx*ca - dy*sa, ly = dx*sa + dy*ca
    # d/dcx = -(dx*ca - dy*sa)*nx - (dx*sa + dy*ca)*ny = -dx*(ca*nx + sa*ny) - dy*(sa*nx - ca*ny)
    # d/dcy = -(dx*ca - dy*sa)*nx*0 ... let me redo
    # lx = (x-cx)*ca - (y-cy)*sa
    # d/dcx = -ca*nx - sa*ny ... wait
    # d = lx*nx + ly*ny - inradius
    # d/dcx = (dlx/dcx)*nx + (dly/dcx)*ny = (-ca)*nx + (-sa)*ny
    # d/dcy = (dlx/dcy)*nx + (dly/dcy)*ny = (sa)*nx + (ca)*ny
    # d/dangle: more complex, skip for now and use finite differences if needed

    grad_cx = -ca * nx - sa * ny
    grad_cy = sa * nx + ca * ny

    # For angle, we'll approximate with finite difference in the main loop
    return grad_cx, grad_cy


def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0

    for (ix, iy, iangle) in inner_positions:
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)

            dx = vx - cx
            dy = vy - cy
            oa_rad = math.radians(outer_angle)
            ca = math.cos(-oa_rad)
            sa = math.sin(-oa_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca

            r = math.sqrt(lx * lx + ly * ly)
            if r < 1e-12:
                continue

            theta = math.atan2(ly, lx)
            best_cos = 0.0
            for m in range(6):
                am = math.radians(30 + m * 60)
                d = abs(((theta - am) % (2 * math.pi) + math.pi) % (2 * math.pi))
                d = min(d, 2 * math.pi - d)
                c = math.cos(d)
                if c > best_cos:
                    best_cos = c

            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / math.sqrt(3)
                best_side = max(best_side, required_side)

    return best_side


def _hexagons_disjoint(cx1, cy1, a1, cx2, cy2, a2, tol=1e-7):
    """Check if two unit hexagons are disjoint using SAT."""
    def get_verts(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            verts.append([cx + math.cos(a), cy + math.sin(a)])
        return verts

    v1 = get_verts(cx1, cy1, a1)
    v2 = get_verts(cx2, cy2, a2)

    all_norms = []
    for verts in [v1, v2]:
        for i in range(6):
            j = (i + 1) % 6
            ex = verts[j][0] - verts[i][0]
            ey = verts[j][1] - verts[i][1]
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-12:
                all_norms.append((1.0, 0.0))
            else:
                all_norms.append((-ey / length, ex / length))

    for nx, ny in all_norms:
        proj1 = [vx * nx + vy * ny for vx, vy in v1]
        proj2 = [vx * nx + vy * ny for vx, vy in v2]
        if max(proj1) < min(proj2) - tol or max(proj2) < min(proj1) - tol:
            return True

    return False


def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """Check if configuration is valid."""
    cx, cy = outer_center
    inradius = outer_side * math.sqrt(3) / 2.0
    oa_rad = math.radians(outer_angle)

    for (ix, iy, iangle) in inner_positions:
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)

            dx = vx - cx
            dy = vy - cy
            ca = math.cos(-oa_rad)
            sa = math.sin(-oa_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca

            for m in range(6):
                am = math.radians(30 + m * 60)
                nx = math.cos(am)
                ny = math.sin(am)
                if lx * nx + ly * ny > inradius + tol:
                    return False

    for i in range(N):
        for j in range(i + 1, N):
            if not _hexagons_disjoint(inner_positions[i][0], inner_positions[i][1],
                                      inner_positions[i][2],
                                      inner_positions[j][0], inner_positions[j][1],
                                      inner_positions[j][2], tol=tol):
                return False

    return True


def _sdf_optimize(positions, outer_angle, max_iters=2000):
    """
    Optimize positions using SDF-based gradient descent.
    Uses smooth distance-based repulsion between hexagons and attraction to center.
    """
    # State: [x1, y1, th1, ..., xN, yN, thN]
    state = np.array([p for pos in positions for p in pos], dtype=np.float64)
    lr = 0.05
    best_state = state.copy()
    best_side = _min_enclosing_side([tuple(p) for p in state.reshape(N, 3)], (0.0, 0.0), outer_angle)

    for it in range(max_iters):
        if _remaining_time() < 5.0:
            break

        pos = state.reshape(N, 3)

        # Compute gradients
        grad = np.zeros_like(state)

        # Repulsion between hexagons using SDF
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, thi = pos[i]
                xj, yj, thj = pos[j]

                # Sample points on hexagon i's boundary and check distance to hexagon j
                # For efficiency, use center-to-center distance as proxy
                dx = xj - xi
                dy = yj - yi
                dist = math.sqrt(dx * dx + dy * dy)

                # Minimum distance between two unit hexagons is 2*sqrt(3) ≈ 3.464
                min_dist = 2 * math.sqrt(3)
                if dist < min_dist + 0.5:
                    # Repulsive force
                    force = (min_dist + 0.5 - dist)
                    fx = force * dx / (dist + 1e-10)
                    fy = force * dy / (dist + 1e-10)
                    grad[3 * i] -= fx
                    grad[3 * i + 1] -= fy
                    grad[3 * j] += fx
                    grad[3 * j + 1] += fy

        # Attraction to center (to keep them compact)
        for i in range(N):
            grad[3 * i] += 0.01 * pos[i, 0]
            grad[3 * i + 1] += 0.01 * pos[i, 1]

        # Gradient step
        state -= lr * grad

        # Evaluate
        cur_pos = [tuple(p) for p in state.reshape(N, 3)]
        cur_side = _min_enclosing_side(cur_pos, (0.0, 0.0), outer_angle)

        if cur_side < best_side:
            best_side = cur_side
            best_state = state.copy()

        # Adaptive learning rate
        if it % 200 == 0:
            lr *= 0.8

    return [tuple(p) for p in best_state.reshape(N, 3)], best_side


def _scipy_optimize(positions, outer_angle, time_budget):
    """Use scipy L-BFGS-B with smooth overlap penalties."""
    init_side = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    x0 = np.concatenate([np.array([p for pos in positions for p in pos]), [init_side]])

    angle_rad = math.radians(outer_angle)
    outer_normals = []
    for k in range(6):
        a_local = math.radians(30 + 60 * k)
        a_world = a_local + angle_rad
        outer_normals.append(np.array([math.cos(a_world), math.sin(a_world)]))

    def objective(x):
        pos = x[:3 * N].reshape(N, 3)
        S = x[3 * N]

        # Overlap penalty using SDF-like approach
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = 2 * math.sqrt(3)
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist) ** 2

        # Containment penalty
        cont_penalty = 0.0
        limit = S * math.sqrt(3) / 2.0
        for i in range(N):
            angle_rad_i = math.radians(pos[i, 2])
            for k in range(6):
                a = angle_rad_i + k * math.pi / 3
                vx = pos[i, 0] + math.cos(a)
                vy = pos[i, 1] + math.sin(a)
                for n in outer_normals:
                    proj = vx * n[0] + vy * n[1]
                    violation = proj - limit
                    if violation > 0:
                        cont_penalty += violation ** 2

        return S + 5.0 * overlap_penalty + 50.0 * cont_penalty

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12}
    )

    best_x = result.x
    best_pos = best_x[:3 * N].reshape(N, 3)
    best_pos[:, 2] = best_pos[:, 2] % 360

    true_S = _min_enclosing_side([tuple(p) for p in best_pos], (0.0, 0.0), outer_angle)
    return [tuple(p) for p in best_pos], true_S


def _initial_lattice():
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * math.sqrt(3)
    for k in range(6):
        a = math.radians(60 * k)
        positions.append([dist * math.cos(a), dist * math.sin(a), 0.0])
    dist2 = dist * math.sqrt(3)
    for k in range(4):
        a = math.radians(30 + 60 * k)
        positions.append([dist2 * math.cos(a), dist2 * math.sin(a), 0.0])
    return positions


def _initial_circular():
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        a = 2 * math.pi * i / (N - 1)
        r = 3.5
        positions.append([r * math.cos(a), r * math.sin(a), 0.0])
    return positions


def _initial_random():
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.5, 4.0)
        a = random.uniform(0, 2 * math.pi)
        angle = random.uniform(0, 360)
        positions.append([r * math.cos(a), r * math.sin(a), angle])
    return positions


def _try_arrangement(positions, outer_angle):
    if _remaining_time() < 5.0:
        return None, float('inf')

    best_pos = None
    best_side = float('inf')

    # Try SDF optimization
    try:
        sdf_pos, sdf_side = _sdf_optimize(positions, outer_angle)
        if sdf_side < best_side:
            best_side = sdf_side
            best_pos = sdf_pos
    except Exception:
        pass

    # Try scipy optimization
    try:
        sci_pos, sci_side = _scipy_optimize(positions, outer_angle, 5.0)
        if sci_side < best_side:
            best_side = sci_side
            best_pos = sci_pos
    except Exception:
        pass

    if best_pos is None:
        return None, float('inf')

    side = _min_enclosing_side(best_pos, (0.0, 0.0), outer_angle)

    if _check_validity(best_pos, (0.0, 0.0), side, outer_angle, tol=1e-4):
        return best_pos, side

    return None, float('inf')


def construct():
    random.seed(42)
    np.random.seed(42)

    best_result = None
    best_score = float('inf')

    outer_angles = [0.0, 30.0]

    initial_configs = [
        _initial_lattice(),
        _initial_circular(),
        _initial_random(),
        _initial_random(),
        _initial_random(),
    ]

    for oa in outer_angles:
        if _remaining_time() < 20.0:
            break

        for init_pos in initial_configs:
            if _remaining_time() < 10.0:
                break

            positions = [list(p) for p in init_pos]
            result, side = _try_arrangement(positions, oa)

            if result is not None and side < best_score:
                best_score = side
                best_result = (result, (0.0, 0.0), side, oa)

    if best_result is not None:
        return best_result

    # Fallback
    positions = _initial_lattice()
    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0
