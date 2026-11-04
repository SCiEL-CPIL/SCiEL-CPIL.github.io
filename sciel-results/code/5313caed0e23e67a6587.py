"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (flat-top when angle=0)."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        ang = a + k * math.pi / 3.0
        pts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
    return np.array(pts)

def _support(pts, u):
    """Support function: max of dot(u, v) over vertices v."""
    return np.max(pts @ u)

def _signed_dist_point_seg(px, py, ax, ay, bx, by):
    """Signed distance from point P to segment AB, positive if left of A->B."""
    dx = bx - ax
    dy = by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-20:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / len2
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    dist = math.hypot(px - cx, py - cy)
    cross = dx * (py - ay) - dy * (px - ax)
    return dist if cross >= 0 else -dist

def _hex_min_distance(p, q, ang_p, ang_q, side=1.0):
    """Minimum distance between two unit hexagons (positive if separated)."""
    vp = hexagon_vertices(p[0], p[1], side, ang_p)
    vq = hexagon_vertices(q[0], q[1], side, ang_q)
    min_d = 1e18
    for i in range(6):
        ax, ay = vp[i]
        bx, by = vp[(i + 1) % 6]
        for k in range(6):
            d = _signed_dist_point_seg(vq[k, 0], vq[k, 1], ax, ay, bx, by)
            if d < min_d:
                min_d = d
    for i in range(6):
        ax, ay = vq[i]
        bx, by = vq[(i + 1) % 6]
        for k in range(6):
            d = _signed_dist_point_seg(vp[k, 0], vp[k, 1], ax, ay, bx, by)
            if d < min_d:
                min_d = d
    return min_d

def _outer_side_length(positions, angles):
    """Minimum side length of a regular hexagon (center at origin, rot=0)
    that contains all unit hexagons. Uses support function: side >= max over
    vertices v and directions u of dot(u, v)."""
    max_s = 0.0
    for i in range(N):
        x, y = positions[i]
        a = angles[i]
        verts = hexagon_vertices(x, y, 1.0, a)
        for vx, vy in verts:
            for k in range(6):
                ang = k * math.pi / 3.0
                u = (math.cos(ang), math.sin(ang))
                s = u[0] * vx + u[1] * vy
                if s > max_s:
                    max_s = s
    return max_s

def _objective(params, penalty=10.0):
    """Objective: outer side length + penalty for overlaps."""
    positions = params[:2 * N].reshape(N, 2)
    angles = params[2 * N:3 * N]
    s = _outer_side_length(positions, angles)
    # Pairwise separation penalty
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _hex_min_distance(positions[i], positions[j], angles[i], angles[j])
            if d < 0.0:
                pen += (-d) ** 2
    return s + penalty * pen

def _initial_positions():
    """Hexagonal cluster: 1 center + 6 ring + 5 outer."""
    pos = []
    # Center
    pos.append((0.0, 0.0))
    # Ring 1: 6 hexagons at distance ~2 (touching)
    for k in range(6):
        ang = k * math.pi / 3.0
        pos.append((2.0 * math.cos(ang), 2.0 * math.sin(ang)))
    # Ring 2: 5 hexagons (skip one to make 12 total)
    for k in range(6):
        if k == 5:  # skip one
            continue
        ang = k * math.pi / 3.0 + math.pi / 6.0
        r = 2.0 * math.sqrt(3)
        pos.append((r * math.cos(ang), r * math.sin(ang)))
    angles = [0.0] * N
    return np.array(pos), np.array(angles)

def _to_params(positions, angles):
    p = np.concatenate([positions.flatten(), angles])
    return p

def _from_params(p):
    positions = p[:2 * N].reshape(N, 2)
    angles = p[2 * N:3 * N]
    return positions, angles

def _polish(p0, max_iter=200):
    """Local polish with Nelder-Mead."""
    try:
        res = minimize(_objective, p0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        return res.x
    except Exception:
        return p0

def construct():
    t0 = time.time()
    positions, angles = _initial_positions()
    p0 = _to_params(positions, angles)
    best = p0
    best_val = _objective(best)

    # Try a few random restarts with small perturbations
    rng = np.random.default_rng(42)
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() - t0 > DEADLINE - 30:
            break
        # Perturb initial positions
        noise = rng.normal(0, 0.3, size=p0.shape)
        p_try = p0 + noise
        # Quick local optimization
        p_try = _polish(p_try, max_iter=100)
        val = _objective(p_try)
        if val < best_val:
            best = p_try
            best_val = val

    # Final high-precision polish
    if time.time() - t0 < DEADLINE - 10:
        best = _polish(best, max_iter=300)
        best_val = _objective(best)

    # Extract result
    positions, angles = _from_params(best)
    side = _outer_side_length(positions, angles)

    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])

    return inner, (0.0, 0.0), float(side), 0.0
