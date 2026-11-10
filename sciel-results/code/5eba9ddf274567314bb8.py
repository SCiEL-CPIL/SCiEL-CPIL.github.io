# EVOLVE-BLOCK-START
import time
import numpy as np

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    # 6 edge normals (unit, outward)
    n = []
    for i in range(6):
        p1 = V[i]
        p2 = V[(i + 1) % 6]
        e = p2 - p1
        L = np.hypot(e[0], e[1])
        # outward normal = (e_y, -e_x)/L for CCW polygon
        n.append(np.array([e[1], -e[0]]) / L)
    return np.array(n)

def _gap(A, B):
    # A, B: vertices (6,2) of two unit hexagons.
    # Return min over 12 edge-normal directions of (gap), where gap>0 means separated.
    # For edge of A with normal n: project all B vertices and A edge endpoints onto n.
    # The two intervals: [min_A, max_A] and [min_B, max_B] on axis n.
    # Separation on this axis = max(0, min_B - max_A, min_A - max_B) ... but we need
    # "on at least one of the 12 edge normals the projected intervals are apart by more than 1e-6".
    # For penalty, we want a smooth measure of violation: for each of 12 normals,
    # compute how much the intervals overlap (negative gap). Take max overlap (closest to 0)
    # as the separation margin; if margin > 1e-6 it's valid. Penalty on max(0, -margin).
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best_margin = -np.inf
    for n in nA:
        projA = A @ n
        projB = B @ n
        loA, hiA = projA.min(), projA.max()
        loB, hiB = projB.min(), projB.max()
        # separation: distance between intervals
        sep = max(loB - hiA, loA - hiB)
        if sep > best_margin:
            best_margin = sep
    for n in nB:
        projA = A @ n
        projB = B @ n
        loA, hiA = projA.min(), projA.max()
        loB, hiB = projB.min(), projB.max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best_margin:
            best_margin = sep
    return best_margin  # larger = more separated; need > 1e-6

def _outer_side_for_points(pts):
    # pts: (N,2). Find minimum side length s of a regular hexagon centered at origin
    # (we will center later) with some orientation such that all pts are inside.
    # A regular hexagon with side s, center (0,0), angle a has 6 edges.
    # Point p is inside iff for each edge normal n_k (outward), p . n_k <= apothem = s * sqrt(3)/2.
    # So s >= max_k (p . n_k) / (sqrt(3)/2) for all p, and we can choose a to minimize max.
    # We'll search over a in [0, 60) degrees (hexagon has 60-deg symmetry).
    # apothem = s * sqrt(3)/2 => s = 2*apothem/sqrt(3)
    best_s = np.inf
    # Use a fine grid then refine
    for a_deg in np.linspace(0, 60, 361):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(th), np.sin(th)))
        # For each normal, max projection
        proj = pts @ normals.T  # (N, 6)
        maxproj = proj.max(axis=0)  # (6,)
        # Need apothem >= max(maxproj), s = 2*apothem/sqrt(3)
        apothem = maxproj.max()
        s = 2.0 * apothem / np.sqrt(3.0)
        if s < best_s:
            best_s = s
    return best_s

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 600.0  # safety margin: return before 630s kill

    # Seed: hexagonal lattice of 11 hexagons
    # Centers in hex lattice: spacing 2.5 in x, 2.165 in y for adjacent
    # Actually for unit hexagons (flat-top, angle 0), the centers of touching hexagons:
    # For pointy-top (angle 90) or flat-top (angle 0)?
    # The task says vertices at (cx + s*cos(a + 60i), cy + s*sin(a + 60i)), i=0..5
    # With a=0: vertices at angles 0,60,120,180,240,300 -> pointy-top (vertex at right)
    # For pointy-top unit hexagons, adjacent centers:
    # Horizontal neighbor: distance = 2 * apothem = sqrt(3) ~ 1.732? No.
    # Actually for regular hexagon with side 1, the distance between centers of two
    # touching hexagons (sharing an edge) is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # But the direction depends on which edge.
    # Let's just use a reasonable seed and let the optimizer figure it out.
    # Seed positions (rough hex arrangement):
    seed = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [-2.0, 0.0],
        [1.0, 1.732],
        [-1.0, 1.732],
        [1.0, -1.732],
        [-1.0, -1.732],
        [3.0, 1.732],
        [-3.0, 1.732],
        [3.0, -1.732],
        [-3.0, -1.732],
    ])
    angles = np.zeros(11)

    # State: (x, y, angle) for each hexagon
    # Flatten: 33 variables
    def unpack(state):
        x = state[0::3]
        y = state[1::3]
        a = state[2::3]
        return x, y, a

    def get_vertices(state):
        x, y, a = unpack(state)
        Vs = []
        for i in range(11):
            Vs.append(_hex_vertices(x[i], y[i], 1.0, a[i]))
        return Vs

    def objective(state):
        """Return (outer_side, penalty)."""
        x, y, a = unpack(state)
        # Center of mass -> shift so outer hex is centered at origin
        cx = x.mean()
        cy = y.mean()
        pts = np.column_stack((x - cx, y - cy))
        s = _outer_side_for_points(pts)
        # Separation penalty
        Vs = get_vertices(state)
        pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                m = _gap(Vs[i], Vs[j])
                if m <= 1e-6:
                    # Violation: penalty proportional to how much it's not separated
                    pen += max(0.0, 1e-6 - m) ** 2
        return s, pen

    # Initial state
    state = np.concatenate([seed[:, 0], seed[:, 1], angles])
    best_state = state.copy()
    best_obj = objective(state)
    best_val = 1.0 / best_obj[0] if best_obj[0] > 0 and best_obj[1] < 1e-9 else -np.inf

    # Gradient descent with numerical gradients
    lr = 0.01
    step = 0
    while time.time() < deadline - 30.0:  # safety margin
        if time.time() - t0 > deadline - 60.0:
            break
        # Compute objective
        s, pen = objective(state)
        total = s + 1000.0 * pen
        # Check if valid and better
        if pen < 1e-9 and s < best_obj[0]:
            best_state = state.copy()
            best_obj = (s, pen)
            best_val = 1.0 / s

        # Numerical gradient
        grad = np.zeros_like(state)
        eps = 1e-5
        for k in range(len(state)):
            sp = state.copy()
            sp[k] += eps
            _, pen_p = objective(sp)
            s_p = _outer_side_for_points(np.column_stack((sp[0::3] - sp[0::3].mean(), sp[1::3] - sp[1::3].mean())))
            fp = s_p + 1000.0 * pen_p

            sm = state.copy()
            sm[k] -= eps
            _, pen_m = objective(sm)
            s_m = _outer_side_for_points(np.column_stack((sm[0::3] - sm[0::3].mean(), sm[1::3] - sm[1::3].mean())))
            fm = s_m + 1000.0 * pen_m

            grad[k] = (fp - fm) / (2.0 * eps)

        # Update
        state = state - lr * grad
        # Clamp angles to reasonable range
        state[2::3] = np.clip(state[2::3], -180, 180)

        # Adaptive learning rate
        step += 1
        if step % 50 == 0:
            lr *= 0.95

        if time.time() > deadline - 30.0:
            break

    # Final evaluation
    s, pen = objective(best_state)
    if pen >= 1e-9 or s <= 0:
        # Fallback to a known valid (loose) packing
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        outer = np.array([0, 0, 0])
        return inner, outer, 8.0

    # Build output
    x, y, a = unpack(best_state)
    cx = x.mean()
    cy = y.mean()
    # Find best outer hex angle
    pts = np.column_stack((x - cx, y - cy))
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, 361):
        a_rad = np.deg2rad(a_deg)
        th = a_rad + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(th), np.sin(th)))
        proj = pts @ normals.T
        maxproj = proj.max(axis=0)
        apothem = maxproj.max()
        s_candidate = 2.0 * apothem / np.sqrt(3.0)
        if s_candidate < best_s:
            best_s = s_candidate
            best_a = a_deg

    inner = np.column_stack((x, y, a))
    outer = np.array([cx, cy, best_a])
    return inner, outer, float(best_s)

# EVOLVE-BLOCK-END
