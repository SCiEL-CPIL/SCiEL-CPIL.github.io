"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_rad, side=1.0):
    a = angle_rad
    theta = a + np.arange(6) * (math.pi / 3.0)
    return np.stack([cx + side * np.cos(theta), cy + side * np.sin(theta)], axis=1)

def sat_overlap_depth(v1, v2, tol=1e-9):
    """Max penetration depth via SAT. Returns 0.0 if separated, >0 if overlapping."""
    depth = 0.0
    for poly in (v1, v2):
        for i in range(6):
            j = (i + 1) % 6
            edge = poly[j] - poly[i]
            length = np.hypot(edge[0], edge[1])
            if length < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / length
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < -tol:
                return 0.0
            if overlap > depth:
                depth = overlap
    return max(0.0, depth)

def compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """Exact minimum outer side length for given inner hexagons and outer orientation."""
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    # Outward normals for each edge
    out_normals = []
    for k in range(6):
        mid_angle = oa + k * math.pi / 3.0 + math.pi / 6.0
        out_normals.append(np.array([math.cos(mid_angle), math.sin(mid_angle)]))
    max_s = 0.0
    for hex_data in inner_hexes:
        cx, cy, ang = hex_data
        verts = hex_vertices(cx, cy, math.radians(ang))
        for on in out_normals:
            proj = (verts - np.array([ox, oy])) @ on
            max_proj = proj.max()
            if max_proj > 0:
                s_req = max_proj / (SQRT3 / 2.0)
                if s_req > max_s:
                    max_s = s_req
    return max_s

def unpack(x):
    """x: (N,3) array of [cx, cy, angle_rad]."""
    cx = x[:, 0]
    cy = x[:, 1]
    ang = x[:, 2]
    return cx, cy, ang

def objective(x, outer_center, outer_angle_rad, outer_side):
    """Total objective: side + penalties for overlap and container violation."""
    cx, cy, ang = unpack(x)
    verts_list = []
    for i in range(N):
        verts_list.append(hex_vertices(cx[i], cy[i], ang[i]))

    # Overlap penalty (smooth-ish: use depth directly)
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = sat_overlap_depth(verts_list[i], verts_list[j])
            if d > 0:
                overlap_penalty += d * d

    # Container violation
    ox, oy = outer_center
    out_normals = []
    for k in range(6):
        mid_angle = outer_angle_rad + k * math.pi / 3.0 + math.pi / 6.0
        out_normals.append(np.array([math.cos(mid_angle), math.sin(mid_angle)]))
    d_outer = outer_side * SQRT3 / 2.0
    container_penalty = 0.0
    for i in range(N):
        for on in out_normals:
            proj = (verts_list[i] - np.array([ox, oy])) @ on
            max_proj = proj.max()
            if max_proj > d_outer:
                violation = max_proj - d_outer
                container_penalty += violation * violation

    # The side length we want to minimize is the exact one; use current outer_side as proxy
    # but the real target is the exact min side. We return outer_side + penalties.
    return outer_side + 50.0 * overlap_penalty + 50.0 * container_penalty

def make_starts():
    """Generate diverse initial configurations as (N,3) arrays."""
    starts = []

    # 1. Honeycomb lattice (compact hexagonal arrangement)
    # 12 hexagons in a hexagonal cluster: center + ring of 6 + partial ring of 5
    def honeycomb():
        pts = []
        # center
        pts.append((0.0, 0.0, 0.0))
        # ring 1: 6 hexagons at distance 2*sqrt(3) (touching)
        for k in range(6):
            a = k * math.pi / 3.0
            r = 2.0 * SQRT3
            pts.append((r * math.cos(a), r * math.sin(a), 0.0))
        # ring 2: add 5 more around the gaps
        for k in range(5):
            a = (k + 0.5) * math.pi / 3.0
            r = 4.0 * SQRT3
            pts.append((r * math.cos(a), r * math.sin(a), 0.0))
        arr = np.array(pts[:N])
        return arr

    starts.append(honeycomb())

    # 2. Hexagonal ring + center (6+6 arrangement)
    def ring_center():
        pts = []
        pts.append((0.0, 0.0, 0.0))
        for k in range(6):
            a = k * math.pi / 3.0
            r = 2.0 * SQRT3
            pts.append((r * math.cos(a), r * math.sin(a), 0.0))
        # second ring: 5 more
        for k in range(5):
            a = (k + 0.5) * math.pi / 3.0
            r = 3.5 * SQRT3
            pts.append((r * math.cos(a), r * math.sin(a), 0.0))
        return np.array(pts[:N])

    starts.append(ring_center())

    # 3. 4x3 grid
    def grid():
        pts = []
        for row in range(4):
            for col in range(3):
                cx = col * 1.5 - 1.5
                cy = row * SQRT3 - 1.5 * SQRT3
                if row % 2 == 1:
                    cx += 0.75
                pts.append((cx, cy, 0.0))
        return np.array(pts[:N])

    starts.append(grid())

    # 4. 3x4 grid (transposed)
    def grid_t():
        pts = []
        for row in range(3):
            for col in range(4):
                cx = col * 1.5 - 2.25
                cy = row * SQRT3 - SQRT3
                if col % 2 == 1:
                    cy += SQRT3 / 2
                pts.append((cx, cy, 0.0))
        return np.array(pts[:N])

    starts.append(grid_t())

    # 5. Random starts
    np.random.seed(42)
    for trial in range(8):
        cx = np.random.uniform(-3, 3, N)
        cy = np.random.uniform(-3, 3, N)
        ang = np.random.uniform(0, math.pi / 3, N)
        starts.append(np.stack([cx, cy, ang], axis=1))

    # 6. Symmetry-broken honeycomb with small rotations
    base = honeycomb()
    for trial in range(4):
        pert = base.copy()
        pert[:, 2] += np.random.uniform(-0.1, 0.1, N)
        pert[:, 0] += np.random.uniform(-0.2, 0.2, N)
        pert[:, 1] += np.random.uniform(-0.2, 0.2, N)
        starts.append(pert)

    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    outer_center = np.array([0.0, 0.0])
    outer_angle_deg = 0.0
    outer_angle_rad = math.radians(outer_angle_deg)

    best_side = float('inf')
    best_x = None

    starts = make_starts()

    for si, x0 in enumerate(starts):
        if time.time() > deadline - 10.0:
            break

        # Start with a generous outer side, shrink over iterations
        current_side = 6.5

        # L-BFGS-B optimization
        x = x0.copy()
        for iteration in range(200):
            if time.time() > deadline - 10.0:
                break

            res = minimize(
                objective,
                x,
                args=(outer_center, outer_angle_rad, current_side),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
            )
            x = res.x

            # Check validity and compute exact side
            cx, cy, ang = unpack(x)
            inner_hexes = [[cx[i], cy[i], math.degrees(ang[i])] for i in range(N)]

            # Verify no overlaps and all inside a reasonable container
            valid = True
            verts_list = [hex_vertices(cx[i], cy[i], ang[i]) for i in range(N)]
            for i in range(N):
                for j in range(i + 1, N):
                    if sat_overlap_depth(verts_list[i], verts_list[j]) > 1e-7:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                exact_side = compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg)
                if exact_side < best_side:
                    best_side = exact_side
                    best_x = x.copy()

            # Shrink outer side to encourage tighter packing
            current_side = max(3.5, current_side - 0.03)

        # After main optimization, do a local polish with the best known side
        if best_x is not None:
            # Polish around best_x with tighter tolerance
            x = best_x + np.random.normal(0, 0.05, best_x.shape)
            for iteration in range(50):
                if time.time() > deadline - 5.0:
                    break
                res = minimize(
                    objective,
                    x,
                    args=(outer_center, outer_angle_rad, best_side),
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False},
                )
                x = res.x
                cx, cy, ang = unpack(x)
                inner_hexes = [[cx[i], cy[i], math.degrees(ang[i])] for i in range(N)]
                verts_list = [hex_vertices(cx[i], cy[i], ang[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if sat_overlap_depth(verts_list[i], verts_list[j]) > 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    exact_side = compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg)
                    if exact_side < best_side:
                        best_side = exact_side
                        best_x = x.copy()

    # Fallback: if no valid configuration found, use the best_x or a grid
    if best_x is None:
        best_x = starts[2]  # grid
        cx, cy, ang = unpack(best_x)
        inner_hexes = [[cx[i], cy[i], math.degrees(ang[i])] for i in range(N)]
        best_side = compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg)

    cx, cy, ang = unpack(best_x)
    inner = [[cx[i], cy[i], math.degrees(ang[i])] for i in range(N)]

    return inner, (0.0, 0.0), best_side, outer_angle_deg

# EVOLVE-BLOCK-END
