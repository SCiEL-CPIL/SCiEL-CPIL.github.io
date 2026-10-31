import numpy as np
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg=0):
    """Vertices of a regular hexagon (pointy-top when angle=0)."""
    angle_rad = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def hexagon_in_box(v, cx, cy, side, angle_deg):
    """Return True if all vertices of hexagon v are inside the outer hexagon."""
    # outer hexagon half-planes: for each of 6 edge normals n_i, (p - c) . n_i <= side*sqrt(3)/2
    a = np.radians(angle_deg)
    normals = np.stack([np.cos(a + i * np.pi / 3), np.sin(a + i * np.pi / 3)] for i in range(6))
    limit = side * np.sqrt(3) / 2.0
    rel = v - np.array([cx, cy])
    d = rel @ normals.T   # shape (6,6)
    return bool(np.all(d <= limit + 1e-9))

def hexagon_overlap(a, b):
    """Return minimal penetration distance (>0) between two convex hexagons, 0 if disjoint."""
    # Separating axis theorem on the 12 edge normals
    axes = []
    for v in (a, b):
        for i in range(6):
            e = v[(i + 1) % 6] - v[i]
            n = np.array([-e[1], e[0]])
            nrm = np.linalg.norm(n)
            if nrm > 1e-12:
                axes.append(n / nrm)
    min_gap = np.inf
    for n in axes:
        pa = a @ n
        pb = b @ n
        gap = max(pb.min() - pa.max(), pa.min() - pb.max())
        if gap > min_gap:
            min_gap = gap
        if gap <= 1e-9:
            return 0.0
    return max(0.0, min_gap)

def overlap_penalty(verts_list):
    total = 0.0
    m = len(verts_list)
    for i in range(m):
        for j in range(i + 1, m):
            pen = hexagon_overlap(verts_list[i], verts_list[j])
            if pen > 0:
                total += pen
    return total

def build_positions(center, spacing, ring_angles, ring_radii, all_rot):
    """Build a honeycomb cluster: center + ring1 (6) + ring2 (up to 12) = 19, take N."""
    pos = []
    rot = []
    # center
    pos.append((center[0], center[1]))
    rot.append(all_rot)
    # ring 1
    for ang in ring_angles:
        pos.append((center[0] + spacing * ring_radii[0] * np.cos(ang),
                    center[1] + spacing * ring_radii[0] * np.sin(ang)))
        rot.append(all_rot)
    # ring 2
    for ang in ring_angles:
        for ang2 in (ang + np.pi / 6, ang - np.pi / 6):
            pos.append((center[0] + spacing * ring_radii[1] * np.cos(ang2),
                        center[1] + spacing * ring_radii[1] * np.sin(ang2)))
            rot.append(all_rot)
    pos = pos[:N]
    rot = rot[:N]
    return np.array(pos), np.array(rot)

def minimal_side(pos, rot, outer_center, outer_angle):
    """Min outer side s so all inner hexagons fit in outer hexagon centered at outer_center."""
    a = np.radians(outer_angle)
    normals = np.stack([np.cos(a + i * np.pi / 3), np.sin(a + i * np.pi / 3)] for i in range(6))
    max_proj = 0.0
    for k in range(N):
        v = hexagon_vertices(pos[k, 0], pos[k, 1], 1.0, rot[k])
        d = (v - np.array(outer_center)) @ normals.T
        max_proj = max(max_proj, float(d.max()))
    return max_proj / (np.sqrt(3) / 2.0)

def evaluate(pos, rot, outer_center, outer_angle):
    """Return (side, overlap_penalty)."""
    verts_list = [hexagon_vertices(pos[k, 0], pos[k, 1], 1.0, rot[k]) for k in range(N)]
    side = minimal_side(pos, rot, outer_center, outer_angle)
    # check containment
    for k in range(N):
        if not hexagon_in_box(verts_list[k], outer_center[0], outer_center[1], side, outer_angle):
            return None, None
    pen = overlap_penalty(verts_list)
    return side, pen

def local_search(pos, rot, outer_center, outer_angle, deadline, rng):
    """Perturbation + accept if improves (side, penalty) lexicographically."""
    def score(side, pen):
        return (pen, side)
    verts_list = [hexagon_vertices(pos[k, 0], pos[k, 1], 1.0, rot[k]) for k in range(N)]
    side, pen = evaluate(pos, rot, outer_center, outer_angle)
    if side is None:
        return pos, rot, outer_center, outer_angle, np.inf
    best_side, best_pen = side, pen
    best_pos, best_rot = pos.copy(), rot.copy()
    best_oc = outer_center.copy()
    best_ang = outer_angle
    step = 0.05
    while time.time() < deadline:
        # random perturbation of a few positions and center
        pos2 = pos.copy()
        rot2 = rot.copy()
        oc2 = outer_center.copy()
        ang2 = outer_angle
        for _ in range(3):
            idx = rng.integers(0, N)
            pos2[idx] += step * rng.standard_normal(2)
            rot2[idx] = (rot2[idx] + step * rng.standard_normal()) % 60
        oc2 += step * 0.3 * rng.standard_normal(2)
        ang2 = (ang2 + step * 0.2 * rng.standard_normal()) % 60
        s2, p2 = evaluate(pos2, rot2, oc2, ang2)
        if s2 is not None:
            if (p2, s2) < (best_pen, best_side):
                best_side, best_pen = s2, p2
                best_pos, best_rot = pos2.copy(), rot2.copy()
                best_oc, best_ang = oc2.copy(), ang2
                pos, rot, outer_center, outer_angle = best_pos, best_rot, best_oc, best_ang
                step = min(step, 0.05)
            else:
                step = max(step * 0.9, 0.005)
    return best_pos, best_rot, best_oc, best_ang, best_side

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin under 150s
    rng = np.random.default_rng(12345)

    best_side = np.inf
    best_result = None

    # base honeycomb geometry
    ring_angles = np.array([i * np.pi / 3 for i in range(6)])
    ring_radii = np.array([1.0, 2.0])  # in units of spacing (center-to-ring1 = spacing, ring2 = 2*spacing)

    # candidate spacings (scale) and global rotations
    spacings = [1.0, 1.001, 1.002, 0.999, 1.005, 1.01]
    global_rots = [0.0, 5.0, 10.0, 15.0, 30.0]
    outer_angles = [0.0, 15.0, 30.0, 45.0]

    # Precompute candidate centers near origin
    centers = [(0.0, 0.0)]

    candidates = []
    for sp in spacings:
        for gr in global_rots:
            pos, rot = build_positions((0.0, 0.0), sp, ring_angles, ring_radii, gr)
            for oa in outer_angles:
                candidates.append((pos, rot, (0.0, 0.0), oa))

    # Evaluate all candidates, keep best, then refine top few with local search
    scored = []
    for pos, rot, oc, oa in candidates:
        if time.time() > deadline - 30:
            break
        s, p = evaluate(pos, rot, oc, oa)
        if s is not None:
            scored.append((p, s, pos, rot, oc, oa))

    scored.sort(key=lambda t: (t[0], t[1]))

    # Refine the best few with local search
    refine_count = 0
    for p, s, pos, rot, oc, oa in scored:
        if time.time() > deadline - 5:
            break
        if refine_count >= 6:
            break
        np2, nr2, noc2, noa2, side2 = local_search(pos, rot, oc, oa, deadline - 5, rng)
        refine_count += 1
        if side2 < best_side:
            best_side = side2
            best_result = (np2, nr2, noc2, noa2)

    if best_result is not None:
        pos, rot, oc, oa = best_result
        inner = [[float(pos[k, 0]), float(pos[k, 1]), float(rot[k])] for k in range(N)]
        return inner, (float(oc[0]), float(oc[1])), float(best_side), float(oa)

    # Fallback: return a valid linear arrangement
    inner = [[float(4.0 * k), 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
