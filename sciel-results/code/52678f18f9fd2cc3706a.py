"""Lattice-based combinatorial packing: place 12 unit hexagons on a honeycomb
lattice (6 orientations) and minimize the enclosing regular-hexagon side via
support-function caging. Optimized with simulated annealing + local refinement."""
import numpy as np
import time
import math

N = 12
SQRT3 = math.sqrt(3.0)
TWO_PI = 2.0 * math.pi
EPS = 1e-12


def construct():
    t0 = time.time()
    DEADLINE = t0 + 128.0  # safety margin before 150 s kill

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def hex_vertices(cx, cy, side, angle):
        a = angle
        return (
            [cx + side * math.cos(a + k * math.pi / 3.0) for k in range(6)],
            [cy + side * math.sin(a + k * math.pi / 3.0) for k in range(6)],
        )

    def outer_side(xs, ys, outer_cx, outer_cy, outer_angle):
        """Minimum regular-hexagon side (with given center/rotation) whose
        6 edge-slabs contain every point (xs[k], ys[k])."""
        max_req = 0.0
        for k in range(6):
            nx = math.cos(outer_angle + k * math.pi / 3.0)
            ny = math.sin(outer_angle + k * math.pi / 3.0)
            for i in range(len(xs)):
                d = (xs[i] - outer_cx) * nx + (ys[i] - outer_cy) * ny
                if d > 0:
                    req = d * 2.0 / SQRT3
                    if req > max_req:
                        max_req = req
        return max_req

    def sat_disjoint(px1, py1, px2, py2, tol=1e-9):
        """Return True iff two convex hexagons (given as 6 vertex lists) are
        disjoint (interiors do not overlap)."""
        for poly_x, poly_y in ((px1, py1), (px2, py2)):
            for i in range(6):
                j = (i + 1) % 6
                ex = poly_x[j] - poly_x[i]
                ey = poly_y[j] - poly_y[i]
                nx = -ey
                ny = ex
                nrm = math.hypot(nx, ny)
                if nrm < 1e-12:
                    continue
                nx /= nrm
                ny /= nrm
                p1 = [poly_x[a] * nx + poly_y[a] * ny for a in range(6)]
                p2 = [px2[a] * nx + py2[a] * ny for a in range(6)]
                if max(p1) < min(p2) - tol or max(p2) < min(p1) - tol:
                    return True
        return False

    # ------------------------------------------------------------------
    # Lattice definition (honeycomb / triangular lattice)
    # ------------------------------------------------------------------
    # Lattice vectors
    a1x, a1y = SQRT3, 0.0          # horizontal spacing
    a2x, a2y = SQRT3 / 2.0, 1.5    # 60° direction

    # Build a pool of lattice sites (i, j) -> (x, y)
    MAX_IJ = 8  # sites with i,j in [-MAX_IJ, MAX_IJ]
    sites = []
    site_index = {}
    for i in range(-MAX_IJ, MAX_IJ + 1):
        for j in range(-MAX_IJ, MAX_IJ + 1):
            x = i * a1x + j * a2x
            y = i * a1y + j * a2y
            site_index[(i, j)] = len(sites)
            sites.append((x, y))
    N_SITES = len(sites)

    # Orientation set: 6 multiples of 30° (full symmetry of hexagon)
    N_ORIENT = 6
    ORIENTS = [k * math.pi / 6.0 for k in range(N_ORIENT)]

    # ------------------------------------------------------------------
    # State: each hexagon = (site_idx, orient_idx)
    # ------------------------------------------------------------------
    def precompute_verts(state):
        """Return list of (px, py) vertex-lists for all N hexagons."""
        verts = []
        for s in state:
            si, oi = s
            cx, cy = sites[si]
            a = ORIENTS[oi]
            px, py = hex_vertices(cx, cy, 1.0, a)
            verts.append((px, py))
        return verts

    def all_disjoint(verts):
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(verts[i][0], verts[i][1], verts[j][0], verts[j][1]):
                    return False
        return True

    def objective(state):
        """Outer side length if configuration is valid, else large penalty."""
        verts = precompute_verts(state)
        if not all_disjoint(verts):
            return 1e6
        # Collect all vertices
        all_x = []
        all_y = []
        for px, py in verts:
            all_x.extend(px)
            all_y.extend(py)
        # Center = centroid of all vertices (optimal for symmetric outer hex)
        cx = sum(all_x) / len(all_x)
        cy = sum(all_y) / len(all_y)
        # Best outer angle: try 6 symmetric orientations
        best = 1e6
        for oa in ORIENTS:
            s = outer_side(all_x, all_y, cx, cy, oa)
            if s < best:
                best = s
        return best

    # ------------------------------------------------------------------
    # Initial configurations
    # ------------------------------------------------------------------
    def make_honeycomb():
        """3x4 honeycomb pattern mapped to nearest lattice sites."""
        state = []
        for r in range(3):
            for c in range(4):
                tx = c * SQRT3 * 1.02 + (r % 2) * SQRT3 / 2.0
                ty = r * 1.52
                # Find nearest lattice site
                best_d = 1e9
                best_si = 0
                for si, (sx, sy) in enumerate(sites):
                    d = (sx - tx) ** 2 + (sy - ty) ** 2
                    if d < best_d:
                        best_d = d
                        best_si = si
                state.append((best_si, 0))
        return state

    def make_ring():
        """6 inner + 6 outer ring."""
        state = []
        for i in range(6):
            ang = TWO_PI * i / 6.0
            tx = 1.8 * math.cos(ang)
            ty = 1.8 * math.sin(ang)
            best_d = 1e9
            best_si = 0
            for si, (sx, sy) in enumerate(sites):
                d = (sx - tx) ** 2 + (sy - ty) ** 2
                if d < best_d:
                    best_d = d
                    best_si = si
            state.append((best_si, 0))
        for i in range(6):
            ang = TWO_PI * i / 6.0 + math.pi / 6.0
            tx = 3.2 * math.cos(ang)
            ty = 3.2 * math.sin(ang)
            best_d = 1e9
            best_si = 0
            for si, (sx, sy) in enumerate(sites):
                d = (sx - tx) ** 2 + (sy - ty) ** 2
                if d < best_d:
                    best_d = d
                    best_si = si
            state.append((best_si, 0))
        return state

    def make_random():
        """Random distinct sites, random orientations."""
        used = set()
        state = []
        attempts = 0
        while len(state) < N and attempts < 5000:
            attempts += 1
            si = np.random.randint(0, N_SITES)
            if si in used:
                continue
            used.add(si)
            oi = np.random.randint(0, N_ORIENT)
            state.append((si, oi))
        # If not enough, fill with far-apart sites
        while len(state) < N:
            state.append((len(state) % N_SITES, 0))
        return state

    # ------------------------------------------------------------------
    # Simulated Annealing
    # ------------------------------------------------------------------
    def simulated_annealing(init_state, t_start, t_end, n_iter):
        state = list(init_state)
        verts = precompute_verts(state)
        cur_obj = objective(state)
        best_state = list(state)
        best_obj = cur_obj

        for it in range(n_iter):
            if time.time() > DEADLINE - 5:
                break
            # Temperature schedule (geometric)
            frac = it / max(1, n_iter - 1)
            T = t_start * (t_end / t_start) ** frac

            # Choose a random hexagon to move
            idx = np.random.randint(0, N)
            old_si, old_oi = state[idx]

            # Choose new site (local move) and orientation
            new_si = old_si
            new_oi = old_oi

            move_type = np.random.random()
            if move_type < 0.6:
                # Move to a nearby site
                # Pick a random offset in (i,j) space
                di = np.random.randint(-2, 3)
                dj = np.random.randint(-2, 3)
                if di == 0 and dj == 0:
                    di = 1
                # Find the site index for (old_i + di, old_j + dj)
                # We need to reverse-map: store (i,j) per site
                # Simpler: pick a random site within small distance
                cx_old, cy_old = sites[old_si]
                candidates = []
                for si, (sx, sy) in enumerate(sites):
                    d = math.hypot(sx - cx_old, sy - cy_old)
                    if d < 4.0 and si != old_si:
                        candidates.append(si)
                if candidates:
                    new_si = np.random.choice(candidates)
            elif move_type < 0.85:
                # Change orientation
                new_oi = np.random.randint(0, N_ORIENT)
            else:
                # Both
                cx_old, cy_old = sites[old_si]
                candidates = []
                for si, (sx, sy) in enumerate(sites):
                    d = math.hypot(sx - cx_old, sy - cy_old)
                    if d < 5.0 and si != old_si:
                        candidates.append(si)
                if candidates:
                    new_si = np.random.choice(candidates)
                new_oi = np.random.randint(0, N_ORIENT)

            state[idx] = (new_si, new_oi)
            new_obj = objective(state)

            delta = new_obj - cur_obj
            if delta < 0 or (T > 1e-9 and np.random.random() < math.exp(-delta / T)):
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_state = list(state)
            else:
                state[idx] = (old_si, old_oi)

        return best_state, best_obj

    # ------------------------------------------------------------------
    # Local refinement: try all single-hexagon moves to improve
    # ------------------------------------------------------------------
    def local_refine(state, max_rounds=50):
        state = list(state)
        cur_obj = objective(state)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < DEADLINE - 5:
            improved = False
            rounds += 1
            for idx in range(N):
                old_si, old_oi = state[idx]
                best_local = (old_si, old_oi)
                best_local_obj = cur_obj
                # Try all nearby sites and orientations
                cx_old, cy_old = sites[old_si]
                for si, (sx, sy) in enumerate(sites):
                    if si == old_si:
                        continue
                    if math.hypot(sx - cx_old, sy - cy_old) > 3.0:
                        continue
                    for oi in range(N_ORIENT):
                        state[idx] = (si, oi)
                        obj = objective(state)
                        if obj < best_local_obj:
                            best_local_obj = obj
                            best_local = (si, oi)
                state[idx] = best_local
                if best_local != (old_si, old_oi):
                    cur_obj = best_local_obj
                    improved = True
        return state, cur_obj

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    np.random.seed(12345)

    inits = [make_honeycomb(), make_ring(), make_random(), make_random()]
    best_state = None
    best_obj = 1e9

    for init in inits:
        if time.time() > DEADLINE - 20:
            break
        st, obj = simulated_annealing(init, t_start=2.0, t_end=0.01, n_iter=3000)
        if obj < best_obj:
            best_obj = obj
            best_state = st

    # Local refinement on best
    if best_state is not None and time.time() < DEADLINE - 10:
        best_state, best_obj = local_refine(best_state, max_rounds=30)

    # Final SA polish with lower temperature
    if best_state is not None and time.time() < DEADLINE - 8:
        st, obj = simulated_annealing(best_state, t_start=0.5, t_end=0.001, n_iter=2000)
        if obj < best_obj:
            best_obj = obj
            best_state = st
        if time.time() < DEADLINE - 5:
            best_state, best_obj = local_refine(best_state, max_rounds=20)

    # ------------------------------------------------------------------
    # Extract result
    # ------------------------------------------------------------------
    if best_state is None or best_obj >= 1e5:
        # Fallback: widely spaced
        inner = []
        for i in range(N):
            inner.append([float(i * 3.5), 0.0, 0.0])
        return inner, (0.0, 0.0), 25.0, 0.0

    # Build vertex list
    verts = precompute_verts(best_state)
    all_x = []
    all_y = []
    for px, py in verts:
        all_x.extend(px)
        all_y.extend(py)

    cx = sum(all_x) / len(all_x)
    cy = sum(all_y) / len(all_y)

    # Find best outer angle
    best_side = 1e9
    best_oa = 0.0
    for oa in ORIENTS:
        s = outer_side(all_x, all_y, cx, cy, oa)
        if s < best_side:
            best_side = s
            best_oa = oa

    # Build inner list
    inner = []
    for si, oi in best_state:
        sx, sy = sites[si]
        a_deg = math.degrees(ORIENTS[oi])
        inner.append([float(sx), float(sy), float(a_deg)])

    return inner, (float(cx), float(cy)), float(best_side), float(math.degrees(best_oa))
