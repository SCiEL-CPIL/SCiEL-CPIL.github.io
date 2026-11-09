"""Pack 11 unit regular hexagons inside a regular hexagon, minimizing outer side length.
Uses rigid-cluster decomposition with differential evolution + Nelder-Mead polishing."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin (evaluator kills at 150s)
    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0  # inradius of unit hexagon

    # Pre-compute unit hexagon vertices (side=1, circumradius=1)
    unit_hex_verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        unit_hex_verts.append((math.cos(theta), math.sin(theta)))
    unit_hex_verts = np.array(unit_hex_verts)  # shape (6, 2)

    # Pre-compute edge normals for unit hexagon (outward)
    hex_normals_unit = []
    for i in range(6):
        ex = unit_hex_verts[(i + 1) % 6, 0] - unit_hex_verts[i, 0]
        ey = unit_hex_verts[(i + 1) % 6, 1] - unit_hex_verts[i, 1]
        nx, ny = -ey, ex
        L = math.sqrt(nx * nx + ny * ny)
        hex_normals_unit.append((nx / L, ny / L))
    hex_normals_unit = np.array(hex_normals_unit)  # shape (6, 2)

    # Outer hexagon normals (for containment check) in local frame
    outer_normals = np.array([[math.cos(math.radians(30 + k * 60)),
                                math.sin(math.radians(30 + k * 60))]
                               for k in range(6)])  # shape (6, 2)

    def get_hex_verts(cx, cy, angle_rad):
        """Vertices of unit hexagon centered at (cx,cy) rotated by angle_rad."""
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return cx + unit_hex_verts[:, 0] * c - unit_hex_verts[:, 1] * s, \
               cy + unit_hex_verts[:, 0] * s + unit_hex_verts[:, 1] * c

    def get_hex_verts_arr(cx, cy, angle_rad):
        """Returns (6,2) array of vertices."""
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        vx = cx + unit_hex_verts[:, 0] * c - unit_hex_verts[:, 1] * s
        vy = cy + unit_hex_verts[:, 0] * s + unit_hex_verts[:, 1] * c
        return np.column_stack((vx, vy))

    def get_rotated_normals(angle_rad):
        """Rotate unit hex normals by angle_rad, returns (6,2)."""
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        nx = hex_normals_unit[:, 0] * c - hex_normals_unit[:, 1] * s
        ny = hex_normals_unit[:, 0] * s + hex_normals_unit[:, 1] * c
        return np.column_stack((nx, ny))

    def min_side_for_config(positions, outer_center, outer_angle_deg):
        """Compute minimum outer side length to contain all inner hexagons.
        positions: list of (x, y, angle_deg)
        Returns the required side length (or inf if any hex is outside a huge bound)."""
        ox, oy = outer_center
        outer_angle_rad = math.radians(outer_angle_deg)
        oc = math.cos(-outer_angle_rad)
        os_ = math.sin(-outer_angle_rad)
        max_side = 0.0
        for (px, py, pad) in positions:
            par = math.radians(pad)
            verts = get_hex_verts_arr(px, py, par)
            for vx, vy in verts:
                dx = vx - ox
                dy = vy - oy
                rx = dx * oc - dy * os_
                ry = dx * os_ + dy * oc
                # Project onto 6 outer normals
                projs = rx * outer_normals[:, 0] + ry * outer_normals[:, 1]
                ms = projs.max()
                side = ms / inradius
                if side > max_side:
                    max_side = side
        return max_side + 1e-12

    def overlap_penalty_fast(p1, p2):
        """Return a smooth overlap penalty between two unit hexagons.
        Uses SAT with minimum separation distance.
        Returns 0 if separated, positive if overlapping (penalty)."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx * dx + dy * dy
        if d2 > 4.0:  # max contact distance is 2 (both circumradius 1)
            return 0.0

        a1 = math.radians(p1[2])
        a2 = math.radians(p2[2])
        v1 = get_hex_verts_arr(p1[0], p1[1], a1)
        v2 = get_hex_verts_arr(p2[0], p2[1], a2)
        n1 = get_rotated_normals(a1)
        n2 = get_rotated_normals(a2)

        min_sep = float('inf')
        for normals, verts_a, verts_b in ((n1, v1, v2), (n2, v1, v2)):
            for ni in range(6):
                nx, ny = normals[ni]
                proj_a = verts_a[:, 0] * nx + verts_a[:, 1] * ny
                proj_b = verts_b[:, 0] * nx + verts_b[:, 1] * ny
                max_a = proj_a.max()
                min_a = proj_a.min()
                max_b = proj_b.max()
                min_b = proj_b.min()
                sep = max_a - min_b  # if this is negative, overlap along this axis
                # separation = min_b - max_a  (positive means gap)
                gap = min_b - max_a
                if gap < 0:
                    # overlap in this direction
                    min_sep = min(min_sep, 0.0)
                    # We can early-exit: if any axis has overlap, they overlap
                    # But for smooth penalty, we want the maximum penetration
                    # Actually for SAT: if NO axis separates, they overlap.
                    # For smooth penalty: use the minimum gap (most negative = deepest overlap)
                    # We want: if all gaps >= 0 -> no overlap. Otherwise penalty.
                    # Let's collect all gaps.
                    pass
                else:
                    min_sep = min(min_sep, gap)

        # Re-do: collect all 12 gaps
        gaps = []
        for normals, verts_a, verts_b in ((n1, v1, v2), (n2, v1, v2)):
            for ni in range(6):
                nx, ny = normals[ni]
                proj_a = verts_a[:, 0] * nx + verts_a[:, 1] * ny
                proj_b = verts_b[:, 0] * nx + verts_b[:, 1] * ny
                gap = proj_b.min() - proj_a.max()
                gaps.append(gap)

        min_gap = min(gaps)
        if min_gap >= 0:
            return 0.0
        else:
            # Smooth penalty: square of penetration depth
            return min_gap * min_gap

    def total_overlap_penalty(positions):
        """Sum of pairwise overlap penalties."""
        total = 0.0
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                total += overlap_penalty_fast(positions[i], positions[j])
        return total

    def is_valid(positions):
        """Check all pairs are non-overlapping (hard constraint)."""
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = positions[i], positions[j]
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if dx * dx + dy * dy > 4.0:
                    continue
                a1 = math.radians(p1[2])
                a2 = math.radians(p2[2])
                v1 = get_hex_verts_arr(p1[0], p1[1], a1)
                v2 = get_hex_verts_arr(p2[0], p2[1], a2)
                n1 = get_rotated_normals(a1)
                n2 = get_rotated_normals(a2)
                for normals, va, vb in ((n1, v1, v2), (n2, v1, v2)):
                    separated = False
                    for ni in range(6):
                        nx, ny = normals[ni]
                        if (vb[:, 0] * nx + vb[:, 1] * ny).min() <= (va[:, 0] * nx + va[:, 1] * ny).max() + 1e-9:
                            continue
                        separated = True
                        break
                    if separated:
                        break
                else:
                    # No separating axis found for this pair -> overlap
                    return False
        return True

    # --- Rigid cluster decomposition ---
    # Hex grid: axial coordinates (q, r)
    # Axial to Cartesian for unit hexagons:
    # x = sqrt(3) * (q + r/2), y = 1.5 * r  ... wait
    # Standard: x = size * sqrt(3) * (q + r/2), y = size * 3/2 * r
    # With size=1: x = sqrt(3)*(q + r/2), y = 1.5*r
    # But we want center-to-center distance = 2*inradius = sqrt(3) for adjacent hexagons.
    # Let's verify: (0,0) and (1,0): dist = sqrt(3). Good.
    # (0,0) and (0,1): dist = sqrt(3*(0+0.5)^2 + 1.5^2) = sqrt(0.75+2.25)=sqrt(3). Good.

    d = sqrt3  # center-to-center distance for touching hexagons

    def axial_to_cart(q, r):
        x = d * (q + r / 2.0)
        y = d * 1.5 * r
        return x, y

    # 11 hexagons in a compact arrangement:
    # Center + 6 neighbors + 4 from next ring (to keep compact)
    axial_coords = [
        (0, 0),       # 0: center
        (1, 0),       # 1
        (0, 1),       # 2
        (-1, 1),      # 3
        (-1, 0),      # 4
        (0, -1),      # 5
        (1, -1),      # 6
        (2, 0),       # 7
        (1, 1),       # 8
        (-2, 1),      # 9
        (-1, -1),     # 10
    ]

    base_positions = [axial_to_cart(q, r) for q, r in axial_coords]

    # Clusters: group hexagons that move together
    # C0: {0} - center
    # C1: {1, 2, 3, 4} - upper-left quadrant neighbors
    # C2: {5, 6, 7, 8} - lower-right quadrant neighbors
    # C3: {9, 10} - two remaining
    clusters = [
        [0],
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10]
    ]
    num_clusters = len(clusters)

    # Each cluster has state: [dx, dy, angle_deg]
    # Initial state: all zero (identity)
    init_cluster_states = [[0.0, 0.0, 0.0] for _ in range(num_clusters)]

    def get_positions_from_clusters(cluster_states):
        """Compute absolute positions of all N hexagons."""
        positions = [None] * N
        for ci, (dx, dy, ang_deg) in enumerate(cluster_states):
            ang_rad = math.radians(ang_deg)
            c = math.cos(ang_rad)
            s = math.sin(ang_rad)
            for hi in clusters[ci]:
                bx, by = base_positions[hi]
                rx = bx * c - by * s
                ry = bx * s + by * c
                positions[hi] = (rx + dx, ry + dy, ang_deg)
        return positions

    def objective(params):
        """Full objective: outer_side + large_penalty * overlap_penalty.
        params: [dx0, dy0, ang0, dx1, dy1, ang1, ..., outer_angle]
        """
        cluster_states = []
        for ci in range(num_clusters):
            base = ci * 3
            cluster_states.append([params[base], params[base + 1], params[base + 2]])
        outer_angle = params[-1]

        positions = get_positions_from_clusters(cluster_states)
        overlap_pen = total_overlap_penalty(positions)
        side = min_side_for_config(positions, (0.0, 0.0), outer_angle)
        penalty = 100.0 * overlap_pen
        return side + penalty

    # --- Multi-start with Differential Evolution + Nelder-Mead ---
    def run_de(start_params, bounds, max_iter=200, pop_size=30):
        """Simple differential evolution for the given parameter vector."""
        n = len(start_params)
        # Population
        pop = [start_params[:]]
        for _ in range(pop_size - 1):
            ind = []
            for i in range(n):
                lo, hi = bounds[i]
                val = random.uniform(lo, hi)
                ind.append(val)
            pop.append(ind)

        best = start_params[:]
        best_f = objective(best)

        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        for gen in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            for i in range(pop_size):
                if time.time() > deadline - 1.0:
                    break
                # Select three distinct individuals
                a, b, c = random.sample([j for j in range(pop_size) if j != i], 3)
                # Create trial vector
                trial = []
                for j in range(n):
                    if random.random() < CR:
                        trial.append(pop[a][j] + F * (pop[b][j] - pop[c][j]))
                    else:
                        trial.append(pop[i][j])
                # Clamp to bounds
                for j in range(n):
                    lo, hi = bounds[j]
                    if trial[j] < lo:
                        trial[j] = lo
                    elif trial[j] > hi:
                        trial[j] = hi
                f_trial = objective(trial)
                f_current = objective(pop[i])
                if f_trial < f_current:
                    pop[i] = trial
                    if f_trial < best_f:
                        best = trial[:]
                        best_f = f_trial
        return best, best_f

    def nelder_mead(x0, bounds, max_iter=300, tol=1e-8):
        """Nelder-Mead simplex optimization."""
        n = len(x0)
        # Build initial simplex
        simplex = [x0[:]]
        for i in range(n):
            perturbation = [v[:] for v in [x0]]
            pt = x0[:]
            step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 0.01 else 0.1
            pt[i] += step
            simplex.append(pt)

        # Evaluate
        fvals = [objective(s) for s in simplex]

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for it in range(max_iter):
            if time.time() > deadline - 1.0:
                break
            # Sort
            order = sorted(range(n + 1), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if fvals[-1] - fvals[0] < tol:
                break

            # Centroid of all but worst
            centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

            # Reflection
            xr = [centroid[j] + alpha * (centroid[j] - simplex[n][j]) for j in range(n)]
            # Clamp
            for j in range(n):
                lo, hi = bounds[j]
                xr[j] = max(lo, min(hi, xr[j]))
            fr = objective(xr)

            if fr < fvals[0]:
                # Expansion
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
                for j in range(n):
                    lo, hi = bounds[j]
                    xe[j] = max(lo, min(hi, xe[j]))
                fe = objective(xe)
                if fe < fr:
                    simplex[n] = xe
                    fvals[n] = fe
                else:
                    simplex[n] = xr
                    fvals[n] = fr
            elif fr < fvals[-1]:
                simplex[n] = xr
                fvals[n] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = [centroid[j] + rho * (simplex[n][j] - centroid[j]) for j in range(n)]
                else:
                    # Inside contraction
                    xc = [centroid[j] + rho * (centroid[j] - simplex[n][j]) for j in range(n)]
                for j in range(n):
                    lo, hi = bounds[j]
                    xc[j] = max(lo, min(hi, xc[j]))
                fc = objective(xc)
                if fc < min(fr, fvals[-1]):
                    simplex[n] = xc
                    fvals[n] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [(simplex[0][j] + simplex[i][j]) / 2 for j in range(n)]
                        fvals[i] = objective(simplex[i])
        # Return best
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        return simplex[order[0]], fvals[order[0]]

    # Parameter bounds
    # Cluster translations: roughly -4 to 4
    # Cluster angles: -180 to 180
    # Outer angle: -180 to 180
    bounds = []
    for ci in range(num_clusters):
        bounds.append((-4.0, 4.0))  # dx
        bounds.append((-4.0, 4.0))  # dy
        bounds.append((-180.0, 180.0))  # angle
    bounds.append((-180.0, 180.0))  # outer angle

    def params_to_cluster_states(params):
        states = []
        for ci in range(num_clusters):
            base = ci * 3
            states.append([params[base], params[base + 1], params[base + 2]])
        return states

    def get_final_result(cluster_states, outer_angle):
        """Compute the actual side length and return the configuration."""
        positions = get_positions_from_clusters(cluster_states)
        # Verify validity
        if not is_valid(positions):
            return None
        side = min_side_for_config(positions, (0.0, 0.0), outer_angle)
        return positions, (0.0, 0.0), side, outer_angle

    # Multi-start
    best_result = None
    best_side = float('inf')

    num_restarts = 8
    max_restarts = 15

    for restart in range(max_restarts):
        if time.time() > deadline - 20.0:
            break

        # Initial random parameters
        init_params = []
        for ci in range(num_clusters):
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            ang = random.gauss(0, 30)
            init_params.extend([dx, dy, ang])
        init_params.append(random.gauss(0, 30))  # outer angle

        # DE phase
        t0 = time.time()
        de_best, de_f = run_de(init_params, bounds, max_iter=150, pop_size=25)
        de_time = time.time() - t0

        # NM polish
        t0 = time.time()
        nm_best, nm_f = nelder_mead(de_best, bounds, max_iter=200)
        nm_time = time.time() - t0

        # Evaluate
        cluster_states = params_to_cluster_states(nm_best)
        outer_angle = nm_best[-1]
        result = get_final_result(cluster_states, outer_angle)
        if result is not None:
            positions, center, side, oang = result
            if side < best_side:
                best_side = side
                best_result = result

        # Also try the DE best
        cluster_states_de = params_to_cluster_states(de_best)
        outer_angle_de = de_best[-1]
        result_de = get_final_result(cluster_states_de, outer_angle_de)
        if result_de is not None:
            positions, center, side, oang = result_de
            if side < best_side:
                best_side = side
                best_result = result_de

        # Check if we have a good result already
        if best_side < 2.8:
            break

    # Final polish: try to optimize outer angle for the best configuration
    if best_result is not None:
        positions, center, side, oang = best_result
        # Try several outer angles
        best_side_final = side
        best_oang_final = oang
        for candidate_ang in [oang + da for da in np.arange(-15, 15.1, 1.0)]:
            s = min_side_for_config(positions, (0.0, 0.0), candidate_ang)
            if s < best_side_final:
                best_side_final = s
                best_oang_final = candidate_ang
        best_result = (positions, center, best_side_final, best_oang_final)
        best_side = best_side_final

    # Fallback: if no valid result found, create a spread-out configuration
    if best_result is None or math.isinf(best_side):
        positions = []
        positions.append((0.0, 0.0, 0.0))
        for i in range(1, N):
            ring = 1 if i <= 6 else 2
            idx_in_ring = i - 1 if i <= 6 else i - 7
            angle = math.radians(60 * idx_in_ring + (30 if ring == 2 else 30))
            r = ring * d
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions.append((x, y, 0.0))
        best_side = min_side_for_config(positions, (0.0, 0.0), 0.0)
        best_result = (positions, (0.0, 0.0), best_side, 0.0)

    return best_result
# EVOLVE-BLOCK-END
