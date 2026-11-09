"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0

    # Precompute hexagon vertex offsets for angle=0
    hex_offsets = np.array([[math.cos(math.pi * k / 3), math.sin(math.pi * k / 3)] for k in range(6)])

    def hex_verts_array(cx, cy, angle_deg):
        """Return 6x2 array of vertices for a unit hexagon."""
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        return (R @ hex_offsets.T).T + np.array([cx, cy])

    def compute_min_side_smooth(positions, outer_angle, beta):
        """Vectorized smooth approximation of min outer side length."""
        angle_rad = math.radians(outer_angle)
        cos_a, sin_a = math.cos(-angle_rad), math.sin(-angle_rad)
        normals = np.array([[math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))] for k in range(6)])

        # Collect all vertex projections
        all_projs = []
        for ix, iy, iang in positions:
            verts = hex_verts_array(ix, iy, iang)
            # Rotate into outer frame
            rx = verts[:, 0] * cos_a - verts[:, 1] * sin_a
            ry = verts[:, 0] * sin_a + verts[:, 1] * cos_a
            rot_verts = np.column_stack([rx, ry])
            # Project onto 6 normals: (6x2) @ (2x6) -> (6x6)
            projs = rot_verts @ normals.T  # shape (6, 6)
            all_projs.append(projs.ravel())

        projs = np.concatenate(all_projs)  # shape (36*6, ) = (216,)
        max_p = projs.max()
        exp_sum = np.exp(beta * (projs - max_p)).sum()
        smooth_side = max_p + (1.0 / beta) * math.log(exp_sum)
        return smooth_side / inradius

    def sat_overlap(p1, p2):
        """Return penetration depth > 0 if overlapping, 0 if disjoint."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d >= 2.0:
            return 0.0
        if d < 1e-9:
            return 2.0

        v1 = hex_verts_array(p1[0], p1[1], p1[2])
        v2 = hex_verts_array(p2[0], p2[1], p2[2])

        # Build separation axes: edge normals from both hexagons + center direction
        axes = []
        for verts in (v1, v2):
            for i in range(6):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % 6]
                ex, ey = x2 - x1, y2 - y1
                L = math.sqrt(ex * ex + ey * ey)
                if L > 1e-12:
                    axes.append(np.array([-ey / L, ex / L]))
        axes.append(np.array([dx / d, dy / d]))

        min_gap = float('inf')
        for ax in axes:
            p1proj = v1 @ ax
            p2proj = v2 @ ax
            p1min, p1max = p1proj.min(), p1proj.max()
            p2min, p2max = p2proj.min(), p2proj.max()
            gap = max(p2min - p1max, p1min - p2max)
            if gap > 0:
                return 0.0
            min_gap = min(min_gap, -gap)
        return min_gap if min_gap < float('inf') else 0.0

    def objective(positions, outer_angle, beta, pen_weight=100.0):
        """Objective: Smooth outer side + penalty for overlaps."""
        side = compute_min_side_smooth(positions, outer_angle, beta)
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                p = sat_overlap(positions[i], positions[j])
                if p > 0:
                    pen += p * p
        return side + pen_weight * pen

    def make_honeycomb():
        positions = []
        positions.append([0.0, 0.0, 0.0])
        ring_dist = sqrt3 + 0.1
        for k in range(6):
            a = math.radians(60 * k)
            positions.append([ring_dist * math.cos(a), ring_dist * math.sin(a), 0.0])
        gap_dist = 2 * ring_dist * math.cos(math.pi / 6) + 0.1
        for ad in [30, 150, 210, 330]:
            a = math.radians(ad)
            positions.append([gap_dist * math.cos(a), gap_dist * math.sin(a), 0.0])
        return positions

    def make_ring_init():
        """Alternative init: 11 hexagons in a tighter ring pattern."""
        positions = []
        positions.append([0.0, 0.0, 0.0])
        # 6 at ring 1
        r1 = sqrt3 + 0.05
        for k in range(6):
            a = math.radians(60 * k)
            positions.append([r1 * math.cos(a), r1 * math.sin(a), 0.0])
        # 4 more at slightly larger radius, offset angles
        r2 = 2 * sqrt3 + 0.15
        for ad in [30, 150, 210, 330]:
            a = math.radians(ad)
            positions.append([r2 * math.cos(a), r2 * math.sin(a), 0.0])
        return positions

    def make_random_init():
        """Random perturbation of honeycomb."""
        pos = make_honeycomb()
        for i in range(N):
            pos[i][0] += random.gauss(0, 0.3)
            pos[i][1] += random.gauss(0, 0.3)
            pos[i][2] += random.gauss(0, 10.0)
        return pos

    def to_params(positions):
        return np.array([x for p in positions for x in p])

    def from_params(params):
        return [[params[3 * i], params[3 * i + 1], params[3 * i + 2]] for i in range(N)]

    # Bonds for cluster moves
    bonds = [(0, i) for i in range(1, 7)]
    for i in range(1, 7):
        bonds.append((i, (i % 6) + 1))
    bonds.extend([(7, 1), (7, 2), (8, 3), (8, 4), (9, 4), (9, 5), (10, 6), (10, 1)])

    neighbors = {i: [] for i in range(N)}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    def get_cluster(start_idx, max_size):
        cluster = [start_idx]
        visited = {start_idx}
        frontier = neighbors[start_idx][:]
        while frontier and len(cluster) < max_size:
            if random.random() < 0.7:
                next_idx = random.choice(frontier)
                if next_idx not in visited:
                    cluster.append(next_idx)
                    visited.add(next_idx)
                    frontier.extend(neighbors[next_idx])
                    frontier = [n for n in frontier if n not in visited]
            else:
                break
        return cluster

    def run_sa(init_positions, time_budget):
        """Run SA from given initial positions within time_budget seconds."""
        current_positions = [p[:] for p in init_positions]
        current_obj = objective(current_positions, 0.0, 1.0)
        best_positions = [p[:] for p in current_positions]
        best_obj = current_obj

        temp = 5.0
        beta = 1.0
        steps = 0

        while time.time() < start_time + (deadline - start_time) - time_budget:
            steps += 1
            temp *= 0.997
            beta = min(80.0, beta + 0.015)

            move_type = random.random()
            old_positions = [p[:] for p in current_positions]

            if move_type < 0.45:
                idx = random.randint(0, N - 1)
                dx = random.gauss(0, 0.12)
                dy = random.gauss(0, 0.12)
                current_positions[idx][0] += dx
                current_positions[idx][1] += dy
                current_positions[idx][2] += random.gauss(0, 6.0)
            elif move_type < 0.75:
                idx = random.randint(0, N - 1)
                current_positions[idx][2] += random.gauss(0, 8.0)
            else:
                start_idx = random.randint(0, N - 1)
                cluster = get_cluster(start_idx, 3)
                cx = sum(current_positions[i][0] for i in cluster) / len(cluster)
                cy = sum(current_positions[i][1] for i in cluster) / len(cluster)
                dx = random.gauss(0, 0.12)
                dy = random.gauss(0, 0.12)
                rot = random.gauss(0, 3.0)
                r_rad = math.radians(rot)
                cos_r, sin_r = math.cos(r_rad), math.sin(r_rad)
                for i in cluster:
                    x = current_positions[i][0] - cx
                    y = current_positions[i][1] - cy
                    current_positions[i][0] = x * cos_r - y * sin_r + cx + dx
                    current_positions[i][1] = x * sin_r + y * cos_r + cy + dy
                    current_positions[i][2] += rot

            new_obj = objective(current_positions, 0.0, beta)

            if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / max(temp, 1e-10)):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_positions = [p[:] for p in current_positions]
            else:
                current_positions = old_positions

        return best_positions, best_obj

    def polish(positions, time_budget):
        """Nelder-Mead polish with high beta."""
        beta_final = 150.0

        def nm_objective(params):
            pos = from_params(params)
            return objective(pos, 0.0, beta_final, pen_weight=5000.0)

        params = to_params(positions)
        try:
            from scipy.optimize import minimize
            res = minimize(nm_objective, params, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < float('inf'):
                return from_params(res.x)
        except Exception:
            pass
        return positions

    def fix_overlaps(positions, time_budget):
        """Iteratively separate overlapping hexagons."""
        positions = [p[:] for p in positions]
        for it in range(200):
            if time.time() > start_time + (deadline - start_time) - time_budget:
                break
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    pen = sat_overlap(positions[i], positions[j])
                    if pen > 1e-6:
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        d = math.sqrt(dx * dx + dy * dy)
                        if d < 1e-9:
                            dx, dy = 1.0, 0.0
                            d = 1.0
                        push = pen * 0.5 + 0.005
                        positions[i][0] += push * dx / d
                        positions[i][1] += push * dy / d
                        positions[j][0] -= push * dx / d
                        positions[j][1] -= push * dy / d
                        moved = True
            if not moved:
                break
        return positions

    def compute_exact_side(positions, outer_angle):
        angle_rad = math.radians(outer_angle)
        cos_a, sin_a = math.cos(-angle_rad), math.sin(-angle_rad)
        normals = np.array([[math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))] for k in range(6)])
        max_side = 0.0
        for ix, iy, iang in positions:
            verts = hex_verts_array(ix, iy, iang)
            rx = verts[:, 0] * cos_a - verts[:, 1] * sin_a
            ry = verts[:, 0] * sin_a + verts[:, 1] * cos_a
            rot_verts = np.column_stack([rx, ry])
            projs = (rot_verts @ normals.T).max()
            req = projs / inradius
            if req > max_side:
                max_side = req
        return max_side

    def optimize_outer_angle(positions, time_budget):
        best_side = float('inf')
        best_angle = 0.0
        # Coarse
        for ang in range(0, 31, 2):
            s = compute_exact_side(positions, float(ang))
            if s < best_side:
                best_side = s
                best_angle = float(ang)
        # Fine
        for _ in range(300):
            if time.time() > start_time + (deadline - start_time) - time_budget:
                break
            trial = best_angle + random.gauss(0, 0.4)
            s = compute_exact_side(positions, trial)
            if s < best_side:
                best_side = s
                best_angle = trial
        # Even finer
        for _ in range(200):
            if time.time() > start_time + (deadline - start_time) - time_budget:
                break
            trial = best_angle + random.gauss(0, 0.1)
            s = compute_exact_side(positions, trial)
            if s < best_side:
                best_side = s
                best_angle = trial
        return best_side, best_angle

    # --- Multi-start SA ---
    inits = [
        make_honeycomb(),
        make_ring_init(),
        make_random_init(),
        make_random_init(),
    ]

    # Allocate time: ~100s for SA, ~20s for polish, ~5s for overlap fix, ~5s for angle
    sa_time_budget = 95.0
    best_overall_positions = None
    best_overall_obj = float('inf')

    for init in inits:
        if time.time() > start_time + (deadline - start_time) - 25.0:
            break
        bp, bo = run_sa(init, sa_time_budget)
        if bo < best_overall_obj:
            best_overall_obj = bo
            best_overall_positions = bp

    if best_overall_positions is None:
        best_overall_positions = make_honeycomb()

    # Polish
    polished = polish(best_overall_positions, 15.0)

    # Fix overlaps
    fixed = fix_overlaps(polished, 5.0)

    # Optimize outer angle
    best_side, best_angle = optimize_outer_angle(fixed, 3.0)

    # Final validity check
    final_pen = sum(sat_overlap(fixed[i], fixed[j]) for i in range(N) for j in range(i + 1, N))
    if final_pen > 1e-4:
        # Fallback
        safe = make_honeycomb()
        best_side, best_angle = optimize_outer_angle(safe, 1.0)
        fixed = safe

    return fixed, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
