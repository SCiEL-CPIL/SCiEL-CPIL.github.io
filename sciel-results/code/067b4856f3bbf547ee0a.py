import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """Return the minimum separation (negative = overlap) between two convex polygons."""
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        return np.min(overlaps)

    # Precompute outer hexagon normals (fixed orientation: pointy-top)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    outer_normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    def compute_min_side(hexes):
        """Compute the minimum outer hexagon side length to contain all inner hexagons."""
        max_proj = 0.0
        for h in hexes:
            proj = h @ outer_normals.T
            max_proj = max(max_proj, np.max(proj))
        return max_proj * 2.0 / np.sqrt(3.0)

    def generate_hexes(params):
        """Generate 12 hexagons from 6 params using 6-fold symmetry (2 orbits of 6)."""
        x1, y1, a1, x2, y2, a2 = params
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s = np.sin(angle)

            x1r = x1 * c - y1 * s
            y1r = x1 * s + y1 * c
            a1r = a1 + angle

            x2r = x2 * c - y2 * s
            y2r = x2 * s + y2 * c
            a2r = a2 + angle

            hexes.append(hex_vertices(x1r, y1r, 1.0, a1r))
            hexes.append(hex_vertices(x2r, y2r, 1.0, a2r))
        return hexes

    def compute_smooth_min_side(hexes, beta=50.0):
        """Soft-max approximation of max projection for smoother gradient."""
        all_projs = []
        for h in hexes:
            proj = h @ outer_normals.T
            all_projs.append(proj)
        all_projs = np.concatenate(all_projs, axis=1)  # shape: (6, 72)
        # Soft max over all 72 projections
        scaled = beta * all_projs
        max_scaled = np.max(scaled, axis=0)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled), axis=0)) / beta) / beta
        return np.max(soft_max) * 2.0 / np.sqrt(3.0)

    def compute_smooth_overlap_penalty(hexes, k=20.0):
        """Smooth penalty for overlaps using softplus."""
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick AABB check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > -1e-6:
                    # Softplus: log(1 + exp(k * overlap))
                    if k * overlap > 50:
                        penalty += k * overlap
                    else:
                        penalty += np.log1p(np.exp(k * overlap))
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        hexes = generate_hexes(params)
        s_min = compute_min_side(hexes)

        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 20.0
        return s_min + penalty

    def smooth_objective(params, beta=30.0, k=15.0):
        """Fully smooth objective for gradient-based methods."""
        if time.time() > deadline:
            return 1e10
        hexes = generate_hexes(params)
        s_min = compute_smooth_min_side(hexes, beta=beta)
        penalty = compute_smooth_overlap_penalty(hexes, k=k)
        return s_min + penalty

    def smooth_objective_tight(params):
        """Tighter smooth objective for polishing."""
        if time.time() > deadline:
            return 1e10
        hexes = generate_hexes(params)
        s_min = compute_smooth_min_side(hexes, beta=100.0)
        penalty = compute_smooth_overlap_penalty(hexes, k=50.0)
        return s_min + penalty

    # Bounds for the 6 free parameters
    bounds = [
        (-5.0, 5.0),  # x1
        (-5.0, 5.0),  # y1
        (0.0, np.pi/3),  # a1
        (-5.0, 5.0),  # x2
        (-5.0, 5.0),  # y2
        (0.0, np.pi/3),  # a2
    ]

    best_s = np.inf
    best_params = None

    # Phase 1: Global search with differential evolution using smooth objective
    try:
        res = differential_evolution(
            lambda p: smooth_objective(p, beta=30.0, k=15.0),
            bounds,
            maxiter=300,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=False,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x.copy()
    except Exception:
        pass

    # Phase 2: Local refinement from best DE result
    if best_params is not None:
        try:
            res_local = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 2000}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass

    # Phase 3: Multiple random restarts with local optimization
    if time.time() < deadline - 20:
        np.random.seed(123)
        n_restarts = 15
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break

            if best_params is not None and r % 3 == 0:
                # Perturb best
                p = best_params + np.random.normal(0, 0.2, 6)
            elif r % 3 == 1:
                # Hex lattice inspired
                spacing = 2.0 * np.sqrt(3.0)
                r1 = np.random.uniform(0.5, spacing * 0.8)
                a1 = np.random.uniform(0, 2 * np.pi)
                r2 = np.random.uniform(spacing * 0.3, spacing * 1.2)
                a2 = np.random.uniform(0, 2 * np.pi)
                rot1 = np.random.uniform(0, np.pi/3)
                rot2 = np.random.uniform(0, np.pi/3)
                p = np.array([r1 * np.cos(a1), r1 * np.sin(a1), rot1,
                              r2 * np.cos(a2), r2 * np.sin(a2), rot2])
            else:
                # Concentric ring inspired
                r1 = np.random.uniform(0.3, 1.5)
                a1 = np.random.uniform(0, 2 * np.pi)
                r2 = np.random.uniform(1.5, 3.0)
                a2 = np.random.uniform(0, 2 * np.pi)
                rot1 = np.random.uniform(0, np.pi/3)
                rot2 = np.random.uniform(0, np.pi/3)
                p = np.array([r1 * np.cos(a1), r1 * np.sin(a1), rot1,
                              r2 * np.cos(a2), r2 * np.sin(a2), rot2])

            p = np.clip(p, [-5, -5, 0, -5, -5, 0], [5, 5, np.pi/3, 5, 5, np.pi/3])

            try:
                # First use smooth objective for better gradient
                res_smooth = minimize(
                    lambda p_: smooth_objective(p_, beta=50.0, k=25.0),
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 1000}
                )
                # Then refine with exact objective
                res_local = minimize(
                    objective,
                    res_smooth.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 1500}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x.copy()
            except Exception:
                pass

    # Phase 4: Final high-precision polish with smooth objective
    if best_params is not None and time.time() < deadline - 3:
        try:
            res_smooth_polish = minimize(
                smooth_objective_tight,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxfun': 10000}
            )
            if res_smooth_polish.fun < best_s:
                best_s = res_smooth_polish.fun
                best_params = res_smooth_polish.x.copy()
        except Exception:
            pass

    # Phase 5: Exact final polish
    if best_params is not None and time.time() < deadline - 1:
        try:
            res_polish = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxfun': 10000}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = np.array([1.5, 0.0, 0.0, 2.5, 0.0, 0.0])

    # Generate final configuration
    hexes = generate_hexes(best_params)
    inner = []
    for h in hexes:
        cx = np.mean(h[:, 0])
        cy = np.mean(h[:, 1])
        # Recover angle from first edge
        edge = h[1] - h[0]
        angle = np.arctan2(edge[1], edge[0])
        inner.append([cx, cy, np.degrees(angle) % 360])

    # Compute final outer side
    s_final = compute_min_side(hexes)

    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
