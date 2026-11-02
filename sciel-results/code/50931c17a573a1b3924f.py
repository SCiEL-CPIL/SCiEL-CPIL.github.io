import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s kill

    def get_triples_indices():
        i, j, k = np.meshgrid(
            np.arange(N), np.arange(N), np.arange(N), indexing='ij'
        )
        mask = (i < j) & (j < k)
        return i[mask], j[mask], k[mask]

    triple_i, triple_j, triple_k = get_triples_indices()
    n_triples = len(triple_i)

    def compute_score_fast(points):
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-12:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]

        xi = x[triple_i]
        xj = x[triple_j]
        xk = x[triple_k]
        yi = y[triple_i]
        yj = y[triple_j]
        yk = y[triple_k]

        cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        areas = 0.5 * cross
        min_area = np.min(areas)

        score = min_area / area_hull
        return score, area_hull

    def soft_min_score(points, temperature=0.05):
        """Smooth surrogate: -log-sum-exp of areas, normalized by hull area."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return -1e10

        if area_hull < 1e-12:
            return -1e10

        x = points[:, 0]
        y = points[:, 1]

        xi = x[triple_i]
        xj = x[triple_j]
        xk = x[triple_k]
        yi = y[triple_i]
        yj = y[triple_j]
        yk = y[triple_k]

        cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        areas = 0.5 * cross

        # Soft minimum: -T * log(sum(exp(-areas/T)))
        scaled = -areas / temperature
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -temperature * log_sum

        score = soft_min / area_hull
        return score

    def neg_soft_score(x, temperature=0.05):
        pts = x.reshape(-1, 2)
        return -soft_min_score(pts, temperature)

    def _regular_polygon(n, radius, center=(0.0, 0.0), angle_offset=0.0):
        if n <= 1:
            if n == 1:
                return np.array([[center[0], center[1]]])
            return np.array([]).reshape(0, 2)
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def make_candidate(n_hull, n_inner, hull_radius, inner_radius,
                       hull_offset=0.0, inner_offset=0.0,
                       inner_center=(0.0, 0.0), jitter=0.0, seed=None):
        if n_hull + n_inner != N:
            return None
        if n_hull < 3:
            return None
        if hull_radius <= 0:
            return None

        pts = _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)
        if n_inner > 0:
            inner = _regular_polygon(n_inner, inner_radius,
                                     center=inner_center, angle_offset=inner_offset)
            pts = np.vstack([pts, inner])

        if pts.shape[0] != N:
            return None

        if jitter > 0:
            rng = np.random.default_rng(seed)
            pts = pts + rng.normal(0, jitter, pts.shape)

        return pts

    def simulated_annealing(pts, n_iterations=5000, t_start=0.5, t_end=0.001,
                            step_scale=0.05, cooling_rate=None):
        """Simulated Annealing with log-sum-exp surrogate."""
        pts = pts.copy().astype(float)
        current_score = soft_min_score(pts, temperature=0.05)
        best_pts = pts.copy()
        best_score = current_score

        if cooling_rate is None:
            cooling_rate = (t_end / t_start) ** (1.0 / max(n_iterations, 1))

        for it in range(n_iterations):
            if time.time() - start_time > time_limit - 10:
                break

            T = t_start * (cooling_rate ** it)

            # Random perturbation
            rng = np.random.default_rng(hash((it, id(pts))) % (2**32))
            delta = rng.normal(0, step_scale * T, pts.shape)
            new_pts = pts + delta

            new_score = soft_min_score(new_pts, temperature=0.05)
            delta_score = new_score - current_score

            if delta_score > 0 or np.random.random() < np.exp(delta_score / max(T, 1e-10)):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()

        return best_pts, best_score

    def newton_equarea_solve(pts, n_steps=50, target_area=None):
        """
        Damped Newton-Raphson: force active triangle areas to equal target.
        Active triangles are those with area closest to the minimum.
        """
        pts = pts.copy().astype(float)

        for step in range(n_steps):
            if time.time() - start_time > time_limit - 10:
                break

            try:
                hull = ConvexHull(pts)
                area_hull = hull.volume
            except Exception:
                break

            if area_hull < 1e-12:
                break

            x = pts[:, 0]
            y = pts[:, 1]

            xi = x[triple_i]
            xj = x[triple_j]
            xk = x[triple_k]
            yi = y[triple_i]
            yj = y[triple_j]
            yk = y[triple_k]

            cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
            areas = 0.5 * cross

            min_area = np.min(areas)
            if target_area is None:
                target_area = min_area

            # Active triangles: those within 20% of minimum
            active_mask = areas < (min_area + 0.2 * target_area)
            if np.sum(active_mask) < 3:
                active_mask = areas < (min_area + 0.5 * target_area)
            if np.sum(active_mask) < 3:
                break

            active_areas = areas[active_mask]
            n_active = len(active_areas)

            # Build Jacobian for active triangles w.r.t. point coordinates
            # Each triangle area = 0.5 * |cross product|
            # We use the signed cross for differentiability
            # area_ij_k = 0.5 * ((xj-xi)*(yk-yi) - (yj-yi)*(xk-xi))

            # Residual: area - target
            # We want to solve J * dx = -(r) where r = area - target

            # For efficiency, use a subset of active triangles
            max_active = min(n_active, 26)
            if n_active > max_active:
                idx = np.argsort(active_areas)[:max_active]
                active_idx = np.where(active_mask)[0][idx]
            else:
                active_idx = np.where(active_mask)[0]

            n_eq = len(active_idx)
            n_var = 2 * N

            # Build Jacobian (n_eq x n_var)
            J = np.zeros((n_eq, n_var))
            r = np.zeros(n_eq)

            for eq_i, tri_idx in enumerate(active_idx):
                i = triple_i[tri_idx]
                j = triple_j[tri_idx]
                k = triple_k[tri_idx]

                xi_v = pts[i, 0]
                yi_v = pts[i, 1]
                xj_v = pts[j, 0]
                yj_v = pts[j, 1]
                xk_v = pts[k, 0]
                yk_v = pts[k, 1]

                # Signed area = 0.5 * ((xj-xi)*(yk-yi) - (yj-yi)*(xk-xi))
                area_val = 0.5 * ((xj_v - xi_v) * (yk_v - yi_v) - (yj_v - yi_v) * (xk_v - xi_v))
                r[eq_i] = area_val - target_area

                # d(area)/d(xi) = -0.5 * (yk - yi)
                # d(area)/d(yi) = 0.5 * (xk - xi)
                # d(area)/d(xj) = 0.5 * (yk - yi)
                # d(area)/d(yj) = -0.5 * (xk - xi)
                # d(area)/d(xk) = -0.5 * (yj - yi)
                # d(area)/d(yk) = 0.5 * (xj - xi)

                J[eq_i, 2*i] = -0.5 * (yk_v - yi_v)
                J[eq_i, 2*i+1] = 0.5 * (xk_v - xi_v)
                J[eq_i, 2*j] = 0.5 * (yk_v - yi_v)
                J[eq_i, 2*j+1] = -0.5 * (xk_v - xi_v)
                J[eq_i, 2*k] = -0.5 * (yj_v - yi_v)
                J[eq_i, 2*k+1] = 0.5 * (xj_v - xi_v)

            # Solve J * dx = -r using least squares
            try:
                dx, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)
            except Exception:
                break

            # Damped step
            damping = 0.1
            pts_new = pts + damping * dx.reshape(N, 2)

            # Check if score improved
            new_score, _ = compute_score_fast(pts_new)
            old_score, _ = compute_score_fast(pts)

            if new_score > old_score:
                pts = pts_new
            else:
                # Try smaller step
                pts_new = pts + 0.01 * dx.reshape(N, 2)
                new_score, _ = compute_score_fast(pts_new)
                if new_score > old_score:
                    pts = pts_new
                else:
                    break

        return pts

    # ------------------------------------------------------------------
    # Phase 1: Generate structured candidates
    # ------------------------------------------------------------------
    best_points = None
    best_score = -float('inf')
    candidates = []

    hull_radii = [0.8, 0.9, 1.0, 1.1, 1.2]
    inner_radii = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    offsets = [0.0, np.pi / 6, np.pi / 4, np.pi / 3]

    for n_hull in [7, 12, 8, 11, 6, 9, 10, 13, 5, 4, 3]:
        n_inner = N - n_hull
        if n_inner < 0:
            continue

        for hr in hull_radii:
            if time.time() - start_time > time_limit - 50:
                break
            for ir in inner_radii:
                if n_inner == 0:
                    for ho in offsets:
                        cand = make_candidate(n_hull, 0, hr, 0.0, hull_offset=ho)
                        if cand is not None:
                            candidates.append(cand)
                else:
                    for ho in offsets:
                        for io in offsets[:3]:
                            cand = make_candidate(n_hull, n_inner, hr, ir,
                                                  hull_offset=ho, inner_offset=io)
                            if cand is not None:
                                candidates.append(cand)

    # Add jittered variants
    rng = np.random.default_rng(42)
    for n_hull in [7, 12, 8, 11]:
        n_inner = N - n_hull
        if n_inner < 0:
            continue
        for trial in range(5):
            hr = rng.uniform(0.9, 1.2)
            ir = rng.uniform(0.15, 0.45) if n_inner > 0 else 0.0
            ho = rng.uniform(0, 2 * np.pi)
            io = rng.uniform(0, 2 * np.pi)
            jitter = rng.uniform(0.03, 0.1)
            cand = make_candidate(n_hull, n_inner, hr, ir,
                                  hull_offset=ho, inner_offset=io,
                                  jitter=jitter, seed=trial * 100 + n_hull)
            if cand is not None:
                candidates.append(cand)

    # Random initial points
    for trial in range(20):
        rng_r = np.random.default_rng(1000 + trial)
        pts = rng_r.uniform(-1, 1, (N, 2))
        candidates.append(pts)

    # Evaluate candidates
    print(f"Evaluating {len(candidates)} candidates...", flush=True)
    for idx, cand in enumerate(candidates):
        if time.time() - start_time > time_limit - 45:
            break
        s, _ = compute_score_fast(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
        if idx % 200 == 0:
            print(f"  Candidate {idx}/{len(candidates)}: best={best_score:.6f}", flush=True)

    print(f"Phase 1 done. Best score: {best_score:.6f}", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: Simulated Annealing with log-sum-exp surrogate
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 30:
        # Run SA from top candidates
        top_candidates = []
        for cand in candidates[:100]:
            s, _ = compute_score_fast(cand)
            top_candidates.append((s, cand))
        top_candidates.sort(key=lambda x: -x[0])

        sa_starts = [c for _, c in top_candidates[:5]]
        sa_starts.append(_regular_polygon(N, radius=1.0))

        for sp_idx, sp in enumerate(sa_starts):
            if time.time() - start_time > time_limit - 20:
                break

            sa_pts, sa_score = simulated_annealing(
                sp, n_iterations=3000, t_start=0.3, t_end=0.005, step_scale=0.08
            )
            hard_score, _ = compute_score_fast(sa_pts)

            if hard_score > best_score:
                best_score = hard_score
                best_points = sa_pts.copy()
                print(f"  SA from start {sp_idx}: {best_score:.6f}", flush=True)

    # ------------------------------------------------------------------
    # Phase 3: Newton equi-area refinement
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 15:
        newton_pts = newton_equarea_solve(best_points, n_steps=80)
        newton_score, _ = compute_score_fast(newton_pts)
        if newton_score > best_score:
            best_score = newton_score
            best_points = newton_pts.copy()
            print(f"After Newton: {best_score:.6f}", flush=True)

    # ------------------------------------------------------------------
    # Phase 4: Nelder-Mead local refinement
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 8:
        def hard_obj(x):
            pts = x.reshape(-1, 2)
            s, _ = compute_score_fast(pts)
            return -s

        try:
            res = opt.minimize(
                hard_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 600, 'adaptive': True,
                         'xatol': 1e-12, 'fatol': 1e-14}
            )
            nm_pts = res.x.reshape(-1, 2)
            nm_score, _ = compute_score_fast(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_pts.copy()
            print(f"After NM: {best_score:.6f}", flush=True)
        except Exception as e:
            print(f"NM failed: {e}", flush=True)

    # ------------------------------------------------------------------
    # Phase 5: Final Newton + NM polish
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 3:
        try:
            polish_pts = newton_equarea_solve(best_points, n_steps=30)
            polish_score, _ = compute_score_fast(polish_pts)
            if polish_score > best_score:
                best_score = polish_score
                best_points = polish_pts.copy()

            res = opt.minimize(
                hard_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'adaptive': True,
                         'xatol': 1e-13, 'fatol': 1e-15}
            )
            final_pts = res.x.reshape(-1, 2)
            final_score, _ = compute_score_fast(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
            print(f"Final polish: {best_score:.6f}", flush=True)
        except Exception as e:
            print(f"Final polish failed: {e}", flush=True)

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    print(f"FINAL SCORE: {best_score:.8f}", flush=True)
    return best_points
