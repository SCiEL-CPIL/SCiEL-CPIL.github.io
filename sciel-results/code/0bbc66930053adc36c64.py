import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s kill

    def get_triples_indices():
        """Return index arrays for all triples (vectorized)."""
        i, j, k = np.meshgrid(
            np.arange(N), np.arange(N), np.arange(N), indexing='ij'
        )
        mask = (i < j) & (j < k)
        return i[mask], j[mask], k[mask]

    triple_i, triple_j, triple_k = get_triples_indices()
    n_triples = len(triple_i)

    def compute_score_fast(points):
        """
        Compute min triangle area / convex hull area.
        points: (N, 2) array
        Returns (score, area_hull)
        """
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
        """
        Build a candidate configuration.
        n_hull: number of points on outer ring
        n_inner: number of interior points (0 allowed)
        Returns (N, 2) array or None if invalid
        """
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
        else:
            pts = pts

        if pts.shape[0] != N:
            return None

        if jitter > 0:
            rng = np.random.default_rng(seed)
            pts = pts + rng.normal(0, jitter, pts.shape)

        return pts

    def physics_repulsion_step(pts, n_steps=200, dt=0.005, repulsion_strength=1.0,
                               containment_radius=1.5, center_pull=0.01):
        """
        Physics-based repulsion: points repel each other with inverse-square force,
        contained within a circular boundary. This pushes points apart to maximize
        minimum distances, which correlates with maximizing minimum triangle area.
        """
        pts = pts.copy().astype(float)
        n = len(pts)
        
        for step in range(n_steps):
            if time.time() - start_time > time_limit - 15:
                break
                
            # Compute pairwise forces
            forces = np.zeros_like(pts)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-10:
                        # Random nudge if coincident
                        forces[i] += 0.01 * np.random.randn(2)
                        continue
                    dist = np.sqrt(dist_sq)
                    # Inverse-square repulsion
                    force_mag = repulsion_strength / (dist_sq + 0.01)
                    forces[i] += force_mag * diff / dist
            
            # Containment: pull back toward center if outside radius
            for i in range(n):
                r = np.linalg.norm(pts[i])
                if r > containment_radius:
                    forces[i] -= center_pull * (r - containment_radius) * pts[i] / r
                else:
                    # Gentle centering force
                    forces[i] -= center_pull * pts[i]
            
            # Update positions
            pts += dt * forces
            
            # Hard constraint: keep within containment radius
            for i in range(n):
                r = np.linalg.norm(pts[i])
                if r > containment_radius:
                    pts[i] *= containment_radius / r
        
        return pts

    def physics_optimize(pts, total_steps=500, adaptive=True):
        """
        Run physics-based repulsion with adaptive parameters.
        Start with strong repulsion, gradually reduce to find equilibrium.
        """
        pts = pts.copy().astype(float)
        n = len(pts)
        
        # Phase 1: Strong repulsion to spread points
        phase1_steps = total_steps // 3
        pts = physics_repulsion_step(pts, n_steps=phase1_steps, dt=0.008,
                                      repulsion_strength=2.0, containment_radius=2.0,
                                      center_pull=0.005)
        
        if time.time() - start_time > time_limit - 10:
            return pts
        
        # Phase 2: Moderate repulsion to settle
        phase2_steps = total_steps // 3
        pts = physics_repulsion_step(pts, n_steps=phase2_steps, dt=0.003,
                                      repulsion_strength=1.0, containment_radius=1.5,
                                      center_pull=0.01)
        
        if time.time() - start_time > time_limit - 5:
            return pts
        
        # Phase 3: Fine-tuning
        phase3_steps = total_steps - phase1_steps - phase2_steps
        pts = physics_repulsion_step(pts, n_steps=phase3_steps, dt=0.001,
                                      repulsion_strength=0.5, containment_radius=1.2,
                                      center_pull=0.02)
        
        return pts

    # ------------------------------------------------------------------
    # Phase 1: Generate structured candidates and evaluate
    # ------------------------------------------------------------------
    best_points = None
    best_score = -float('inf')
    candidates = []

    # Focused sweep: emphasize 7-6 and 12-1 splits (known good structures)
    hull_radii = [0.8, 0.9, 1.0, 1.1, 1.2]
    inner_radii = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    offsets = [0.0, np.pi / 6, np.pi / 4, np.pi / 3]

    for n_hull in [7, 12, 8, 11, 6, 9, 10, 13, 5, 4, 3]:
        n_inner = N - n_hull
        if n_inner < 0:
            continue

        for hr in hull_radii:
            if time.time() - start_time > time_limit - 40:
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
        for trial in range(3):
            hr = rng.uniform(0.9, 1.2)
            ir = rng.uniform(0.15, 0.45) if n_inner > 0 else 0.0
            ho = rng.uniform(0, 2 * np.pi)
            io = rng.uniform(0, 2 * np.pi)
            jitter = rng.uniform(0.03, 0.08)
            cand = make_candidate(n_hull, n_inner, hr, ir,
                                  hull_offset=ho, inner_offset=io,
                                  jitter=jitter, seed=trial * 100 + n_hull)
            if cand is not None:
                candidates.append(cand)

    # Evaluate candidates
    print(f"Evaluating {len(candidates)} candidates...", flush=True)
    for idx, cand in enumerate(candidates):
        if time.time() - start_time > time_limit - 35:
            break
        s, _ = compute_score_fast(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
        if idx % 200 == 0:
            print(f"  Candidate {idx}/{len(candidates)}: best={best_score:.6f}", flush=True)

    print(f"Phase 1 done. Best score: {best_score:.6f}", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: Physics-based repulsion from multiple starting points
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 20:
        # Generate diverse starting points for physics optimization
        start_points_list = []
        
        # Use the best candidate
        start_points_list.append(best_points.copy())
        
        # Add a few other good candidates
        top_candidates = []
        for cand in candidates[:50]:
            s, _ = compute_score_fast(cand)
            top_candidates.append((s, cand))
        top_candidates.sort(key=lambda x: -x[0])
        for _, cand in top_candidates[1:4]:
            start_points_list.append(cand.copy())
        
        # Add a pure regular polygon
        start_points_list.append(_regular_polygon(N, radius=1.0))
        
        # Run physics optimization from each start point
        for sp_idx, sp in enumerate(start_points_list):
            if time.time() - start_time > time_limit - 15:
                break
            
            phys_pts = physics_optimize(sp, total_steps=400)
            phys_score, _ = compute_score_fast(phys_pts)
            
            if phys_score > best_score:
                best_score = phys_score
                best_points = phys_pts.copy()
                print(f"  Physics from start {sp_idx}: {best_score:.6f}", flush=True)

    # ------------------------------------------------------------------
    # Phase 3: Local refinement with Nelder-Mead
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
                options={'maxiter': 800, 'adaptive': True,
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
    # Phase 4: Final polish
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 3:
        try:
            res = opt.minimize(
                hard_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'adaptive': True,
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

# EVOLVE-BLOCK-END
