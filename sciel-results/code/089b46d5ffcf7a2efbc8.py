import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_min(areas, beta):
        """Compute soft minimum of areas using log-sum-exp trick."""
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 0.0
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        return -log_sum_exp / beta

    def _soft_objective(points_flat, beta=100.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        soft_min_area = _soft_min(areas, beta)
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _generate_initializations(rng, count=10):
        """Generate a diverse set of initializations."""
        inits = []
        # 1. Regular 13-gon
        inits.append(_regular_polygon(13, 1.0))
        
        # 2. 12 outer + 1 center
        inits.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
        
        # 3. Two rings (7+6)
        for r_inner in [0.4, 0.5, 0.6]:
            outer = _regular_polygon(7, 1.0)
            inner = _regular_polygon(6, r_inner, offset=np.pi/6)
            inits.append(np.vstack([outer, inner]))
            
        # 4. Three rings (5+5+3)
        inits.append(np.vstack([
            _regular_polygon(5, 1.0),
            _regular_polygon(5, 0.6, offset=np.pi/5),
            _regular_polygon(3, 0.3)
        ]))
        
        # 5. Random structured
        for _ in range(count):
            n_outer = rng.choice([6, 7, 8, 9, 10, 11])
            n_inner = N - n_outer
            r_outer = rng.uniform(0.9, 1.0)
            r_inner = rng.uniform(0.3, 0.7)
            outer = _regular_polygon(n_outer, r_outer)
            inner = _regular_polygon(n_inner, r_inner, offset=rng.uniform(0, 2*np.pi/max(n_inner,1)))
            pts = np.vstack([outer, inner])
            pts += rng.randn(*pts.shape) * 0.1
            inits.append(pts)
            
        return inits

    def _optimize_single(points, time_budget=5.0):
        """Optimize a single point set with a given time budget."""
        current = _normalize_points(points.copy())
        start_opt = time.time()
        
        # Phase 1: Soft objective with L-BFGS-B (smooth global search)
        if time.time() - start_opt < time_budget * 0.4:
            betas = [20.0, 50.0, 100.0, 200.0]
            for beta in betas:
                if time.time() - start_opt > time_budget * 0.4:
                    break
                try:
                    res = opt.minimize(
                        lambda x, b=beta: _soft_objective(x, b),
                        current.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    current = res.x.reshape(-1, 2)
                except Exception:
                    pass
        
        # Phase 2: Hard objective with Powell (local refinement)
        if time.time() - start_opt < time_budget * 0.8:
            try:
                res = opt.minimize(
                    _hard_objective,
                    current.flatten(),
                    method='Powell',
                    options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-10}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
                
        # Phase 3: Final Polish with Nelder-Mead or Coordinate Descent
        if time.time() - start_opt < time_budget:
             try:
                res = opt.minimize(
                    _hard_objective,
                    current.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                current = res.x.reshape(-1, 2)
             except Exception:
                 pass
                 
        current = _normalize_points(current)
        return current

    # Main Loop
    best_score = -float('inf')
    best_points = None
    rng = np.random.RandomState(42)
    
    # Keep a pool of the best solutions found so far to use for perturbation
    elite_pool = []
    
    while time.time() < start_time + time_limit - 2.0:
        # Determine how many initializations to generate
        # If we have time, generate more. If time is tight, focus on refining elites.
        remaining_time = start_time + time_limit - 2.0 - time.time()
        
        if remaining_time > 20.0:
            n_new = 10
        elif remaining_time > 10.0:
            n_new = 5
        else:
            n_new = 1 # Focus on polishing
            
        # Generate initializations
        candidates = []
        
        # 1. New random/structured starts
        if n_new > 0:
            new_inits = _generate_initializations(rng, count=n_new)
            candidates.extend(new_inits)
            
        # 2. Perturbations of elites
        if elite_pool:
            # Pick a few elites to perturb
            n_perturb = min(3, len(elite_pool), n_new)
            for _ in range(n_perturb):
                elite_idx = rng.randint(0, len(elite_pool))
                base = elite_pool[elite_idx]
                perturbed = base + rng.randn(*base.shape) * 0.05
                candidates.append(perturbed)
        
        if not candidates:
            break
            
        # Optimize each candidate
        time_per_candidate = max(0.5, remaining_time / len(candidates))
        
        for cand in candidates:
            if time.time() > start_time + time_limit - 2.0:
                break
            
            # Adaptive time budget: give more time to better candidates? 
            # For now, uniform budget per candidate in this batch.
            optimized = _optimize_single(cand, time_budget=time_per_candidate)
            
            score = compute_score(optimized)
            
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
                
            # Add to elite pool if good
            if score > 0.5 * best_score and score > 1e-6: # Threshold to avoid junk
                # Limit elite pool size
                if len(elite_pool) < 20:
                    elite_pool.append(optimized)
                else:
                    # Replace worst in pool? Or just keep best 20.
                    # Simple: if better than worst in pool, replace
                    scores_in_pool = [compute_score(p) for p in elite_pool]
                    min_idx = np.argmin(scores_in_pool)
                    if score > scores_in_pool[min_idx]:
                        elite_pool[min_idx] = optimized

    if best_points is None:
        # Fallback
        best_points = _normalize_points(_regular_polygon(13, 1.0))

    # Final high-precision polish on the absolute best found
    if time.time() < start_time + time_limit - 1.0:
        try:
            res = opt.minimize(
                _hard_objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            final_points = res.x.reshape(-1, 2)
            final_score = compute_score(final_points)
            if final_score > best_score:
                best_points = final_points
                best_score = final_score
        except Exception:
            pass
            
    return best_points
