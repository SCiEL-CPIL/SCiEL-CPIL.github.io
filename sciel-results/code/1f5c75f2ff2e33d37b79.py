# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start = time.time()
    deadline = 570.0

    def _normalize(pts):
        c = pts.mean(axis=0)
        d = pts - c
        mean_r = np.mean(np.linalg.norm(d, axis=1))
        if mean_r <= 0.01:
            return None
        return d / mean_r

    def _vertex_score(dists):
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        D = 0.5 * (dists[k] + dists[k + 1])
        if D < 0.1:
            return 0.0
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k - 1, k, k + 1, k + 2]
        sel = dists[idx]
        raw = np.mean(np.maximum(sel / D, D / sel))
        if raw <= 0.0:
            return 0.0
        return min(1.0, 1.0 / raw)

    def _polygon_score(pts):
        p = _normalize(pts)
        if p is None:
            return 0.0
        m = p.shape[0]
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        best = 1.0
        for i in range(m):
            other = np.delete(dist[i], i)
            other.sort()
            s = _vertex_score(other)
            if s < best:
                best = s
                if best <= 0.0:
                    break
        return best

    def _is_convex(pts):
        """Check if the polygon defined by pts is strictly convex."""
        n_pts = len(pts)
        if n_pts < 3:
            return False
        prev_sign = 0
        for i in range(n_pts):
            p0 = pts[i]
            p1 = pts[(i + 1) % n_pts]
            p2 = pts[(i + 2) % n_pts]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-10:
                return False
            sign = 1 if cross > 0 else -1
            if prev_sign == 0:
                prev_sign = sign
            elif sign != prev_sign:
                return False
        return True

    def _build_polygon_4fold(params):
        """
        Build 20-gon with 4-fold rotational symmetry.
        params: 5 angles in [0, pi/2) and 5 radii in [0.3, 1.5].
        """
        angles = np.sort(params[0:5])
        radii = params[5:10]
        
        if np.any(angles < 0) or np.any(angles >= np.pi/2):
            return None
        if np.any(np.diff(angles) < 1e-8):
            return None
        wrap_gap = (angles[0] + np.pi/2) - angles[4]
        if wrap_gap < 1e-8:
            return None
        if np.any(radii <= 0.1):
            return None

        # Construct full angles and radii
        q1_angles = angles
        q2_angles = angles + np.pi/2
        q3_angles = angles + np.pi
        q4_angles = angles + 3*np.pi/2
        
        all_angles = np.concatenate([q1_angles, q2_angles, q3_angles, q4_angles])
        all_radii = np.tile(radii, 4)
        
        sort_idx = np.argsort(all_angles)
        final_angles = all_angles[sort_idx]
        final_radii = all_radii[sort_idx]
        
        x = final_radii * np.cos(final_angles)
        y = final_radii * np.sin(final_angles)
        pts = np.stack([x, y], axis=1)
        
        if not _is_convex(pts):
            return None
            
        return pts

    def _obj_4fold(params):
        pts = _build_polygon_4fold(params)
        if pts is None:
            return 1.0
        return -_polygon_score(pts)

    def _build_polygon_5fold(params):
        """
        Build 20-gon with 5-fold rotational symmetry.
        params: 4 angles in [0, 2*pi/5) and 4 radii in [0.3, 1.5].
        """
        angles = np.sort(params[0:4])
        radii = params[4:8]
        
        if np.any(angles < 0) or np.any(angles >= 2*np.pi/5):
            return None
        if np.any(np.diff(angles) < 1e-8):
            return None
        wrap_gap = (angles[0] + 2*np.pi/5) - angles[3]
        if wrap_gap < 1e-8:
            return None
        if np.any(radii <= 0.1):
            return None

        q1_angles = angles
        q2_angles = angles + 2*np.pi/5
        q3_angles = angles + 4*np.pi/5
        q4_angles = angles + 6*np.pi/5
        q5_angles = angles + 8*np.pi/5
        
        all_angles = np.concatenate([q1_angles, q2_angles, q3_angles, q4_angles, q5_angles])
        all_radii = np.tile(radii, 5)
        
        sort_idx = np.argsort(all_angles)
        final_angles = all_angles[sort_idx]
        final_radii = all_radii[sort_idx]
        
        x = final_radii * np.cos(final_angles)
        y = final_radii * np.sin(final_angles)
        pts = np.stack([x, y], axis=1)
        
        if not _is_convex(pts):
            return None
            
        return pts

    def _obj_5fold(params):
        pts = _build_polygon_5fold(params)
        if pts is None:
            return 1.0
        return -_polygon_score(pts)

    # Initialize best
    best_pts = None
    best_score = 0.0

    # Baseline: Regular 20-gon
    base_angles_full = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base_pts = np.stack([np.cos(base_angles_full), np.sin(base_angles_full)], axis=1)
    base_score = _polygon_score(base_pts)
    if base_score > best_score:
        best_pts = base_pts
        best_score = base_score

    # Search 4-fold symmetry
    # Baseline for 4-fold: Regular pentagon vertices in quadrant? No, 5 points in quadrant.
    # For regular 20-gon, angles are multiples of pi/10.
    # In [0, pi/2): 0, pi/10, 2pi/10, 3pi/10, 4pi/10.
    # Radii: all 1.
    init_4fold = np.concatenate([np.array([0, np.pi/10, 2*np.pi/10, 3*np.pi/10, 4*np.pi/10]), np.ones(5)])
    
    if (time.time() - start) < deadline - 100:
        try:
            bounds_4fold = [(0.0, np.pi/2) for _ in range(5)] + [(0.3, 1.5) for _ in range(5)]
            res_4fold = differential_evolution(
                _obj_4fold,
                bounds_4fold,
                seed=42,
                maxiter=1500,
                popsize=30,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init="sobol",
                callback=lambda xk, convergence: (time.time() - start) < deadline - 120.0
            )
            if res_4fold is not None and res_4fold.x is not None:
                pts = _build_polygon_4fold(res_4fold.x)
                if pts is not None:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts = pts
                        best_score = sc
                        best_params_4fold = res_4fold.x.copy()
        except Exception:
            pass

    # Search 5-fold symmetry
    # Baseline for 5-fold: Regular 20-gon.
    # Angles in [0, 2pi/5): 0, pi/10, 2pi/10, 3pi/10.
    # Radii: all 1.
    init_5fold = np.concatenate([np.array([0, np.pi/10, 2*np.pi/10, 3*np.pi/10]), np.ones(4)])
    
    if (time.time() - start) < deadline - 100:
        try:
            bounds_5fold = [(0.0, 2*np.pi/5) for _ in range(4)] + [(0.3, 1.5) for _ in range(4)]
            res_5fold = differential_evolution(
                _obj_5fold,
                bounds_5fold,
                seed=43,
                maxiter=1500,
                popsize=30,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init="sobol",
                callback=lambda xk, convergence: (time.time() - start) < deadline - 120.0
            )
            if res_5fold is not None and res_5fold.x is not None:
                pts = _build_polygon_5fold(res_5fold.x)
                if pts is not None:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts = pts
                        best_score = sc
                        best_params_5fold = res_5fold.x.copy()
        except Exception:
            pass

    # Local refinement on the best found so far
    # We need to know which symmetry was best to refine.
    # Let's just try refining the best point by trying to map it back? No, that's hard.
    # Instead, we can run a generic local optimization on the 20 points directly?
    # No, that's 40 dimensions and convexity constraints are hard.
    # Better: If we have best_params_4fold, refine that. If best_params_5fold, refine that.
    
    # Track which one was best
    best_sym = None
    best_params = None
    
    # Re-evaluate baselines to see if they beat the DE results?
    # DE usually finds something better than regular polygon.
    
    # Let's store the best params from the DE runs
    if 'best_params_4fold' in locals():
         if _polygon_score(_build_polygon_4fold(best_params_4fold)) > best_score:
             best_score = _polygon_score(_build_polygon_4fold(best_params_4fold))
             best_pts = _build_polygon_4fold(best_params_4fold)
             best_sym = 4
             best_params = best_params_4fold
             
    if 'best_params_5fold' in locals():
         if _polygon_score(_build_polygon_5fold(best_params_5fold)) > best_score:
             best_score = _polygon_score(_build_polygon_5fold(best_params_5fold))
             best_pts = _build_polygon_5fold(best_params_5fold)
             best_sym = 5
             best_params = best_params_5fold

    # If no symmetry found better than regular, just use regular
    if best_sym is None:
        # Try to refine the regular polygon? It's symmetric to both.
        # Let's just use the regular polygon as the best.
        best_pts = base_pts
        best_score = base_score

    # Refinement
    if (time.time() - start) < deadline - 30:
        if best_sym == 4:
            try:
                r2 = minimize(
                    _obj_4fold,
                    best_params,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 5000,
                        "xatol": 1e-12,
                        "fatol": 1e-14,
                        "adaptive": True
                    }
                )
                if r2 is not None and r2.x is not None:
                    pts = _build_polygon_4fold(r2.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts = pts
                            best_score = sc
            except Exception:
                pass
        elif best_sym == 5:
            try:
                r2 = minimize(
                    _obj_5fold,
                    best_params,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 5000,
                        "xatol": 1e-12,
                        "fatol": 1e-14,
                        "adaptive": True
                    }
                )
                if r2 is not None and r2.x is not None:
                    pts = _build_polygon_5fold(r2.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts = pts
                            best_score = sc
            except Exception:
                pass
        else:
            # If we are at the regular polygon, maybe we should try a small perturbation?
            # Or just return it. The regular polygon is a valid local minimum?
            # Actually, for the "equidistant" metric, the regular polygon might not be optimal.
            # But without specific params to refine, we can't easily do local search on 40D.
            # We could try to initialize 4-fold params from the regular polygon and refine?
            # But we already did that in DE.
            pass

    # Final safety check
    if best_pts is None or _polygon_score(best_pts) <= 0.0:
        base_angles_full = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([np.cos(base_angles_full), np.sin(base_angles_full)], axis=1)

    return best_pts

# EVOLVE-BLOCK-END
