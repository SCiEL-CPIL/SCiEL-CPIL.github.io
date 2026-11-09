# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start = time.time()
    deadline = 560.0
    radius = 1.0

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

    def _vertex_score_smooth(dists, beta=10.0):
        """
        Smooth surrogate for _vertex_score.
        Blends the scores from the top 3 candidates for k to smooth out the argmin.
        """
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        
        # Find top 3 smallest gaps
        # We want to blend the scores based on the gap size.
        # Sort diffs to get indices of smallest gaps
        sorted_indices = np.argsort(diffs)
        top_k_indices = sorted_indices[:3]
        
        # Weights based on soft-min of diffs
        # Lower diff -> higher weight
        # Use softmax with negative diffs
        neg_diffs_top = -diffs[top_k_indices] / beta
        neg_diffs_top -= neg_diffs_top.max()
        weights = np.exp(neg_diffs_top)
        weights /= np.sum(weights)
        
        total_score = 0.0
        for w, k in zip(weights, top_k_indices):
            k = int(k)
            D = 0.5 * (dists[k] + dists[k + 1])
            if D < 0.1:
                # Penalty for invalid D
                total_score += w * 0.0
                continue
            if k == 0:
                idx = [0, 1, 2, 3]
            elif k == 17:
                idx = [15, 16, 17, 18]
            else:
                idx = [k - 1, k, k + 1, k + 2]
            sel = dists[idx]
            raw = np.mean(np.maximum(sel / D, D / sel))
            if raw <= 0.0:
                total_score += w * 0.0
                continue
            
            # Smooth min(1, 1/raw)
            a = 1.0
            b = 1.0 / raw
            t = beta
            m = min(a, b)
            soft_min_val = m - t * np.log(np.exp(-(a - m) / t) + np.exp(-(b - m) / t))
            total_score += w * max(0.0, soft_min_val)
            
        return total_score

    def _polygon_score(pts, smooth=False):
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
            if smooth:
                s = _vertex_score_smooth(other)
            else:
                s = _vertex_score(other)
            if s < best:
                best = s
                if best <= 0.0:
                    break
        return best

    def _build_polygon(offsets):
        """
        Build 20-gon with 5-fold rotational symmetry.
        offsets: 4 values in [0, 2*pi/5) representing angular positions within one sector.
        """
        offs = np.sort(np.asarray(offsets, dtype=float))
        if len(offs) < 4:
            return None
        if np.any(np.diff(offs) < 1e-10):
            return None
        wrap_gap = (offs[0] + 2 * np.pi / 5) - offs[3]
        if wrap_gap < 1e-10:
            return None

        angles = np.array([])
        for j in range(5):
            sector_start = j * 2 * np.pi / 5
            angles = np.append(angles, offs + sector_start)
        angles = np.sort(angles)

        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        pts = np.stack([x, y], axis=1)
        return pts

    def _obj(offsets, smooth=False):
        pts = _build_polygon(offsets)
        if pts is None:
            return 1.0
        return -_polygon_score(pts, smooth=smooth)

    # Initial candidates
    candidate_offsets = []
    base = np.array([0.0, np.pi/10, 2*np.pi/10, 3*np.pi/10])
    candidate_offsets.append(base)
    
    # Perturbations
    for pert in [0.01, -0.01, 0.02, -0.02, 0.05, -0.05]:
        candidate_offsets.append(base + np.array([0, pert, -pert, 0]))
        candidate_offsets.append(base + np.array([pert, 0, 0, -pert]))
        candidate_offsets.append(base + np.array([0, 0, pert, -pert]))
    
    best_score = 0.0
    best_offsets = None
    for cand in candidate_offsets:
        cand = np.sort(cand)
        if cand[0] < 0 or cand[-1] >= 2 * np.pi / 5:
            continue
        if np.any(np.diff(cand) < 1e-10):
            continue
        wrap_gap = (cand[0] + 2 * np.pi / 5) - cand[-1]
        if wrap_gap < 1e-10:
            continue
        pts = _build_polygon(cand)
        if pts is None:
            continue
        sc = _polygon_score(pts)
        if sc > best_score:
            best_score = sc
            best_offsets = cand.copy()
            
    if best_offsets is None:
        best_offsets = base.copy()
        best_score = _polygon_score(_build_polygon(best_offsets))
        
    best_pts = _build_polygon(best_offsets)

    # Differential Evolution
    bounds = [(0.0, 2.0 * np.pi / 5) for _ in range(4)]
    try:
        res = differential_evolution(
            _obj,
            bounds,
            seed=20240611,
            maxiter=1500,
            popsize=30,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            init="sobol",
            callback=lambda xk, convergence: (
                True if (time.time() - start) < deadline - 150.0 else False
            ),
        )
        if res is not None and res.x is not None:
            pts = _build_polygon(res.x)
            if pts is not None:
                sc = _polygon_score(pts)
                if sc > best_score:
                    best_pts, best_score = pts, sc
                    best_offsets = np.sort(res.x).copy()
    except Exception:
        pass

    # Nelder-Mead Refinement
    # Use smooth objective for the first pass to navigate non-smoothness
    if (time.time() - start) < deadline - 60.0:
        try:
            x0 = best_offsets.copy()
            r2 = minimize(
                lambda x: _obj(x, smooth=True),
                x0,
                method="Nelder-Mead",
                options={
                    "maxiter": 20000,
                    "xatol": 1e-12,
                    "fatol": 1e-14,
                    "adaptive": True
                }
            )
            if r2 is not None and r2.x is not None:
                pts = _build_polygon(r2.x)
                if pts is not None:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts, best_score = pts, sc
                        best_offsets = np.sort(r2.x).copy()
        except Exception:
            pass
        
        # Second pass with exact objective
        if (time.time() - start) < deadline - 30.0:
            try:
                x0 = best_offsets.copy()
                r3 = minimize(
                    lambda x: _obj(x, smooth=False),
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 30000,
                        "xatol": 1e-14,
                        "fatol": 1e-16,
                        "adaptive": True
                    }
                )
                if r3 is not None and r3.x is not None:
                    pts = _build_polygon(r3.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts, best_score = pts, sc
                            best_offsets = np.sort(r3.x).copy()
            except Exception:
                pass

        # Multi-start from random perturbations
        if (time.time() - start) < deadline - 15.0:
            rng = np.random.RandomState(42)
            for trial in range(10):
                if (time.time() - start) >= deadline - 10.0:
                    break
                try:
                    x0 = best_offsets + rng.normal(0, 0.02, size=4)
                    x0 = np.sort(x0)
                    if x0[0] < 0 or x0[-1] >= 2 * np.pi / 5:
                        continue
                    if np.any(np.diff(x0) < 1e-10):
                        continue
                    wrap_gap = (x0[0] + 2 * np.pi / 5) - x0[-1]
                    if wrap_gap < 1e-10:
                        continue
                    
                    # Use smooth objective for these random starts
                    r4 = minimize(
                        lambda x: _obj(x, smooth=True),
                        x0,
                        method="Nelder-Mead",
                        options={
                            "maxiter": 10000,
                            "xatol": 1e-12,
                            "fatol": 1e-14,
                            "adaptive": True
                        }
                    )
                    if r4 is not None and r4.x is not None:
                        pts = _build_polygon(r4.x)
                        if pts is not None:
                            sc = _polygon_score(pts)
                            if sc > best_score:
                                best_pts, best_score = pts, sc
                                best_offsets = np.sort(r4.x).copy()
                except Exception:
                    pass

    # Final safety check
    if _polygon_score(best_pts) <= 0.0:
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)

    return best_pts

# EVOLVE-BLOCK-END
