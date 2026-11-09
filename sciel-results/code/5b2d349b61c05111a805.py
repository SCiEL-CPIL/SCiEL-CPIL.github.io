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

    def _vertex_score_smooth(dists, beta=50.0):
        """
        Smooth surrogate for _vertex_score using soft-min for the k selection
        and soft-min for the final min(1, 1/raw).
        """
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        # Soft-min over diffs to get a smooth k-equivalent
        # We use log-sum-exp trick for min: min(x) = -log(sum(exp(-x/beta)))*beta
        neg_diffs = -diffs / beta
        neg_diffs -= neg_diffs.max()
        log_sum = np.log(np.sum(np.exp(neg_diffs)))
        soft_k_val = -beta * log_sum  # This is approximately min(diffs)
        
        # We need the actual index k for the discrete logic (which idx to pick).
        # For the smooth objective, we can't easily switch idx sets smoothly.
        # However, the "k" that minimizes the gap is usually stable in local regions.
        # To make it fully smooth, we would need to blend the 4 possible index sets?
        # No, k determines WHICH 4 distances to use.
        # If we just use the argmin index, it's still non-differentiable at the boundaries.
        # But Nelder-Mead handles non-smoothness reasonably well.
        # The main issue is the final min(1, 1/raw).
        
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
        
        # Smooth min(1, 1/raw)
        # min(a, b) ~ -beta * log(1/beta * (exp(-a/beta) + exp(-b/beta)))
        a = 1.0
        b = 1.0 / raw
        val = a - beta * np.log(np.exp(-a/beta) + np.exp(-b/beta))
        # Wait, the formula for smooth min is:
        # soft_min(x, y, t) = -t * log( (exp(-x/t) + exp(-y/t)) / 2 )
        # Let's use t=beta
        soft_min_val = -beta * np.log((np.exp(-a/beta) + np.exp(-b/beta)) / 2.0)
        return max(0.0, soft_min_val)

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
        # Sort offsets to ensure they are in order within the sector
        offs = np.sort(np.asarray(offsets, dtype=float))
        # Ensure strictly increasing with minimum gap
        if len(offs) < 4:
            return None
        if np.any(np.diff(offs) < 1e-10):
            return None
        # Check wrap-around gap within sector
        wrap_gap = (offs[0] + 2 * np.pi / 5) - offs[3]
        if wrap_gap < 1e-10:
            return None

        # Build all 20 angles with 5-fold symmetry
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

    # Regular 20-gon as baseline: 5-fold symmetry with evenly spaced within each sector
    # 4 vertices per 72-degree sector, evenly spaced: 0, 18, 36, 54 degrees = 0, pi/10, 2pi/10, 3pi/10
    base_offsets = np.array([0.0, np.pi/10, 2*np.pi/10, 3*np.pi/10])
    best_pts = _build_polygon(base_offsets)
    if best_pts is None:
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)

    best_score = _polygon_score(best_pts)
    best_offsets = base_offsets.copy()

    # Differential Evolution
    bounds = [(0.0, 2.0 * np.pi / 5) for _ in range(4)]
    try:
        res = differential_evolution(
            _obj,
            bounds,
            seed=20240611,
            maxiter=3000,
            popsize=40,
            tol=1e-12,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            init="sobol",
            callback=lambda xk, convergence: (
                True if (time.time() - start) < deadline - 100.0 else False
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

    # Nelder-Mead refinement with smooth objective
    if (time.time() - start) < deadline - 60.0:
        try:
            x0 = best_offsets.copy()
            r2 = minimize(
                lambda x: _obj(x, smooth=True),
                x0,
                method="Nelder-Mead",
                options={
                    "maxiter": 30000,
                    "xatol": 1e-13,
                    "fatol": 1e-15,
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
        
        # Additional smooth refinement from a perturbed start
        if (time.time() - start) < deadline - 45.0:
            try:
                rng = np.random.default_rng(123)
                noise = rng.uniform(-0.01, 0.01, size=4)
                x0 = best_offsets + noise
                x0 = np.clip(x0, 0.0, 2.0 * np.pi / 5)
                x0 = np.sort(x0)
                if np.all(np.diff(x0) > 1e-8):
                    r2b = minimize(
                        lambda x: _obj(x, smooth=True),
                        x0,
                        method="Nelder-Mead",
                        options={
                            "maxiter": 20000,
                            "xatol": 1e-13,
                            "fatol": 1e-15,
                            "adaptive": True
                        }
                    )
                    if r2b is not None and r2b.x is not None:
                        pts = _build_polygon(r2b.x)
                        if pts is not None:
                            sc = _polygon_score(pts)
                            if sc > best_score:
                                best_pts, best_score = pts, sc
                                best_offsets = np.sort(r2b.x).copy()
            except Exception:
                pass
        
        # Multi-start Nelder-Mead with exact objective to escape local minima
        if (time.time() - start) < deadline - 30.0:
            rng = np.random.default_rng(42)
            n_starts = 8
            for si in range(n_starts):
                if (time.time() - start) >= deadline - 15.0:
                    break
                if si == 0:
                    x0 = best_offsets.copy()
                else:
                    # Perturb best solution with random noise
                    noise_scale = 0.02 * (1.0 + si * 0.5)
                    x0 = best_offsets + rng.uniform(-noise_scale, noise_scale, size=4)
                    x0 = np.clip(x0, 0.0, 2.0 * np.pi / 5)
                    x0 = np.sort(x0)
                    if np.any(np.diff(x0) < 1e-8):
                        continue
                
                try:
                    r3 = minimize(
                        lambda x: _obj(x, smooth=False),
                        x0,
                        method="Nelder-Mead",
                        options={
                            "maxiter": 15000,
                            "xatol": 1e-13,
                            "fatol": 1e-15,
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
        
        # Final ultra-precise polish on the best solution found
        if (time.time() - start) < deadline - 5.0:
            try:
                x0 = best_offsets.copy()
                r4 = minimize(
                    lambda x: _obj(x, smooth=False),
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 30000,
                        "xatol": 1e-15,
                        "fatol": 1e-17,
                        "adaptive": True
                    }
                )
                if r4 is not None and r4.x is not None:
                    pts = _build_polygon(r4.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts, best_score = pts, sc
            except Exception:
                pass

    # Final safety check
    if _polygon_score(best_pts) <= 0.0:
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)

    return best_pts

# EVOLVE-BLOCK-END
