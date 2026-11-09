# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start = time.time()
    deadline = 620.0
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

    def _vertex_score_smooth(dists, beta=30.0):
        """
        Smooth surrogate: blend over nearby k values using softmax weights.
        """
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        
        # Soft weights over k values using softmax on negative diffs
        neg_diffs = -diffs / beta
        neg_diffs -= neg_diffs.max()
        exp_vals = np.exp(neg_diffs)
        weights = exp_vals / exp_vals.sum()
        
        total = 0.0
        for k in range(19):
            if weights[k] < 1e-15:
                continue
            D = 0.5 * (dists[k] + dists[k + 1])
            if D < 0.1:
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
                continue
            # Smooth min(1, 1/raw) using log-sum-exp
            a = 1.0
            b = 1.0 / raw
            soft_min_val = -beta * np.log((np.exp(-a / beta) + np.exp(-b / beta)) / 2.0)
            total += weights[k] * max(0.0, soft_min_val)
        
        return total

    def _polygon_score(pts, smooth=False, beta=30.0):
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
                s = _vertex_score_smooth(other, beta=beta)
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

    def _build_bipartite(R1, R2, phi):
        """
        Build 20-gon as alternating vertices on two concentric circles.
        R1, R2: radii of the two circles (normalized so mean dist to centroid = 1)
        phi: angular offset for the second circle relative to the first
        Vertices alternate: v0 on circle 1, v1 on circle 2, etc.
        """
        # Ensure positive radii and reasonable values
        R1 = max(0.3, min(2.0, R1))
        R2 = max(0.3, min(2.0, R2))
        phi = np.mod(phi, 2.0 * np.pi)
        
        # 10 vertices on each circle, alternating
        angles1 = np.linspace(0, 2*np.pi, 10, endpoint=False)
        angles2 = angles1 + phi + np.pi/10  # offset by half-step plus phi
        
        x1 = R1 * np.cos(angles1)
        y1 = R1 * np.sin(angles1)
        x2 = R2 * np.cos(angles2)
        y2 = R2 * np.sin(angles2)
        
        # Interleave: v0 from circle 1, v1 from circle 2, etc.
        # Need to sort all 20 by angle to get cyclic order
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        all_angles = np.concatenate([angles1, angles2])
        
        # Sort by angle
        order = np.argsort(all_angles)
        x = all_x[order]
        y = all_y[order]
        
        pts = np.stack([x, y], axis=1)
        return pts

    def _obj(offsets, smooth=False, beta=30.0):
        pts = _build_polygon(offsets)
        if pts is None:
            return 1.0
        return -_polygon_score(pts, smooth=smooth, beta=beta)

    # Regular 20-gon as baseline
    base_offsets = np.array([0.0, np.pi/10, 2*np.pi/10, 3*np.pi/10])
    best_pts = _build_polygon(base_offsets)
    if best_pts is None:
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)

    best_score = _polygon_score(best_pts)
    best_offsets = base_offsets.copy()
    
    # Also try bipartite radial parameterization as an alternative baseline
    bip_base = _build_bipartite(1.0, 1.0, 0.0)
    if bip_base is not None:
        bip_score = _polygon_score(bip_base)
        if bip_score > best_score:
            best_pts = bip_base
            best_score = bip_score

    # Phase 1: Differential Evolution with smooth objective
    bounds = [(0.0, 2.0 * np.pi / 5) for _ in range(4)]
    try:
        res = differential_evolution(
            lambda x: _obj(x, smooth=True, beta=50.0),
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
    
    # Phase 1b: Differential Evolution on bipartite radial parameters
    if (time.time() - start) < deadline - 100.0:
        def _obj_bipartite(params, smooth=False, beta=50.0):
            R1, R2, phi = params
            pts = _build_bipartite(R1, R2, phi)
            if pts is None:
                return 1.0
            return -_polygon_score(pts, smooth=smooth, beta=beta)
        
        bounds_bip = [(0.5, 1.5), (0.5, 1.5), (0.0, np.pi/5)]
        try:
            res_bip = differential_evolution(
                lambda x: _obj_bipartite(x, smooth=True, beta=50.0),
                bounds_bip,
                seed=20240612,
                maxiter=2000,
                popsize=30,
                tol=1e-12,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init="sobol",
                callback=lambda xk, convergence: (
                    True if (time.time() - start) < deadline - 100.0 else False
                ),
            )
            if res_bip is not None and res_bip.x is not None:
                pts = _build_bipartite(res_bip.x[0], res_bip.x[1], res_bip.x[2])
                if pts is not None:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts, best_score = pts, sc
        except Exception:
            pass

    # Phase 2: Adaptive smooth Nelder-Mead with decreasing beta
    if (time.time() - start) < deadline - 100.0:
        betas = [40.0, 25.0, 15.0, 8.0]
        for beta in betas:
            if (time.time() - start) >= deadline - 50.0:
                break
            try:
                x0 = best_offsets.copy()
                r2 = minimize(
                    lambda x: _obj(x, smooth=True, beta=beta),
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 25000,
                        "xatol": 1e-14,
                        "fatol": 1e-16,
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
    
    # Phase 2b: Nelder-Mead polish on bipartite parameters
    if (time.time() - start) < deadline - 50.0:
        # Start from a reasonable bipartite initial guess
        bip_starts = [
            (1.0, 1.0, 0.0),
            (0.9, 1.1, np.pi/20),
            (1.1, 0.9, np.pi/15),
            (0.95, 1.05, np.pi/25),
        ]
        
        for R1_0, R2_0, phi_0 in bip_starts:
            if (time.time() - start) >= deadline - 30.0:
                break
            try:
                r_bip = minimize(
                    lambda x: _obj_bipartite(x, smooth=True, beta=20.0),
                    np.array([R1_0, R2_0, phi_0]),
                    method="Nelder-Mead",
                    options={
                        "maxiter": 15000,
                        "xatol": 1e-14,
                        "fatol": 1e-16,
                        "adaptive": True
                    }
                )
                if r_bip is not None and r_bip.x is not None:
                    pts = _build_bipartite(r_bip.x[0], r_bip.x[1], r_bip.x[2])
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts, best_score = pts, sc
            except Exception:
                pass

    # Phase 3: Multi-start with exact objective for escaping local minima
    if (time.time() - start) < deadline - 40.0:
        rng = np.random.default_rng(42)
        n_starts = 15
        for si in range(n_starts):
            if (time.time() - start) >= deadline - 20.0:
                break
            if si == 0:
                x0 = best_offsets.copy()
            else:
                # Increasingly diverse perturbations
                noise_scale = 0.015 * (1.0 + si * 0.3)
                x0 = best_offsets + rng.uniform(-noise_scale, noise_scale, size=4)
                x0 = np.clip(x0, 0.0, 2.0 * np.pi / 5)
                x0 = np.sort(x0)
                if np.any(np.diff(x0) < 1e-8):
                    continue

            try:
                # Use smooth objective first
                r3 = minimize(
                    lambda x: _obj(x, smooth=True, beta=20.0),
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 12000,
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
    
    # Phase 3b: Multi-start on bipartite parameters
    if (time.time() - start) < deadline - 20.0:
        rng_bip = np.random.default_rng(123)
        n_bip_starts = 8
        for si in range(n_bip_starts):
            if (time.time() - start) >= deadline - 10.0:
                break
            R1_0 = rng_bip.uniform(0.7, 1.3)
            R2_0 = rng_bip.uniform(0.7, 1.3)
            phi_0 = rng_bip.uniform(0.0, np.pi/5)
            
            try:
                r_bip3 = minimize(
                    lambda x: _obj_bipartite(x, smooth=True, beta=15.0),
                    np.array([R1_0, R2_0, phi_0]),
                    method="Nelder-Mead",
                    options={
                        "maxiter": 8000,
                        "xatol": 1e-14,
                        "fatol": 1e-16,
                        "adaptive": True
                    }
                )
                if r_bip3 is not None and r_bip3.x is not None:
                    pts = _build_bipartite(r_bip3.x[0], r_bip3.x[1], r_bip3.x[2])
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts, best_score = pts, sc
            except Exception:
                pass

    # Phase 4: Final ultra-precise polish with exact objective
    if (time.time() - start) < deadline - 5.0:
        try:
            x0 = best_offsets.copy()
            r4 = minimize(
                lambda x: _obj(x, smooth=False),
                x0,
                method="Nelder-Mead",
                options={
                    "maxiter": 50000,
                    "xatol": 1e-16,
                    "fatol": 1e-18,
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
    
    # Phase 4b: Final polish on bipartite parameters with exact objective
    if (time.time() - start) < deadline - 2.0:
        # Try to extract bipartite-like parameters from best_pts if it came from bipartite
        # For now, just do a local polish around the current best if it looks bipartite
        p_norm = _normalize(best_pts)
        if p_norm is not None:
            # Check if points roughly alternate between two radii
            radii = np.linalg.norm(p_norm, axis=1)
            # Sort by angle
            angles = np.arctan2(p_norm[:, 1], p_norm[:, 0])
            order = np.argsort(angles)
            radii_sorted = radii[order]
            
            # Check if alternating radii pattern exists
            if len(radii_sorted) >= 20:
                r_odd = radii_sorted[::2]
                r_even = radii_sorted[1::2]
                mean_odd = np.mean(r_odd)
                mean_even = np.mean(r_even)
                var_odd = np.var(r_odd)
                var_even = np.var(r_even)
                
                # If the two groups have similar means and low variance, it's bipartite-like
                if (np.abs(mean_odd - mean_even) < 0.3 and 
                    var_odd < 0.01 and var_even < 0.01):
                    # Extract phi from angle differences
                    angles_sorted = angles[order]
                    # Average angle difference between consecutive odd/even points
                    phi_est = np.mean(angles_sorted[1::2] - angles_sorted[::2])
                    phi_est = np.mod(phi_est, np.pi/10)  # Should be small
                    
                    try:
                        r_bip4 = minimize(
                            lambda x: _obj_bipartite(x, smooth=False),
                            np.array([mean_even, mean_odd, phi_est]),
                            method="Nelder-Mead",
                            options={
                                "maxiter": 20000,
                                "xatol": 1e-16,
                                "fatol": 1e-18,
                                "adaptive": True
                            }
                        )
                        if r_bip4 is not None and r_bip4.x is not None:
                            pts = _build_bipartite(r_bip4.x[0], r_bip4.x[1], r_bip4.x[2])
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
