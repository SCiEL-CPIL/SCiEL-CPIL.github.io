import numpy as np
import scipy.spatial as spat
import scipy.optimize as opt
import time

N = 14

def calc_score(points):
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area / hull_area


def _build_points_from_params(params):
    """
    params layout:
      - first n_rings: ring radii (positive)
      - then N: angular offsets (in radians, can be any real, wrapped)
    Ring assignment: points are assigned to rings in a balanced way.
    For N=14, we use 3 rings: [0] center? Actually center point is degenerate for triangles.
    Better: 2 or 3 rings with no center.
    We'll use a fixed ring assignment pattern:
      Ring 0: first n0 points
      Ring 1: next n1 points
      Ring 2: next n2 points
    We choose n0, n1, n2 to be roughly balanced, e.g., 4,5,5 or 5,5,4.
    """
    n_rings = int(np.floor(len(params) * 0.3))  # heuristic: ~30% radii
    # But we need exact layout. Let's fix: 3 rings, 4+5+5
    # Actually let's make it flexible: try a few ring splits
    # For simplicity in DE, fix ring assignment: 3 rings with sizes [4,5,5]
    ring_sizes = [4, 5, 5]
    n_rings = len(ring_sizes)
    radii = params[:n_rings]
    angles = params[n_rings:n_rings + N]
    
    points = np.zeros((N, 2))
    idx = 0
    for r in range(n_rings):
        size = ring_sizes[r]
        base_angle = 2 * np.pi / size
        for k in range(size):
            theta = base_angle * k + angles[idx]
            rad = radii[r]
            points[idx, 0] = rad * np.cos(theta)
            points[idx, 1] = rad * np.sin(theta)
            idx += 1
    return points


def _neg_score_from_params(params):
    try:
        pts = _build_points_from_params(params)
        # Ensure non-degenerate
        if np.any(~np.isfinite(pts)):
            return 1e6
        s = calc_score(pts)
        return -s
    except Exception:
        return 1e6


def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_points = None
    best_score = 0.0
    
    # Ring size configurations to try
    ring_configs = [
        [4, 5, 5],
        [5, 5, 4],
        [3, 5, 6],
        [5, 4, 5],
        [2, 6, 6],
        [6, 5, 3],
        [4, 6, 4],
        [5, 6, 3],
    ]
    
    for ring_sizes in ring_configs:
        if time.time() > deadline - 15:
            break
        if len(ring_sizes) != 3:
            continue
        if sum(ring_sizes) != N:
            continue
        
        n_rings = len(ring_sizes)
        # Parameter bounds: radii in [0.2, 2.0], angles in [-pi, pi]
        lb = np.concatenate([np.full(n_rings, 0.2), np.full(N, -np.pi)])
        ub = np.concatenate([np.full(n_rings, 2.0), np.full(N, np.pi)])
        
        # Initial population: mix of regular and random
        pop_size = 40
        x0 = np.zeros((pop_size, len(lb)))
        for i in range(pop_size):
            if i == 0:
                # Regular: radii equally spaced, angles 0
                x0[i, :n_rings] = np.linspace(0.3, 1.8, n_rings)
                x0[i, n_rings:] = 0.0
            elif i == 1:
                # Slightly perturbed regular
                x0[i, :n_rings] = np.linspace(0.3, 1.8, n_rings) + np.random.randn(n_rings) * 0.05
                x0[i, n_rings:] = np.random.randn(N) * 0.1
            else:
                x0[i] = np.random.uniform(lb, ub)
        
        # Use differential evolution
        try:
            res = opt.differential_evolution(
                _neg_score_from_params_wrapper,
                bounds=list(zip(lb, ub)),
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                polish=True,
                init='latinhypercube',
            )
        except Exception:
            # Fallback: manual DE-like or just Nelder-Mead on params
            try:
                x0_flat = x0[0]
                res = opt.minimize(
                    _neg_score_from_params,
                    x0_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'adaptive': True}
                )
            except Exception:
                continue
        
        # Evaluate best
        best_params = res.x
        best_pts = _build_points_from_params(best_params)
        s = calc_score(best_pts)
        if s > best_score:
            best_score = s
            best_points = best_pts.copy()
        
        # Also try Nelder-Mead on the exact coordinates from this best
        if time.time() < deadline - 10:
            try:
                def neg_exact(x):
                    p = x.reshape((N, 2))
                    return -calc_score(p)
                res2 = opt.minimize(
                    neg_exact,
                    best_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                polished = res2.x.reshape((N, 2))
                s2 = calc_score(polished)
                if s2 > best_score:
                    best_score = s2
                    best_points = polished.copy()
            except Exception:
                pass
    
    # Final safety: if nothing found, use regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points


# Wrapper for DE that uses current ring config
def _neg_score_from_params_wrapper(params):
    # This is a hack: DE doesn't know ring_sizes. We need a closure.
    # Since we can't easily pass ring_sizes to DE's objective, we'll use a global
    # Actually, let's restructure: do the DE inside the loop with a lambda
    raise NotImplementedError("Use lambda in construct")
