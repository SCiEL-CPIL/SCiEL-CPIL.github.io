"""Program for Heilbronn convex regions n=14 using polar coordinates with sorted angles and differential evolution."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

DEADLINE = 110.0  # Safety margin before 120s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Compute minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a


def _polar_to_cartesian(params):
    """
    Convert parameters to Cartesian coordinates.
    params: [r_0, r_1, ..., r_{N-1}, dtheta_1, dtheta_2, ..., dtheta_{N-1}]
    where theta_0 = 0 and theta_i = sum(dtheta_1..dtheta_i)
    Total parameters: N + (N-1) = 2N-1
    """
    n = N
    radii = params[:n]
    dthetas = params[n:]  # N-1 values
    
    # Build angles: theta_0 = 0, theta_i = cumsum(dthetas[:i])
    angles = np.zeros(n)
    if n > 1:
        angles[1:] = np.cumsum(dthetas)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _objective(params):
    """Objective function for minimization: negative of score."""
    points = _polar_to_cartesian(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 1e10
    min_area = _min_triangle_area(points)
    if min_area < 1e-14:
        return 1e10
    return -min_area / hull_a


def _initial_params_regular():
    """Generate initial polar parameters for a regular polygon."""
    # Regular N-gon with circumradius R
    R = np.sqrt(2.0 / (N * np.sin(2.0 * np.pi / N)))
    
    radii = np.ones(N) * R
    dthetas = np.full(N - 1, 2.0 * np.pi / N)
    
    return np.concatenate([radii, dthetas])


def _local_refine(params, max_iter=200, tol=1e-12):
    """Local refinement using L-BFGS-B with multiple attempts."""
    best_params = params.copy()
    best_val = _objective(params)
    
    # Try different initial perturbations
    for attempt in range(3):
        try:
            if attempt == 0:
                p0 = params.copy()
            else:
                p0 = params + np.random.normal(0, 0.01, size=params.shape)
            
            result = minimize(
                _objective,
                p0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-9}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params


def _fine_local_refine(params, max_iter=500):
    """Very fine local refinement."""
    try:
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.fun < _objective(params):
            return result.x
        return params
    except Exception:
        return params


def construct():
    start_time = time.time()
    
    # Define bounds for polar coordinates
    # Radii: [0.05, 4.0] - wider range to explore
    # dthetas: [0.005, 2*pi/N * 4] - allow more angular variation
    min_dtheta = 0.005
    max_dtheta = 2.0 * np.pi / N * 4.0
    
    r_bounds = [(0.05, 4.0)] * N
    dtheta_bounds = [(min_dtheta, max_dtheta)] * (N - 1)
    bounds = r_bounds + dtheta_bounds
    
    # Initial guess: regular polygon
    x0 = _initial_params_regular()
    
    best_score = 0.0
    best_params = None
    
    # Stage 1: Coarse differential evolution with larger population
    try:
        result1 = differential_evolution(
            _objective,
            bounds=bounds,
            x0=x0,
            seed=42,
            maxiter=400,
            tol=1e-8,
            popsize=15,
            polish=False,
            init='sobol',
            updating='immediate',
            mutation=(0.3, 1.0),
            recombination=0.8
        )
        
        if result1.x is not None:
            cand_pts = _polar_to_cartesian(result1.x)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_params = result1.x
    except Exception:
        pass
    
    # Local refinement after coarse DE
    if best_params is not None and time.time() - start_time < DEADLINE - 40:
        try:
            refined = _local_refine(best_params, max_iter=300)
            cand_pts = _polar_to_cartesian(refined)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_params = refined
        except Exception:
            pass
    
    # Stage 2: Fine differential evolution around best solution
    if time.time() - start_time < DEADLINE - 30 and best_params is not None:
        try:
            # Tighter bounds around current best
            center = best_params
            spread = 0.3
            fine_bounds = []
            for i, (lo, hi) in enumerate(bounds):
                center_val = center[i]
                fine_lo = max(lo, center_val - spread)
                fine_hi = min(hi, center_val + spread)
                if fine_lo < fine_hi:
                    fine_bounds.append((fine_lo, fine_hi))
                else:
                    fine_bounds.append((lo, hi))
            
            result2 = differential_evolution(
                _objective,
                bounds=fine_bounds,
                x0=center,
                seed=43,
                maxiter=300,
                tol=1e-10,
                popsize=12,
                polish=False,
                init='sobol',
                updating='immediate',
                mutation=(0.2, 0.8),
                recombination=0.7
            )
            
            if result2.x is not None:
                cand_pts = _polar_to_cartesian(result2.x)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_params = result2.x
        except Exception:
            pass
    
    # Fine local refinement
    if best_params is not None and time.time() - start_time < DEADLINE - 15:
        try:
            refined2 = _fine_local_refine(best_params, max_iter=500)
            cand_pts = _polar_to_cartesian(refined2)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_params = refined2
        except Exception:
            pass
    
    # Stage 3: Another DE run with different seed and mutation strategy
    if time.time() - start_time < DEADLINE - 20:
        try:
            result3 = differential_evolution(
                _objective,
                bounds=bounds,
                x0=x0 + np.random.normal(0, 0.1, size=x0.shape),
                seed=44,
                maxiter=250,
                tol=1e-10,
                popsize=10,
                polish=False,
                init='sobol',
                updating='immediate',
                mutation=(0.5, 1.2),
                recombination=0.9
            )
            
            if result3.x is not None:
                refined3 = _local_refine(result3.x, max_iter=200)
                cand_pts = _polar_to_cartesian(refined3)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_params = refined3
        except Exception:
            pass
    
    # Final polish
    if best_params is not None and time.time() - start_time < DEADLINE - 5:
        try:
            final = _fine_local_refine(best_params, max_iter=1000)
            cand_pts = _polar_to_cartesian(final)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_params = final
        except Exception:
            pass
    
    # Fallback: regular polygon
    if best_params is None:
        best_params = x0
    
    return _polar_to_cartesian(best_params)
# EVOLVE-BLOCK-END
