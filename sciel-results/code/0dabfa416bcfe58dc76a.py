"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Compute the ratio: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _objective_full(x):
    """Objective for full 28D optimization."""
    points = x.reshape(-1, 2)
    return -_score(points)

# Fixed angles for radial parameterization
_ANGLES_RAD = np.linspace(0, 2 * np.pi, N, endpoint=False)

def _points_from_radii(radii):
    """Convert radial coordinates to Cartesian points."""
    x = radii * np.cos(_ANGLES_RAD)
    y = radii * np.sin(_ANGLES_RAD)
    return np.column_stack((x, y))

def _objective_radial(radii):
    """Objective for 14D radial optimization."""
    points = _points_from_radii(radii)
    return -_score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Initial guess: regular 14-gon on unit circle
    x0_full = np.column_stack([np.cos(_ANGLES_RAD), np.sin(_ANGLES_RAD)]).flatten()
    r0_radial = np.ones(N)
    
    best_x = x0_full.copy()
    best_score = _score(x0_full.reshape(-1, 2))
    
    # --- Track 1: Full 28D Basin Hopping ---
    # Allocate ~50% of time
    t1_end = start_time + (deadline - start_time) * 0.5
    
    def stop_cb_1(x, f, accept, gfc, step, info):
        return time.time() > t1_end - 5.0
        
    try:
        bh1 = basinhopping(
            _objective_full,
            x0_full,
            minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}},
            niter=100,
            stepsize=0.5,
            callback=stop_cb_1,
            seed=42
        )
        if bh1.fun < _objective_full(best_x):
            best_x = bh1.x
            best_score = _score(best_x.reshape(-1, 2))
    except Exception:
        pass

    # --- Track 2: Radial 14D Basin Hopping ---
    # Allocate remaining time
    t2_end = deadline - 10.0 # Leave 10s for final refinement
    
    def stop_cb_2(x, f, accept, gfc, step, info):
        return time.time() > t2_end - 5.0
        
    try:
        bh2 = basinhopping(
            _objective_radial,
            r0_radial,
            minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}},
            niter=100,
            stepsize=0.2,
            callback=stop_cb_2,
            seed=123
        )
        # Convert result to full coordinates
        x_radial_res = _points_from_radii(bh2.x).flatten()
        if bh2.fun < _objective_full(best_x):
            best_x = x_radial_res
            best_score = _score(best_x.reshape(-1, 2))
    except Exception:
        pass

    # --- Final Refinement ---
    if time.time() < deadline - 2:
        try:
            res = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < _objective_full(best_x):
                best_x = res.x
                best_score = _score(best_x.reshape(-1, 2))
        except Exception:
            pass

    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
