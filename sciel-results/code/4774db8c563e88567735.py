import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Pre-compute all triangle index combinations
    tri_indices = np.array(list(combinations(range(N), 3)))
    
    def compute_min_tri_area(pts):
        """Compute minimum triangle area for N points."""
        # Vectorized: for each triangle (i,j,k), area = 0.5*|(pj-pi) x (pk-pi)|
        pi = pts[tri_indices[:, 0]]
        pj = pts[tri_indices[:, 1]]
        pk = pts[tri_indices[:, 2]]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return np.min(areas)
    
    def hull_area(pts):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1e-10
    
    def score(pts):
        """Score = min triangle area / hull area."""
        h_area = hull_area(pts)
        if h_area < 1e-10:
            return 0.0
        m_area = compute_min_tri_area(pts)
        return m_area / h_area
    
    def neg_score_angular(angles):
        """Negative score for angular parameterization on unit circle."""
        if time.time() > deadline:
            return 0.0
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return -score(pts)
    
    def neg_score_free(flat_pts):
        """Negative score for free 2D points."""
        if time.time() > deadline:
            return 0.0
        pts = flat_pts.reshape(N, 2)
        return -score(pts)
    
    best_score_val = 0.0
    best_result = None
    
    # Strategy 1: Optimize angular positions on unit circle
    # Start with regular 14-gon
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Try multiple perturbations of the regular polygon
    rng = np.random.default_rng(42)
    
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        
        if trial == 0:
            angles0 = base_angles.copy()
        else:
            # Perturb angles
            angles0 = base_angles + rng.normal(0, 0.05 * (trial + 1), N)
            # Normalize to keep in [0, 2*pi)
            angles0 = np.sort(angles0) % (2*np.pi)
        
        try:
            res = minimize(neg_score_angular, angles0, method='Powell',
                          options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            if res.fun < 0:
                s = -res.fun
                if s > best_score_val:
                    best_score_val = s
                    best_result = np.column_stack([np.cos(res.x), np.sin(res.x)])
        except Exception:
            pass
    
    # Strategy 2: Free 2D optimization starting from best angular result
    if best_result is not None and time.time() < deadline - 10:
        x0_free = best_result.flatten()
        try:
            res = minimize(neg_score_free, x0_free, method='Powell',
                          options={'maxiter': 300, 'ftol': 1e-12, 'disp': False})
            if res.fun < 0:
                s = -res.fun
                if s > best_score_val:
                    best_score_val = s
                    best_result = res.x.reshape(N, 2)
        except Exception:
            pass
    
    # Strategy 3: Try Nelder-Mead on angular parameters with different starts
    if time.time() < deadline - 10:
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            angles0 = base_angles + rng.normal(0, 0.02, N)
            angles0 = np.sort(angles0) % (2*np.pi)
            try:
                res = minimize(neg_score_angular, angles0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
                if res.fun < 0:
                    s = -res.fun
                    if s > best_score_val:
                        best_score_val = s
                        best_result = np.column_stack([np.cos(res.x), np.sin(res.x)])
            except Exception:
                pass
    
    # Strategy 4: Try elliptical parameterization
    if time.time() < deadline - 10:
        def neg_score_ellipse(params):
            if time.time() > deadline:
                return 0.0
            angles = params[:N]
            a = params[N]  # semi-major axis
            b = params[N+1]  # semi-minor axis
            if a < 0.1 or b < 0.1:
                return 0.0
            pts = np.column_stack([a * np.cos(angles), b * np.sin(angles)])
            return -score(pts)
        
        # Start with circle (a=b=1)
        params0 = np.concatenate([base_angles, [1.0, 1.0]])
        try:
            res = minimize(neg_score_ellipse, params0, method='Powell',
                          options={'maxiter': 300, 'ftol': 1e-12, 'disp': False})
            if res.fun < 0:
                s = -res.fun
                if s > best_score_val:
                    best_score_val = s
                    angles = res.x[:N]
                    a = res.x[N]
                    b = res.x[N+1]
                    best_result = np.column_stack([a * np.cos(angles), b * np.sin(angles)])
        except Exception:
            pass
    
    # Final: if no result, return regular polygon
    if best_result is None:
        best_result = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    return best_result
