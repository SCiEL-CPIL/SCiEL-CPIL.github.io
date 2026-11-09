import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def get_score(points):
    """Compute score: min triangle area / convex hull area."""
    N = len(points)
    if N < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0
    
    min_area = compute_min_triangle_area(points)
    if min_area <= 0:
        return 0.0
    return min_area / hull_area

def soft_min(areas, tau):
    """
    Compute the soft minimum of an array of areas using log-sum-exp trick.
    soft_min(x) = min_val - 1/tau * log(sum(exp(-tau * (x - min_val))))
    """
    min_val = np.min(areas)
    shifted = areas - min_val
    # For numerical stability, clip large negative values
    shifted = np.maximum(shifted, -50.0 / tau)
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    return min_val - (1.0 / tau) * log_sum

def compute_all_areas(points):
    """Compute all triangle areas for n points."""
    n = len(points)
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    N = 13
    best_score = 0.0
    best_points = None
    
    def time_left():
        return deadline - time.time()
    
    # Helper: normalize points for optimization
    # Fix: point[0] at origin, point[1] on positive x-axis
    # This reduces 26 params to 24 (remove translation 2 + rotation 1, but we fix 1 angle)
    # Actually: 26 - 3 (2 trans + 1 rot) = 23, but we also can scale... 
    # Since score is scale invariant, fix |p1 - p0| = 1: 26 - 4 = 22 params
    # Let's use: p0 = [0,0], p1 = [1,0], rest free: 22 params
    
    def params_to_points(params):
        """Convert 22 params to 13 points with p0=[0,0], p1=[1,0]."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(N - 2, 2)
        return pts
    
    def points_to_params(pts):
        """Convert 13 points to 22 params (p0=[0,0], p1=[1,0] fixed)."""
        # Normalize: translate so p0 at origin, rotate so p1 on x-axis, scale so |p1-p0|=1
        p0 = pts[0]
        p1 = pts[1]
        d = p1 - p0
        dist = np.linalg.norm(d)
        if dist < 1e-10:
            return np.zeros(2 * (N - 2))
        # Translate
        pts_t = pts - p0
        # Rotate: angle of d
        angle = np.arctan2(d[1], d[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        pts_r = pts_t @ R.T
        # Scale
        pts_s = pts_r / dist
        # Extract free params
        return pts_s[2:].flatten()
    
    def smooth_objective(params, tau):
        """Smooth objective: -soft_min(areas) / hull_area"""
        try:
            pts = params_to_points(params)
            areas = compute_all_areas(pts)
            sm = soft_min(areas, tau)
            
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
            except:
                return 1e6
            
            score = sm / hull_area
            return -score
        except:
            return 1e6
    
    # Generate initial configurations
    def make_regular_ngon(n, center=None, radius=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        pts = np.column_stack((center[0] + radius * np.cos(angles),
                               center[1] + radius * np.sin(angles)))
        return pts
    
    # Initial guesses
    initial_configs = []
    
    # 1. Regular 13-gon
    initial_configs.append(make_regular_ngon(13))
    
    # 2. Regular 12-gon + center
    ring12 = make_regular_ngon(12)
    center_pt = np.array([[0.0, 0.0]])
    initial_configs.append(np.vstack([ring12, center_pt]))
    
    # 3. Regular 11-gon + 2 interior points
    ring11 = make_regular_ngon(11)
    interior = np.array([[0.3, 0.0], [-0.3, 0.0]])
    initial_configs.append(np.vstack([ring11, interior]))
    
    # 4. Two concentric rings: 7 outer + 6 inner
    outer7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.5)
    initial_configs.append(np.vstack([outer7, inner6]))
    
    # 5. 9 outer + 4 inner (square-ish)
    outer9 = make_regular_ngon(9, radius=1.0)
    inner4 = np.array([[0.4, 0.4], [-0.4, 0.4], [-0.4, -0.4], [0.4, -0.4]])
    initial_configs.append(np.vstack([outer9, inner4]))
    
    # 6. 10 outer + 3 inner
    outer10 = make_regular_ngon(10, radius=1.0)
    inner3 = make_regular_ngon(3, radius=0.4)
    initial_configs.append(np.vstack([outer10, inner3]))
    
    # Evaluate all initial configs
    for cfg in initial_configs:
        s = get_score(cfg)
        if s > best_score:
            best_score = s
            best_points = cfg.copy()
    
    # Optimize each initial config
    for cfg_idx, cfg in enumerate(initial_configs):
        if time_left() < 15.0:
            break
        
        x0 = points_to_params(cfg)
        
        # Progressive tau increase for sharper min approximation
        taus = [30, 50, 80, 120]
        
        for tau in taus:
            if time_left() < 10.0:
                break
            
            # Try SLSQP first (uses gradient info)
            try:
                res = minimize(smooth_objective, x0, args=(tau,), method='SLSQP',
                               options={'maxiter': 500, 'ftol': 1e-8})
                if res.success or res.fun < 1e5:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res.x)
                        x0 = res.x
            except:
                pass
            
            # Try L-BFGS-B
            if time_left() < 5.0:
                continue
            try:
                res2 = minimize(smooth_objective, x0, args=(tau,), method='L-BFGS-B',
                                options={'maxiter': 300, 'ftol': 1e-10})
                current_score = -res2.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = params_to_points(res2.x)
                    x0 = res2.x
            except:
                pass
        
        # Final refinement with high tau
        if time_left() > 10.0:
            try:
                res_final = minimize(smooth_objective, x0, args=(200,), method='L-BFGS-B',
                                     options={'maxiter': 200, 'ftol': 1e-12})
                current_score = -res_final.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = params_to_points(res_final.x)
            except:
                pass
    
    # Local search: perturb best points and re-optimize
    if best_points is not None and time_left() > 20.0:
        x_best = points_to_params(best_points)
        
        # Try several perturbations
        for pert_scale in [0.02, 0.05, 0.1]:
            if time_left() < 10.0:
                break
            n_perturb = 5
            for _ in range(n_perturb):
                if time_left() < 5.0:
                    break
                x_pert = x_best + np.random.normal(0, pert_scale, len(x_best))
                try:
                    res_p = minimize(smooth_objective, x_pert, args=(100,), method='L-BFGS-B',
                                     options={'maxiter': 200, 'ftol': 1e-10})
                    current_score = -res_p.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res_p.x)
                        x_best = res_p.x
                except:
                    pass
    
    # Final exact score check
    if best_points is not None:
        final_score = get_score(best_points)
        if final_score > best_score:
            best_score = final_score
    
    return best_points
