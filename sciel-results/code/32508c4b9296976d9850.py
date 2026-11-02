import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
# C(14, 3) = 364
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _soft_min_objective(angles, temperature=0.01):
    """
    Soft-min objective using log-sum-exp for smoothness.
    angles: array of shape (N,) with sorted angles in [0, 2*pi)
    """
    # Ensure angles are sorted and in [0, 2*pi)
    angles = np.sort(angles)
    # Normalize to [0, 2*pi)
    angles = angles - angles[0]
    if angles[-1] < 2*np.pi - 1e-10:
        # Add the wrap-around gap
        angles = np.append(angles, angles[-1] + (2*np.pi - angles[-1]))
    
    # Place points on unit circle
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Compute all triangle areas
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    
    # Soft min using log-sum-exp
    shifted = -areas / temperature
    shifted -= np.max(shifted)  # For numerical stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = -(log_sum + np.max(shifted)) * temperature
    
    # The hull area for points on unit circle is fixed
    # But we need to normalize by actual hull area
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    return -soft_min_val / hull_area

def _angle_to_points(angles):
    """Convert sorted angles to points on unit circle."""
    angles = np.sort(angles)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _points_to_angles(points):
    """Convert points to their angles (for convex position)."""
    # Only valid if points are in convex position
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    return np.sort(angles)

def _soft_min_cartesian(flat_points, temperature=0.01):
    """
    Soft-min objective in Cartesian space for L-BFGS-B.
    flat_points: array of shape (2*N,)
    """
    points = flat_points.reshape(N, 2)
    
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    
    # Soft min using log-sum-exp
    shifted = -areas / temperature
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = -(log_sum + np.max(shifted)) * temperature
    
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1.0  # Large penalty
    
    return -soft_min_val / hull_area

def _soft_min_cartesian_grad(flat_points, temperature=0.01):
    """
    Numerical gradient for soft-min Cartesian objective.
    """
    eps = 1e-7
    grad = np.zeros_like(flat_points)
    f0 = _soft_min_cartesian(flat_points, temperature)
    for i in range(len(flat_points)):
        fp = flat_points.copy()
        fp[i] += eps
        grad[i] = (_soft_min_cartesian(fp, temperature) - f0) / eps
    return grad

def _make_diverse_inits():
    """Create diverse initial configurations."""
    inits = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(angles.copy())
    
    # Perturbed regular
    for amp in [0.05, 0.1, 0.2]:
        for freq in [0.5, 1.0, 1.5]:
            a = angles + amp * np.sin(np.arange(N) * freq)
            inits.append(a)
    
    # Concentric ring pattern: some points on inner ring, some on outer
    # 7 on inner (radius 0.5), 7 on outer (radius 1.0)
    inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_angles = np.linspace(np.pi/7, 2*np.pi, 7, endpoint=False)
    # Convert to angle-only representation (all on unit circle for angle optimization)
    # For angle optimization, we just need sorted angles
    mixed_angles = np.sort(np.concatenate([inner_angles, outer_angles]))
    inits.append(mixed_angles)
    
    # Hexagonal lattice-inspired angles
    hex_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3,
                           np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6,
                           np.pi/12, 5*np.pi/12])
    inits.append(np.sort(hex_angles))
    
    # Random perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        a = angles + rng.normal(0, 0.15, N)
        inits.append(a)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (150s limit, keep 15s buffer)
    
    best_points = None
    best_score = 0.0
    
    def _elapsed():
        return time.time() - start_time
    
    def _remaining():
        return deadline - time.time()
    
    # Strategy 1: Optimize in angle space with multiple methods and diverse inits
    angle_inits = _make_diverse_inits()
    
    # Objective for angle optimization
    def angle_objective(angles_flat):
        angles = angles_flat.copy()
        angles = np.mod(angles, 2*np.pi)
        angles = np.sort(angles)
        return _soft_min_objective(angles, temperature=0.01)
    
    # Try L-BFGS-B on angles (smooth objective)
    for i, ang_init in enumerate(angle_inits):
        if _remaining() < 10.0:
            break
        
        ang_init = np.mod(ang_init, 2*np.pi)
        ang_init = np.sort(ang_init)
        
        # L-BFGS-B with bounds
        try:
            bounds = [(0, 2*np.pi)] * N
            res = minimize(angle_objective, ang_init, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < angle_objective(ang_init):
                cand_angles = np.mod(res.x, 2*np.pi)
                cand_angles = np.sort(cand_angles)
                cand_points = _angle_to_points(cand_angles)
                cand_score = _score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
        except Exception:
            pass
        
        if _remaining() < 10.0:
            break
        
        # Nelder-Mead on angles
        try:
            res_nm = minimize(angle_objective, ang_init, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res_nm.fun < angle_objective(ang_init):
                cand_angles = np.mod(res_nm.x, 2*np.pi)
                cand_angles = np.sort(cand_angles)
                cand_points = _angle_to_points(cand_angles)
                cand_score = _score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
        except Exception:
            pass
    
    # Strategy 2: Cartesian space optimization with L-BFGS-B using soft-min
    if best_points is not None and _remaining() > 15.0:
        # Use soft-min Cartesian objective for smoother optimization
        try:
            res_lbfgs = minimize(_soft_min_cartesian, best_points.flatten(), 
                                 method='L-BFGS-B',
                                 options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_lbfgs.fun < _soft_min_cartesian(best_points.flatten()):
                cand_pts = res_lbfgs.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_points = cand_pts
                    best_score = cand_score
        except Exception:
            pass
    
    # Strategy 3: Refine with Nelder-Mead in Cartesian space
    if best_points is not None and _remaining() > 10.0:
        def cart_objective(flat_points):
            pts = flat_points.reshape(N, 2)
            return -_score(pts)
        
        try:
            res3 = minimize(cart_objective, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res3.fun < cart_objective(best_points.flatten()):
                cand_pts = res3.x.reshape(N, 2)
                cand_score = -res3.fun
                if cand_score > best_score:
                    best_points = cand_pts
                    best_score = cand_score
        except Exception:
            pass
    
    # Strategy 4: Final polish
    if best_points is not None and _remaining() > 5.0:
        def cart_objective(flat_points):
            pts = flat_points.reshape(N, 2)
            return -_score(pts)
        
        try:
            res4 = minimize(cart_objective, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res4.fun < cart_objective(best_points.flatten()):
                cand_pts = res4.x.reshape(N, 2)
                cand_score = -res4.fun
                if cand_score > best_score:
                    best_points = cand_pts
                    best_score = cand_score
        except Exception:
            pass
    
    # Fallback: regular 14-gon
    if best_points is None or best_score < 0.001:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)
    
    return best_points
