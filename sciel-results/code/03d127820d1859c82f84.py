import numpy as np
import time
from scipy.spatial import ConvexHull

N = 13

def _min_triangle_area_and_indices(points):
    """Return (min_area, (i,j,k)) for the minimum-area triangle using vectorized computation."""
    n = len(points)
    # Generate all triplets
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_f, j_f, k_f = i_idx[mask], j_idx[mask], k_idx[mask]
    
    a = points[i_f]
    b = points[j_f]
    c = points[k_f]
    
    # Vectorized area computation
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_pos = np.argmin(areas)
    min_area = areas[min_pos]
    min_idx = (i_f[min_pos], j_f[min_pos], k_f[min_pos])
    return min_area, min_idx

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area, _ = _min_triangle_area_and_indices(points)
    return min_area / hull_area

def _soft_min_triangle_area(points, temperature=0.01):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    Returns (soft_min_value, gradient_array) where gradient has shape (n, 2).
    """
    n = len(points)
    # Generate all triplets
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_f, j_f, k_f = i_idx[mask], j_idx[mask], k_idx[mask]
    
    a = points[i_f]
    b = points[j_f]
    c = points[k_f]
    
    # Vectorized area computation
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -T * log(sum(exp(-areas/T)))
    # For stability, subtract max
    max_area = np.max(areas)
    shifted = (max_area - areas) / temperature
    # Clip for numerical stability
    shifted = np.clip(shifted, -100, 100)
    weights = np.exp(shifted)
    sum_weights = np.sum(weights)
    soft_min = max_area - temperature * np.log(sum_weights)
    
    # Gradient of soft_min w.r.t. each point
    # d(soft_min)/d(point_p) = sum_k w_k * d(area_k)/d(point_p) / sum_weights
    # where w_k = exp((max_area - area_k)/T)
    
    grad = np.zeros((n, 2))
    
    # For each triangle, compute gradient contribution
    # area_k = 0.5 * |cross_k|
    # d(area_k)/da = 0.5 * sign(cross_k) * (-v1, v0) where v = c-a
    # d(area_k)/db = 0.5 * sign(cross_k) * (v1, -v0)
    # d(area_k)/dc = 0.5 * sign(cross_k) * (-u1, u0) where u = b-a
    
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    u = b - a  # b - a
    v = c - a  # c - a
    
    # Gradient contributions for each point in each triangle
    grad_a = 0.5 * signs[:, np.newaxis] * np.column_stack([-v[:, 1], v[:, 0]])
    grad_b = 0.5 * signs[:, np.newaxis] * np.column_stack([v[:, 1], -v[:, 0]])
    grad_c = 0.5 * signs[:, np.newaxis] * np.column_stack([-u[:, 1], u[:, 0]])
    
    # Weighted sum
    weights_col = weights[:, np.newaxis]
    grad[i_f] += np.sum(weights_col * grad_a, axis=0)
    grad[j_f] += np.sum(weights_col * grad_b, axis=0)
    grad[k_f] += np.sum(weights_col * grad_c, axis=0)
    
    grad /= sum_weights
    
    return soft_min, grad

def _subgradient_step(points, lr, hull_area):
    """
    Take a subgradient ascent step on min triangle area / hull area.
    Uses the exact min triangle for direction.
    """
    min_area, (i, j, k) = _min_triangle_area_and_indices(points)
    if min_area < 1e-12:
        # Degenerate: push points apart randomly
        rng = np.random.default_rng()
        points = points + rng.normal(0, 0.01, points.shape)
        return points
    
    a = points[i]
    b = points[j]
    c = points[k]
    
    cross_val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    sign = np.sign(cross_val) if cross_val != 0 else 1.0
    
    u = b - a
    v = c - a
    
    grad_a = np.array([-v[1], v[0]]) * 0.5 * sign
    grad_b = np.array([v[1], -v[0]]) * 0.5 * sign
    grad_c = np.array([-u[1], u[0]]) * 0.5 * sign
    
    new_points = points.copy()
    new_points[i] += lr * grad_a
    new_points[j] += lr * grad_b
    new_points[k] += lr * grad_c
    
    return new_points

def _optimize_from(points, deadline, max_iter=500):
    """
    Subgradient ascent to maximize min triangle area, with smooth surrogate polish.
    """
    points = points.copy()
    best_points = points.copy()
    best_score = get_score(points)
    
    lr = 0.01
    n_no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            break
        
        new_points = _subgradient_step(points, lr, hull_area)
        
        new_score = get_score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            n_no_improve = 0
            lr = min(lr * 1.05, 0.05)
        else:
            n_no_improve += 1
            lr *= 0.95
            if n_no_improve > 50:
                break
        
        points = new_points
    
    # Final polish: use soft-min smooth surrogate for finer convergence
    if time.time() < deadline - 2.0:
        polished = _polish_with_soft_min(best_points, deadline)
        polished_score = get_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points, best_score

def _polish_with_soft_min(points, deadline, max_iter=200):
    """
    Polish using the smooth soft-min surrogate objective.
    This allows gradient-based optimization that isn't stuck at kinks.
    """
    points = points.copy()
    best_points = points.copy()
    best_score = get_score(points)
    
    lr = 0.005
    temperature = 0.02
    n_no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            break
        
        soft_min, grad = _soft_min_triangle_area(points, temperature)
        
        # Scale gradient by 1/hull_area to account for normalization
        grad_scaled = grad / hull_area
        
        new_points = points + lr * grad_scaled
        
        new_score = get_score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            n_no_improve = 0
            # Decay temperature for sharper min approximation
            temperature = max(temperature * 0.97, 0.005)
        else:
            n_no_improve += 1
            lr *= 0.93
            if n_no_improve > 40:
                break
        
        points = new_points
    
    return best_points

def _generate_regular_polygon_configs():
    """Generate structured initial configurations."""
    configs = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Regular 12-gon + center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    configs.append(np.vstack([pts12, np.array([[0.0, 0.0]])]))
    
    # Regular 11-gon + 2 inner
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    pts11 = np.column_stack([np.cos(angles11), np.sin(angles11)])
    configs.append(np.vstack([pts11, np.array([[0.5, 0.0], [-0.5, 0.0]])]))
    
    # Regular 10-gon + 3 inner (triangle at center)
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    pts10 = np.column_stack([np.cos(angles10), np.sin(angles10)])
    inner_angles = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    inner = np.column_stack([0.4*np.cos(inner_angles), 0.4*np.sin(inner_angles)])
    configs.append(np.vstack([pts10, inner]))
    
    # Regular 9-gon + 4 inner (square at center)
    angles9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    pts9 = np.column_stack([np.cos(angles9), np.sin(angles9)])
    inner_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner = np.column_stack([0.35*np.cos(inner_angles), 0.35*np.sin(inner_angles)])
    configs.append(np.vstack([pts9, inner]))
    
    # Two concentric rings: 7 outer, 6 inner
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack([0.5*np.cos(angles6), 0.5*np.sin(angles6)])
    configs.append(np.vstack([outer, inner]))
    
    # 8 outer, 5 inner
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer = np.column_stack([np.cos(angles8), np.sin(angles8)])
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = np.column_stack([0.45*np.cos(angles5), 0.45*np.sin(angles5)])
    configs.append(np.vstack([outer, inner]))
    
    # 6 outer, 7 inner (two rings)
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer6 = np.column_stack([np.cos(angles6), np.sin(angles6)])
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    inner7 = np.column_stack([0.55*np.cos(angles7), 0.55*np.sin(angles7)])
    configs.append(np.vstack([outer6, inner7]))
    
    # 5 outer, 8 inner
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    outer5 = np.column_stack([np.cos(angles5), np.sin(angles5)])
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner8 = np.column_stack([0.5*np.cos(angles8), 0.5*np.sin(angles8)])
    configs.append(np.vstack([outer5, inner8]))
    
    # Three concentric rings: 5 outer, 5 middle, 3 inner
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    ring1 = np.column_stack([np.cos(angles5), np.sin(angles5)])
    angles5b = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    ring2 = np.column_stack([0.6*np.cos(angles5b), 0.6*np.sin(angles5b)])
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    ring3 = np.column_stack([0.3*np.cos(angles3), 0.3*np.sin(angles3)])
    configs.append(np.vstack([ring1, ring2, ring3]))
    
    # Random with seed
    for seed in [1, 7, 13, 99, 42, 123]:
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2))
        configs.append(pts)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    configs = _generate_regular_polygon_configs()
    
    best_score = -np.inf
    best_points = None
    
    # Phase 1: Quick optimization from all configs
    for idx, init_points in enumerate(configs):
        if time.time() > deadline - 30.0:
            break
        
        try:
            optimized, score = _optimize_from(init_points, deadline - 30.0, max_iter=200)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            pass
    
    # Phase 2: Deep optimization from best few configs
    if best_points is not None and time.time() < deadline - 5.0:
        # Re-optimize the best config with more iterations
        try:
            optimized, score = _optimize_from(best_points, deadline - 2.0, max_iter=800)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            pass
        
        # Also try from the top 2 initial configs with more budget
        for idx in [0, 1]:
            if time.time() > deadline - 5.0:
                break
            try:
                optimized, score = _optimize_from(configs[idx], deadline - 2.0, max_iter=500)
                if score > best_score:
                    best_score = score
                    best_points = optimized
            except Exception:
                pass
    
    if best_points is None:
        best_points = configs[0]
    
    return best_points
