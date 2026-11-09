import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _score_polygon(polygon):
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_d = dists_to_centroid.mean()
    if mean_d <= 0.01:
        return 0.0
    centered = centered / mean_d
    
    # Check convexity: no repeated points, all turns same sign, total turn = ±2π
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(centered[i] - centered[j]) < 1e-10:
                return 0.0
    
    cross_products = []
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        cross_products.append(cross)
    
    nonzero = [c for c in cross_products if abs(c) > 1e-12]
    if len(nonzero) != n:
        return 0.0
    if any(c > 0 for c in nonzero) and any(c < 0 for c in nonzero):
        return 0.0
    
    angles = []
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        if norm_e1 < 1e-12 or norm_e2 < 1e-12:
            return 0.0
        cos_angle = np.dot(e1, e2) / (norm_e1 * norm_e2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if cross_products[i] > 0:
            angle = angle
        else:
            angle = -angle
        angles.append(angle)
    
    total_turn = sum(angles)
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return 0.0
    
    min_score = 1.0
    for i in range(n):
        v = centered[i]
        d = np.linalg.norm(centered - v, axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)[:19]
        
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
        
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        
        vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        min_score = min(min_score, vertex_score)
    
    return min_score


def _build_polygon_from_offsets(offsets):
    """Build 20-gon with 4-fold rotational symmetry from 5 offsets."""
    n = 20
    base_offsets = np.sort(offsets)
    angles = np.zeros(20)
    for i in range(20):
        idx = i % 5
        block = i // 5
        angles[i] = base_offsets[idx] + block * (np.pi / 2.0)
    angles = np.sort(angles)
    angles = angles % (2 * np.pi)
    angles = np.sort(angles)
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    return polygon


def _score_symmetric(offsets):
    """Score a 20-gon with 4-fold rotational symmetry."""
    polygon = _build_polygon_from_offsets(offsets)
    return _score_polygon(polygon)


def _fast_score_symmetric(offsets):
    """Faster scoring for optimization loop."""
    polygon = _build_polygon_from_offsets(offsets)
    n = 20
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_d = dists_to_centroid.mean()
    if mean_d <= 0.01:
        return 0.0
    centered = centered / mean_d
    
    # Quick convexity check
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross_products[i] = e1[0]*e2[1] - e1[1]*e2[0]
    
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    if np.any(cross_products > 0) and np.any(cross_products < 0):
        return 0.0
    
    # Total turning
    total_turn = 0.0
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        cos_a = np.dot(e1, e2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        a = np.arccos(cos_a)
        if cross_products[i] < 0:
            a = -a
        total_turn += a
    
    if abs(abs(total_turn) - 2*np.pi) > 1e-5:
        return 0.0
    
    # Vertex scores
    min_score = 1.0
    for i in range(n):
        v = centered[i]
        d = np.linalg.norm(centered - v, axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)[:19]
        
        if d_sorted[0] < 0.05:
            return 0.0
        
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            return 0.0
        
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        min_score = min(min_score, vertex_score)
    
    return min_score


def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin
    
    n_offsets = 5
    # Regular polygon: equally spaced in [0, pi/2)
    base_angles = np.linspace(0.0, np.pi/2.0, n_offsets, endpoint=False)
    best_score = _score_symmetric(base_angles)
    best_offsets = base_angles.copy()
    
    # Stage 1: Multiple random restarts with differential evolution
    rng = np.random.default_rng(42)
    bounds = [(0.0, np.pi/2.0)] * n_offsets
    
    def objective(offsets_flat):
        if time.time() > deadline - 30.0:
            return 0.0
        s = _fast_score_symmetric(offsets_flat)
        return -s
    
    # Try several different seeds and initial populations
    for seed in [42, 123, 456, 789, 1024]:
        if time.time() > deadline - 60.0:
            break
        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=seed,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol',
                updating='deferred'
            )
            
            if result.fun < 0:
                opt_offsets = result.x
                opt_score = _score_symmetric(opt_offsets)
                if opt_score > best_score:
                    best_score = opt_score
                    best_offsets = opt_offsets.copy()
        except Exception:
            pass
    
    # Stage 2: Local refinement with coordinate descent
    if time.time() < deadline - 20.0:
        current_offsets = best_offsets.copy()
        current_score = best_score
        
        # Adaptive coordinate descent
        step_sizes = np.full(n_offsets, 0.05)
        for iteration in range(20000):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            for idx in range(n_offsets):
                # Try positive and negative steps
                for sign in [1.0, -1.0]:
                    new_offsets = current_offsets.copy()
                    new_offsets[idx] += sign * step_sizes[idx]
                    new_offsets[idx] = np.clip(new_offsets[idx], 0.0, np.pi/2.0)
                    
                    new_score = _fast_score_symmetric(new_offsets)
                    if new_score > current_score:
                        current_offsets = new_offsets
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_offsets = current_offsets.copy()
                        step_sizes[idx] *= 1.5
                        improved = True
                        break
                else:
                    step_sizes[idx] *= 0.7
            
            if not improved and np.all(step_sizes < 1e-8):
                break
    
    # Stage 3: Fine-tune with scipy minimize on the best solution
    if time.time() < deadline - 15.0:
        def neg_score(offsets):
            return -_fast_score_symmetric(offsets)
        
        try:
            res = minimize(
                neg_score,
                best_offsets,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < 0:
                res_score = _score_symmetric(res.x)
                if res_score > best_score:
                    best_score = res_score
                    best_offsets = res.x.copy()
        except Exception:
            pass
    
    # Build final polygon
    polygon = _build_polygon_from_offsets(best_offsets)
    return polygon
