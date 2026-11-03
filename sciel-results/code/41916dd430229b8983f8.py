"""Optimized max-min triangle area: barycentric LP + local refinement."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

H = np.sqrt(3) / 2


def _all_triples():
    """Precompute all C(11,3)=165 triples."""
    return np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])


_TRIPLES = _all_triples()
_N_TRIPLES = _TRIPLES.shape[0]


def _bary_to_cart(bary):
    """Convert barycentric (b2, b3) to Cartesian. bary shape (n, 2)."""
    b = bary[:, 0]
    c = bary[:, 1]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])


def _cart_to_bary(cart):
    """Convert Cartesian to barycentric (b2, b3). cart shape (n, 2)."""
    x = cart[:, 0]
    y = cart[:, 1]
    c = y / H
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _min_triangle_area_fast(points):
    """Fast vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _is_inside_triangle(points, eps=1e-6):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    
    if np.any(y < -eps) or np.any(y > H + eps) or np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    
    left_mask = x <= 0.5
    if np.any(left_mask):
        if np.any(y[left_mask] > np.sqrt(3) * x[left_mask] + eps):
            return False
    
    right_mask = ~left_mask
    if np.any(right_mask):
        if np.any(y[right_mask] > np.sqrt(3) * (1 - x[right_mask]) + eps):
            return False
    
    return True


def _clip_to_triangle(points):
    """Clip points to be inside the triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    points[:, 0] = x
    points[:, 1] = y
    
    left_mask = x <= 0.5
    if np.any(left_mask):
        max_y_left = np.sqrt(3) * x[left_mask]
        y[left_mask] = np.minimum(y[left_mask], max_y_left)
    
    right_mask = ~left_mask
    if np.any(right_mask):
        max_y_right = np.sqrt(3) * (1 - x[right_mask])
        y[right_mask] = np.minimum(y[right_mask], max_y_right)
    
    return points


def _objective_bary(bary_flat):
    """Objective in barycentric coords: -min_area + penalty for out-of-triangle."""
    bary = bary_flat.reshape(N, 2)
    # Clip to valid barycentric range: b >= 0, c >= 0, b + c <= 1
    b = np.clip(bary[:, 0], 0, 1)
    c = np.clip(bary[:, 1], 0, 1)
    # Ensure b + c <= 1
    excess = np.maximum(b + c - 1, 0)
    b = b - 0.5 * excess
    c = c - 0.5 * excess
    b = np.clip(b, 0, 1)
    c = np.clip(c, 0, 1)
    
    points = _bary_to_cart(np.column_stack([b, c]))
    return -_min_triangle_area_fast(points)


def _refine_local(points, max_iter=500, tol=1e-12):
    """Local refinement using active-set gradient method."""
    points = points.copy()
    n = points.shape[0]
    
    for it in range(max_iter):
        p1 = points[_TRIPLES[:, 0]]
        p2 = points[_TRIPLES[:, 1]]
        p3 = points[_TRIPLES[:, 2]]
        
        d1 = p2 - p1
        d2 = p3 - p1
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        # Active set: triangles with area within tolerance of min
        active_mask = areas <= min_area + 1e-10
        active_idx = np.where(active_mask)[0]
        
        if len(active_idx) == 0:
            break
        
        # Compute gradient of min area w.r.t. each point
        # For each active triangle, compute gradient and average
        grad = np.zeros((n, 2))
        for idx in active_idx:
            i, j, k = _TRIPLES[idx]
            a = points[i]
            b = points[j]
            c = points[k]
            
            d1 = b - a
            d2 = c - a
            sign = np.sign(cross[idx])
            if sign == 0:
                sign = 1.0
            
            factor = 0.5 * sign
            grad[i] += factor * (-(np.array([d2[1], -d2[0]])) - np.array([-d1[1], d1[0]]))
            grad[j] += factor * np.array([d2[1], -d2[0]])
            grad[k] += factor * np.array([-d1[1], d1[0]])
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        grad /= gnorm
        
        # Line search with backtracking
        best_step = 0.0
        best_area = min_area
        step = 0.05
        for _ in range(20):
            new_points = points + step * grad
            new_points = _clip_to_triangle(new_points)
            
            if not _is_inside_triangle(new_points, eps=1e-8):
                step *= 0.5
                continue
            
            new_area = _min_triangle_area_fast(new_points)
            if new_area > best_area + 1e-15:
                best_area = new_area
                best_step = step
            step *= 0.5
        
        if best_step <= 0:
            break
        
        points = _clip_to_triangle(points + best_step * grad)
        
        if best_area - min_area < tol:
            break
    
    return points


def _generate_structured_inits():
    """Generate structured initial configurations with symmetry."""
    inits = []
    
    # Highly symmetric: 3 corners + 3 edge midpoints + 5 interior
    bary_sets = [
        # Corners + edge centers + centroid pattern
        np.array([
            [0.01, 0.01], [0.99, 0.01], [0.01, 0.99],
            [0.50, 0.01], [0.01, 0.50], [0.50, 0.50],
            [0.33, 0.33], [0.20, 0.60], [0.60, 0.20],
            [0.40, 0.10], [0.10, 0.40],
        ]),
        # Alternative symmetric
        np.array([
            [0.02, 0.02], [0.98, 0.02], [0.02, 0.98],
            [0.49, 0.02], [0.02, 0.49], [0.49, 0.49],
            [0.33, 0.33], [0.25, 0.55], [0.55, 0.25],
            [0.45, 0.15], [0.15, 0.45],
        ]),
        # Grid-like
        np.array([
            [0.05, 0.05], [0.95, 0.05], [0.05, 0.95],
            [0.50, 0.05], [0.05, 0.50], [0.50, 0.50],
            [0.33, 0.33], [0.25, 0.60], [0.60, 0.25],
            [0.40, 0.15], [0.15, 0.40],
        ]),
    ]
    
    for bary in bary_sets:
        cart = _bary_to_cart(bary)
        cart = _clip_to_triangle(cart)
        inits.append(cart.flatten())
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(0, 1), (0, 1)] * N  # barycentric bounds
            de_res = differential_evolution(
                _objective_bary,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                init='sobol',
                polish=False,
                workers=1
            )
            
            if de_res.fun < 0:
                bary_opt = de_res.x.reshape(N, 2)
                # Clip to valid barycentric
                b = np.clip(bary_opt[:, 0], 0, 1)
                c = np.clip(bary_opt[:, 1], 0, 1)
                excess = np.maximum(b + c - 1, 0)
                b = b - 0.5 * excess
                c = c - 0.5 * excess
                b = np.clip(b, 0, 1)
                c = np.clip(c, 0, 1)
                
                points = _bary_to_cart(np.column_stack([b, c]))
                points = _clip_to_triangle(points)
                
                if _is_inside_triangle(points, eps=1e-5):
                    score = _min_triangle_area_fast(points)
                    if score > best_score:
                        best_score = score
                        best_points = points
        except Exception:
            pass
    
    # Phase 2: Local refinement from structured inits
    inits = _generate_structured_inits()
    
    for init_flat in inits:
        if time.time() > deadline - 40:
            break
        
        try:
            init_points = init_flat.reshape(N, 2)
            init_points = _clip_to_triangle(init_points)
            
            if not _is_inside_triangle(init_points, eps=1e-5):
                continue
            
            refined = _refine_local(init_points, max_iter=300)
            if _is_inside_triangle(refined, eps=1e-5):
                score = _min_triangle_area_fast(refined)
                if score > best_score:
                    best_score = score
                    best_points = refined
        except Exception:
            continue
    
    # Phase 3: Refine best point further
    if best_points is not None and time.time() < deadline - 20:
        try:
            refined = _refine_local(best_points, max_iter=500)
            if _is_inside_triangle(refined, eps=1e-5):
                score = _min_triangle_area_fast(refined)
                if score > best_score:
                    best_score = score
                    best_points = refined
        except Exception:
            pass
    
    # Phase 4: Multi-start refinement near best
    if best_points is not None and time.time() < deadline - 10:
        for i in range(8):
            if time.time() > deadline - 5:
                break
            try:
                rng_i = np.random.RandomState(42 + i * 100)
                for scale in [0.05, 0.03, 0.01]:
                    perturbed = best_points + rng_i.randn(N, 2) * scale
                    perturbed = _clip_to_triangle(perturbed)
                    
                    if not _is_inside_triangle(perturbed, eps=1e-5):
                        continue
                    
                    refined = _refine_local(perturbed, max_iter=200)
                    if _is_inside_triangle(refined, eps=1e-5):
                        score = _min_triangle_area_fast(refined)
                        if score > best_score:
                            best_score = score
                            best_points = refined
            except Exception:
                continue
    
    # Final fallback
    if best_points is None or not _is_inside_triangle(best_points, eps=1e-3):
        fallback_bary = np.array([
            [0.02, 0.02], [0.98, 0.02], [0.02, 0.98],
            [0.49, 0.02], [0.02, 0.49], [0.49, 0.49],
            [0.33, 0.33], [0.25, 0.55], [0.55, 0.25],
            [0.45, 0.15], [0.15, 0.45],
        ])
        best_points = _bary_to_cart(fallback_bary)
    
    best_points = _clip_to_triangle(best_points)
    if not _is_inside_triangle(best_points, eps=1e-3):
        best_points = np.array([
            [0.01, 0.01],
            [0.99, 0.01],
            [0.5, H * 0.99],
            [0.25, 0.01],
            [0.75, 0.01],
            [0.5, H * 0.5],
            [0.5, H * 0.25],
            [0.3, H * 0.3],
            [0.7, H * 0.3],
            [0.5, H * 0.7],
            [0.5, H * 0.1],
        ])
    
    return best_points
# EVOLVE-BLOCK-END
