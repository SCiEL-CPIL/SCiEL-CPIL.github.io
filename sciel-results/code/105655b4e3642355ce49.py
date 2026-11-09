import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def get_points(deltas):
    """Generate 20 points on unit circle with cyclic order preserved.
    deltas: 19 values added to angles 1..19 (angle 0 is fixed at 0)."""
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    thetas = base_angles.copy()
    if len(deltas) > 0:
        thetas[1:] += deltas
    # Wrap to [0, 2*pi) and sort to maintain cyclic order
    thetas = thetas % (2*np.pi)
    thetas = np.sort(thetas)
    x = np.cos(thetas)
    y = np.sin(thetas)
    return np.column_stack((x, y))

def normalize_polygon(points):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def check_convexity(points):
    """Check if points form a strictly convex polygon."""
    n = len(points)
    if n < 3:
        return False
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-12:
                return False
    # Check that all turns are nonzero and of one sign
    cross_products = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i+1) % n]
        p2 = points[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_products.append(cross)
    
    # All cross products must be nonzero
    if any(abs(c) < 1e-12 for c in cross_products):
        return False
    # All must have the same sign
    signs = [np.sign(c) for c in cross_products]
    if len(set(signs)) != 1:
        return False
    return True

def vertex_score(v_idx, points):
    """Compute the score for a single vertex."""
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    
    if sorted_dists[0] < 0.05:
        return 0.0
    
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
    
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def soft_min(scores, beta=50.0):
    """Soft minimum for smooth optimization."""
    return -np.log(np.sum(np.exp(-beta * scores))) / beta

def smooth_objective(deltas):
    """Smooth objective using soft-min for gradient-based optimization."""
    points = get_points(deltas)
    normalized = normalize_polygon(points)
    if normalized is None:
        return 10.0
    if not check_convexity(normalized):
        return 10.0
    n = 20
    scores = np.array([vertex_score(i, normalized) for i in range(n)])
    return soft_min(scores, beta=50.0)

def exact_objective(deltas):
    """Exact objective: negative of the minimum vertex score."""
    points = get_points(deltas)
    normalized = normalize_polygon(points)
    if normalized is None:
        return 10.0
    if not check_convexity(normalized):
        return 10.0
    n = 20
    scores = np.array([vertex_score(i, normalized) for i in range(n)])
    return -np.min(scores)

def reconstruct_deltas(points):
    """Reconstruct angular deltas from point coordinates."""
    thetas = np.arctan2(points[:, 1], points[:, 0])
    thetas = thetas % (2*np.pi)
    base_angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    return thetas[1:] - base_angles[1:]

def equidistant_points_polygon20() -> np.ndarray:
    """Main function: optimize the 20-gon vertex positions."""
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s limit
    
    n = 20
    x0 = np.zeros(19)
    best_result = get_points(x0)
    best_score = -exact_objective(x0)
    
    # Strategy: use L-BFGS-B with smooth objective, then refine with exact
    rng = np.random.default_rng(12345)
    
    # Phase 1: Smooth optimization from multiple random starts
    scales = [0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]
    
    for scale in scales:
        if time.time() > deadline - 60:
            break
        
        for trial in range(4):
            if time.time() > deadline - 60:
                break
            
            x_init = rng.normal(0, scale, size=19)
            
            try:
                # Use smooth objective for gradient-based optimization
                res = minimize(
                    smooth_objective,
                    x_init,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 40000}
                )
                if res.x is not None:
                    # Evaluate with exact objective
                    exact_val = exact_objective(res.x)
                    current_score = -exact_val
                    if current_score > best_score:
                        best_score = current_score
                        best_result = get_points(res.x)
            except Exception:
                pass
        
        # Also try starting from the best result found so far with a small perturbation
        if time.time() > deadline - 60:
            break
        try:
            x_init = reconstruct_deltas(best_result)
            x_init_perturbed = x_init + rng.normal(0, scale/10, size=19)
            
            res = minimize(
                smooth_objective,
                x_init_perturbed,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 30000}
            )
            if res.x is not None:
                exact_val = exact_objective(res.x)
                current_score = -exact_val
                if current_score > best_score:
                    best_score = current_score
                    best_result = get_points(res.x)
        except Exception:
            pass
    
    # Phase 2: Exact refinement from best solution
    if time.time() < deadline - 30:
        try:
            current_deltas = reconstruct_deltas(best_result)
            
            res = minimize(
                exact_objective,
                current_deltas,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 80000}
            )
            if res.x is not None:
                exact_val = exact_objective(res.x)
                current_score = -exact_val
                if current_score > best_score:
                    best_score = current_score
                    best_result = get_points(res.x)
        except Exception:
            pass
    
    # Phase 3: Fine-tuning with very small perturbations
    if time.time() < deadline - 15:
        try:
            current_deltas = reconstruct_deltas(best_result)
            
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                x_perturbed = current_deltas + rng.normal(0, 0.005, size=19)
                res = minimize(
                    exact_objective,
                    x_perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 50000}
                )
                if res.x is not None:
                    exact_val = exact_objective(res.x)
                    current_score = -exact_val
                    if current_score > best_score:
                        best_score = current_score
                        best_result = get_points(res.x)
                        current_deltas = res.x
        except Exception:
            pass
    
    # Ensure the result is normalized and convex
    normalized = normalize_polygon(best_result)
    if normalized is not None and check_convexity(normalized):
        return normalized
    else:
        # Fallback: return the regular polygon (which should be valid)
        regular = np.column_stack((np.cos(np.linspace(0, 2*np.pi, n, endpoint=False)),
                                   np.sin(np.linspace(0, 2*np.pi, n, endpoint=False))))
        return normalize_polygon(regular)
