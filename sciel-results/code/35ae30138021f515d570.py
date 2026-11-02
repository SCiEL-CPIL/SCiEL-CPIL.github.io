"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0

    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas using cross product
    # For points i, j, k: area = 0.5 * |cross(pj-pi, pk-pi)|
    min_area = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area

    return min_area / hull_area


def _params_to_points(params):
    """
    Convert symmetric ring parameters to 13 points.
    
    Structure: 1 center + 6 on ring 1 + 6 on ring 2
    params = [r1, r2, phase1, phase2]
    """
    r1, r2, phase1, phase2 = params
    
    pts = []
    
    # Center point
    pts.append([0.0, 0.0])
    
    # Ring 1: 6 points
    for i in range(6):
        angle = phase1 + i * np.pi / 3
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 6 points
    for i in range(6):
        angle = phase2 + i * np.pi / 3
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    return np.array(pts)


def _neg_score_params(params):
    """Negative score for minimization."""
    pts = _params_to_points(params)
    return -_score_points(pts)


def _score_params(params):
    """Score for maximization."""
    pts = _params_to_points(params)
    return _score_points(pts)


def _optimize_ring_structure(deadline):
    """Optimize the 1+6+6 ring structure."""
    # Bounds for parameters
    # r1, r2: radial distances (0 to ~1.5)
    # phase1, phase2: angular offsets (0 to 2*pi)
    bounds = [(0.05, 1.5), (0.05, 1.5), (0.0, 2 * np.pi), (0.0, 2 * np.pi)]
    
    # Multi-start with different initial guesses
    best_score = 0.0
    best_params = None
    best_pts = None
    
    # Generate diverse initial points
    np.random.seed(42)
    initial_points = []
    
    # Some structured starts
    initial_points.append([0.5, 1.0, 0.0, 0.0])
    initial_points.append([0.5, 1.0, 0.0, np.pi / 6])
    initial_points.append([0.7, 1.2, 0.0, 0.0])
    initial_points.append([0.3, 0.8, 0.0, np.pi / 3])
    initial_points.append([0.6, 1.1, np.pi / 12, np.pi / 12])
    
    # Random starts
    for _ in range(20):
        p = [np.random.uniform(0.1, 1.2), np.random.uniform(0.1, 1.5),
             np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)]
        initial_points.append(p)
    
    for init in initial_points:
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(_neg_score_params, init, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            score = _score_params(result.x)
            if score > best_score:
                best_score = score
                best_params = result.x
                best_pts = _params_to_points(result.x)
        except Exception:
            continue
    
    return best_score, best_pts


def _optimize_hexagon_structure(deadline):
    """
    Alternative: 7 on hexagon vertices + 6 interior points on symmetry axes.
    Or: 1 center + 12 on two hexagonal rings with different phase.
    """
    # Try 1 + 12 structure: center + 12 points on a single ring
    # But 12 points on a ring might be too regular. Let's try 1 + 6 + 6 with different radii.
    # Actually, let's try a different decomposition: 7 on outer hexagon + 6 on inner hexagon
    
    def _params_to_points_hex(params):
        r_outer, r_inner, phase_outer, phase_inner = params
        
        pts = []
        
        # Outer hexagon: 7 points (6 vertices + 1 center? No, let's do 6 vertices)
        # Actually, let's do 6 outer + 6 inner + 1 center = 13
        pts.append([0.0, 0.0])
        
        for i in range(6):
            angle = phase_outer + i * np.pi / 3
            pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
        
        for i in range(6):
            angle = phase_inner + i * np.pi / 3
            pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
        
        return np.array(pts)
    
    def _neg_score_hex(params):
        pts = _params_to_points_hex(params)
        return -_score_points(pts)
    
    bounds = [(0.05, 1.5), (0.05, 1.5), (0.0, 2 * np.pi), (0.0, 2 * np.pi)]
    
    best_score = 0.0
    best_pts = None
    
    np.random.seed(123)
    initial_points = []
    
    # Structured starts
    initial_points.append([1.0, 0.5, 0.0, 0.0])
    initial_points.append([1.0, 0.5, 0.0, np.pi / 6])
    initial_points.append([1.2, 0.7, 0.0, np.pi / 3])
    
    # Random starts
    for _ in range(15):
        p = [np.random.uniform(0.5, 1.5), np.random.uniform(0.1, 1.0),
             np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)]
        initial_points.append(p)
    
    for init in initial_points:
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(_neg_score_hex, init, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            score = _score_points(_params_to_points_hex(result.x))
            if score > best_score:
                best_score = score
                best_pts = _params_to_points_hex(result.x)
        except Exception:
            continue
    
    return best_score, best_pts


def _try_de_global_search(deadline):
    """Use differential evolution for global search over ring parameters."""
    def _neg_score_de(params):
        pts = _params_to_points(params)
        return -_score_points(pts)
    
    bounds = [(0.05, 1.5), (0.05, 1.5), (0.0, 2 * np.pi), (0.0, 2 * np.pi)]
    
    try:
        result = differential_evolution(_neg_score_de, bounds, maxiter=100,
                                        popsize=20, seed=42, tol=1e-6,
                                        polish=True)
        score = _score_points(_params_to_points(result.x))
        return score, _params_to_points(result.x)
    except Exception:
        return 0.0, None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Strategy 1: Optimize 1+6+6 ring structure
    score1, pts1 = _optimize_ring_structure(deadline)
    if pts1 is not None and score1 > best_score:
        best_score = score1
        best_pts = pts1
    
    if time.time() > deadline - 10:
        if best_pts is not None:
            return best_pts
        # Fallback: regular hexagon + center
        pts = np.array([[0.0, 0.0]])
        for i in range(12):
            angle = i * np.pi / 6
            r = 1.0 if i % 2 == 0 else 0.5
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        return pts
    
    # Strategy 2: Try differential evolution for global search
    score2, pts2 = _try_de_global_search(deadline)
    if pts2 is not None and score2 > best_score:
        best_score = score2
        best_pts = pts2
    
    # Strategy 3: Try alternative hexagon structure
    if time.time() < deadline - 10:
        score3, pts3 = _optimize_hexagon_structure(deadline)
        if pts3 is not None and score3 > best_score:
            best_score = score3
            best_pts = pts3
    
    # Fallback if nothing worked
    if best_pts is None:
        pts = np.array([[0.0, 0.0]])
        for i in range(12):
            angle = i * np.pi / 6
            r = 1.0 if i % 2 == 0 else 0.5
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        best_pts = pts
    
    return best_pts
# EVOLVE-BLOCK-END
