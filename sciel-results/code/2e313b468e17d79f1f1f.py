"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

_start = time.time()
DEADLINE = 150.0
SAFETY = 5.0  # seconds before wall-clock limit

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle


def point_in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the triangle."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                 - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area


def objective_flat(x):
    """Negative of min triangle area (to minimize), with penalty for out-of-triangle points."""
    pts = x.reshape(-1, 2)
    # Check if any point is outside the triangle
    for p in pts:
        if not point_in_triangle(p, eps=-1e-6):
            # Penalize heavily for being outside
            return 1e6
    return -min_triangle_area(pts)


def project_into_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = pts.copy()
    for i in range(len(pts)):
        p = pts[i]
        if not point_in_triangle(p, eps=-1e-8):
            # Find closest point on triangle boundary
            # Use barycentric coordinate approach: project to nearest edge
            # Simple approach: scale toward centroid
            centroid = (V0 + V1 + V2) / 3.0
            for t in np.linspace(0.99, 0.5, 50):
                q = p * t + centroid * (1 - t)
                if point_in_triangle(q, eps=-1e-9):
                    pts[i] = q
                    break
            else:
                pts[i] = centroid
    return pts


def nelder_mead(x0, maxiter=2000, tol=1e-10):
    """Simple Nelder-Mead implementation with triangle constraint handling."""
    from scipy.optimize import minimize
    result = minimize(objective_flat, x0, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol,
                               'adaptive': True})
    pts = result.x.reshape(-1, 2)
    pts = project_into_triangle(pts)
    return pts, -result.fun


def construct():
    global _start
    # Structured grid initialization: use symmetric positions
    # Best known configurations for 11 points in equilateral triangle
    # Use several structured starting points
    
    best_pts = None
    best_val = 0.0
    
    # Initial structured grid: points on a triangular lattice
    # For 11 points, use a 3-3-3-2 arrangement or similar
    # Let's create several candidate initial configurations
    
    candidates = []
    
    # Candidate 1: Triangular lattice with some perturbation
    # Rows: 3, 3, 3, 2 points
    h = np.sqrt(3) / 2
    # Row 0 (bottom): 3 points
    row0 = np.array([[0.15, 0.1], [0.5, 0.1], [0.85, 0.1]])
    # Row 1: 3 points
    row1 = np.array([[0.3, 0.3], [0.5, 0.3], [0.7, 0.3]])
    # Row 2: 3 points
    row2 = np.array([[0.4, 0.5], [0.5, 0.5], [0.6, 0.5]])
    # Row 3 (top): 2 points
    row3 = np.array([[0.45, 0.65], [0.55, 0.65]])
    cand1 = np.vstack([row0, row1, row2, row3])
    candidates.append(cand1)
    
    # Candidate 2: More spread out
    cand2 = np.array([
        [0.1, 0.05], [0.5, 0.05], [0.9, 0.05],
        [0.25, 0.25], [0.5, 0.25], [0.75, 0.25],
        [0.35, 0.45], [0.5, 0.45], [0.65, 0.45],
        [0.42, 0.6], [0.58, 0.6]
    ])
    candidates.append(cand2)
    
    # Candidate 3: Hexagonal-like pattern
    cand3 = np.array([
        [0.5, 0.1],
        [0.2, 0.2], [0.8, 0.2],
        [0.35, 0.4], [0.65, 0.4],
        [0.5, 0.5],
        [0.3, 0.6], [0.7, 0.6],
        [0.4, 0.75], [0.6, 0.75],
        [0.5, 0.85]
    ])
    candidates.append(cand3)
    
    # Candidate 4: Symmetric with center
    cand4 = np.array([
        [0.5, 0.08],
        [0.15, 0.15], [0.85, 0.15],
        [0.25, 0.35], [0.5, 0.35], [0.75, 0.35],
        [0.35, 0.55], [0.5, 0.55], [0.65, 0.55],
        [0.42, 0.72], [0.58, 0.72]
    ])
    candidates.append(cand4)
    
    # Candidate 5: Uniform-ish grid
    xs = np.array([0.2, 0.5, 0.8])
    ys = np.array([0.1, 0.3, 0.5, 0.7])
    cand5 = []
    for y in ys:
        for x in xs:
            if point_in_triangle([x, y], eps=-1e-6):
                cand5.append([x, y])
    # Take first 11
    cand5 = np.array(cand5[:11]) if len(cand5) >= 11 else np.array(cand5)
    if len(cand5) == 11:
        candidates.append(cand5)
    
    # Add random perturbations to each candidate
    rng = np.random.RandomState(42)
    all_starts = []
    for cand in candidates:
        if len(cand) == 11:
            all_starts.append(cand.copy())
            # Add perturbed versions
            for _ in range(3):
                perturbed = cand + rng.randn(11, 2) * 0.03
                perturbed = project_into_triangle(perturbed)
                all_starts.append(perturbed)
    
    # Also add purely random starts
    for _ in range(5):
        rand_pts = rng.rand(11, 2)
        # Map to triangle
        mapped = []
        for i in range(11):
            u, v = rand_pts[i]
            if u + v < 1.0:
                mapped.append([u + 0.5 * v, h * v])
        if len(mapped) == 11:
            all_starts.append(np.array(mapped))
    
    # Evaluate all starts
    for start in all_starts:
        if time.time() - _start > DEADLINE - SAFETY:
            break
        val = min_triangle_area(start)
        if val > best_val:
            best_val = val
            best_pts = start.copy()
    
    # Multi-start Nelder-Mead refinement
    for start in all_starts:
        if time.time() - _start > DEADLINE - SAFETY:
            break
        try:
            refined, val = nelder_mead(start.flatten(), maxiter=1500, tol=1e-10)
            if val > best_val:
                best_val = val
                best_pts = refined.copy()
        except Exception:
            pass
    
    # Final high-precision polish on the best solution
    if best_pts is not None and time.time() - _start < DEADLINE - SAFETY:
        try:
            polished, val = nelder_mead(best_pts.flatten(), maxiter=3000, tol=1e-12)
            if val > best_val:
                best_val = val
                best_pts = polished.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return the first candidate
        best_pts = candidates[0]
    
    # Final safety check
    best_pts = project_into_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
