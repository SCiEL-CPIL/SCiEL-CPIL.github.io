"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import numpy as np
import time
import itertools

N = 14


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = np.inf
    # Vectorized: compute cross products for all triples
    # Use the formula: area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            # Compute cross with all k > j
            v2s = pts[j + 1:] - pts[i]
            crosses = v2s[:, 0] * v1[1] - v2s[:, 1] * v1[0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Find convex hull vertices in order
    # Use monotone chain algorithm
    pts_sorted = np.lexsort((pts[:, 1], pts[:, 0]))
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    
    # Shoelace formula
    m = len(hull)
    if m < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return area


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_area


def _is_convex(pts):
    """Check if points form a strictly convex polygon (all cross products same sign)."""
    n = len(pts)
    signs = []
    for i in range(n):
        o = pts[i]
        a = pts[(i + 1) % n]
        b = pts[(i + 2) % n]
        cr = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        signs.append(np.sign(cr))
    # All should be same sign and non-zero
    if any(s == 0 for s in signs):
        return False
    return all(s == signs[0] for s in signs)


def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for the 120s limit
    
    # Strategy: Start with regular 14-gon, then use Nelder-Mead on 
    # angular and radial perturbations to maximize min_triangle_area / hull_area
    
    # Initial: regular 14-gon
    angles0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Parameterize: each point has angle = angles0[i] + dtheta[i], radius = 1 + dr[i]
    # Perturbations constrained to maintain convexity
    # We'll optimize 2N parameters: dtheta[0..N-1], dr[0..N-1]
    
    def params_to_points(params):
        dtheta = params[:N]
        dr = params[N:]
        angles = angles0 + dtheta
        radii = 1.0 + dr
        # Clamp radii to keep positive
        radii = np.maximum(radii, 0.1)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts
    
    def objective(params):
        pts = params_to_points(params)
        if not _is_convex(pts):
            return -1e10  # Penalty for non-convex
        return -_score(pts)  # Minimize negative score
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    # Initial perturbation: small random
    np.random.seed(42)
    x0 = np.zeros(2 * N)
    x0[:N] = np.random.randn(N) * 0.01
    x0[N:] = np.random.randn(N) * 0.01
    
    best_score = _score(params_to_points(x0))
    best_params = x0.copy()
    
    # Try multiple restarts with different initial perturbations
    n_restarts = 5
    restart_times = [0] + list(np.linspace(0, time_limit * 0.9, n_restarts)[1:])
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize
        if r == 0:
            x0 = np.zeros(2 * N)
        else:
            x0 = np.random.randn(2 * N) * 0.15
        
        # Nelder-Mead
        try:
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8,
                         'adaptive': True}
            )
            if result.fun < -best_score + 1e-15:
                best_score = -result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() - start_time > time_limit * 0.85:
            break
    
    # Local refinement: try to move points to reduce hull area while maintaining min triangle area
    # This is the key insight: moving points inward reduces hull area, improving the ratio
    # We do a coordinate descent on individual point positions
    
    best_pts = params_to_points(best_params)
    current_score = best_score
    
    # Refinement phase: for each point, try small moves in x and y
    max_refine_iters = 50
    for it in range(max_refine_iters):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            # Try small perturbations in x and y
            for dx in [-0.02, -0.01, 0.01, 0.02]:
                for dy in [-0.02, -0.01, 0.01, 0.02]:
                    trial = best_pts.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    if _is_convex(trial):
                        s = _score(trial)
                        if s > current_score + 1e-12:
                            current_score = s
                            best_pts = trial
                            improved = True
        if not improved:
            break
    
    return best_pts


# Quick self-test
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
