"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull

def _triangle_area(p1, p2, p3):
    """Compute area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def _all_triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas.append((i, j, k, _triangle_area(points[i], points[j], points[k])))
    return areas

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    areas = _all_triangle_areas(points)
    return min(a[3] for a in areas)

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _newton_solve(points, active_tris, max_iter=100, tol=1e-8):
    """
    Solve the equi-area system using damped Newton-Raphson.
    
    Unknowns: 2*N point coordinates + 1 target area A = 2*N+1 unknowns
    Equations: 
      - For each active triangle (i,j,k): area(i,j,k) - A = 0
      - Convex hull area - 1 = 0
    
    We use a subset of active triangles (those with smallest areas) to keep the system square or overdetermined.
    """
    n = len(points)
    x = points.flatten().copy()
    
    # Start with A = current minimum area
    areas = _all_triangle_areas(points)
    sorted_areas = sorted(areas, key=lambda t: t[3])
    current_min = sorted_areas[0][3]
    A = current_min
    
    # Select active triangles: those with area close to the minimum
    # We need 2*N+1 equations total
    num_equations_needed = 2 * n + 1
    active = []
    for tri in sorted_areas:
        if len(active) < num_equations_needed:
            active.append(tri)
    
    # If we don't have enough distinct triangles, use what we have
    # and add hull constraint
    
    def compute_residual(x_flat, A_val):
        """Compute the residual vector for the Newton system."""
        pts = x_flat.reshape(-1, 2)
        res = []
        
        # Equations for active triangles: area - A = 0
        for (i, j, k, _) in active:
            area = _triangle_area(pts[i], pts[j], pts[k])
            res.append(area - A_val)
        
        # Hull area equation: hull_area - 1 = 0
        hull_area = _convex_hull_area(pts)
        res.append(hull_area - 1.0)
        
        return np.array(res)
    
    def compute_jacobian(x_flat, A_val):
        """Compute the Jacobian of the residual system."""
        pts = x_flat.reshape(-1, 2)
        num_res = len(active) + 1
        J = np.zeros((num_res, 2*n + 1))
        
        # Jacobian of triangle area w.r.t. point coordinates
        for eq_idx, (i, j, k, _) in enumerate(active):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            # d(area)/d(p1) = 0.5 * (-(p3[1]-p2[1]), (p3[0]-p2[0]))
            # d(area)/d(p2) = 0.5 * ((p3[1]-p1[1]), -(p3[0]-p1[0]))
            # d(area)/d(p3) = 0.5 * (-(p2[1]-p1[1]), (p2[0]-p1[0]))
            
            da_dp1 = 0.5 * np.array([-(p3[1]-p2[1]), (p3[0]-p2[0])])
            da_dp2 = 0.5 * np.array([(p3[1]-p1[1]), -(p3[0]-p1[0])])
            da_dp3 = 0.5 * np.array([-(p2[1]-p1[1]), (p2[0]-p1[0])])
            
            J[eq_idx, 2*i:2*i+2] = da_dp1
            J[eq_idx, 2*j:2*j+2] = da_dp2
            J[eq_idx, 2*k:2*k+2] = da_dp3
            
            # d(area)/dA = -1
            J[eq_idx, 2*n] = -1.0
        
        # Jacobian of hull area: approximate using finite differences
        # This is expensive but necessary
        eps = 1e-7
        for col in range(2*n + 1):
            x_plus = x_flat.copy()
            x_plus[col] += eps
            if col < 2*n:
                # Perturb a point coordinate
                pts_plus = x_plus.reshape(-1, 2)
            else:
                # Perturb A - but hull area doesn't depend on A directly
                pts_plus = x_plus.reshape(-1, 2)
            
            hull_plus = _convex_hull_area(pts_plus)
            hull_minus_pts = x_flat.copy()
            hull_minus_pts[col] -= eps
            if col < 2*n:
                pts_minus = hull_minus_pts.reshape(-1, 2)
            else:
                pts_minus = hull_minus_pts.reshape(-1, 2)
            hull_minus = _convex_hull_area(pts_minus)
            
            J[num_res-1, col] = (hull_plus - hull_minus) / (2*eps)
        
        return J
    
    # Newton iteration
    for it in range(max_iter):
        x_full = np.concatenate([x, [A]])
        r = compute_residual(x, A)
        
        if np.linalg.norm(r) < tol:
            break
        
        J = compute_jacobian(x, A)
        
        # Solve J * dx = -r
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            # Use least squares if singular
            dx, residuals, rank, s = np.linalg.lstsq(J, -r, rcond=None)
        
        # Damping: take a small step if needed
        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * dx[:2*n]
            A_new = A + alpha * dx[2*n]
            x_new = np.clip(x_new, -10, 10)
            
            r_new = compute_residual(x_new, A_new)
            if np.linalg.norm(r_new) < np.linalg.norm(r):
                break
            alpha *= 0.5
        
        x = x_new
        A = A_new
    
    return x.reshape(-1, 2), A

def construct():
    """
    Place 13 points using Newton-Raphson to solve the equi-area system.
    """
    best_score = 0.0
    best_points = None
    
    # Multiple initial configurations
    rng = np.random.RandomState(42)
    initial_configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(reg_points.copy())
    
    # 2. Perturbed regular polygon
    for seed in [1, 2, 3]:
        rng2 = np.random.RandomState(seed)
        perturbed = reg_points.copy()
        for i in range(N):
            perturbed[i] += rng2.rand(2) * 0.2
        initial_configs.append(perturbed)
    
    # 3. Grid-like
    grid_points = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_points.append([i/3.0, j/3.0])
    grid_points = np.array(grid_points)
    initial_configs.append(grid_points)
    
    # 4. Random
    for seed in [10, 20, 30]:
        rng3 = np.random.RandomState(seed)
        rand_pts = rng3.rand(N, 2) * 2 - 1
        initial_configs.append(rand_pts)
    
    for init_points in initial_configs:
        # Normalize to unit hull area
        hull_area = _convex_hull_area(init_points)
        if hull_area < 1e-10:
            continue
        init_points = init_points / np.sqrt(hull_area)
        
        # Run Newton solver
        try:
            solved_points, A_target = _newton_solve(init_points, None, max_iter=50, tol=1e-6)
            
            # Verify and score
            hull_area = _convex_hull_area(solved_points)
            if hull_area < 1e-10:
                continue
            
            min_area = _min_triangle_area(solved_points)
            score = min_area / hull_area
            
            if score > best_score:
                best_score = score
                best_points = solved_points.copy()
        except Exception:
            pass
    
    # Fallback: use best initial config if Newton didn't work well
    if best_points is None or best_score < 0.01:
        best_points = reg_points
        hull_area = _convex_hull_area(best_points)
        if hull_area > 1e-10:
            best_points = best_points / np.sqrt(hull_area)
        best_score = _score(best_points)
    
    # Final normalization
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        best_points = best_points / np.sqrt(hull_area)
    
    return best_points
# EVOLVE-BLOCK-END
