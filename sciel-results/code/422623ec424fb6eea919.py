"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

N = 11
# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
# Area of the triangle
TRI_AREA = np.sqrt(3) / 4


def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points (vectorized)."""
    n = len(points)
    # Compute all pairwise differences
    # diff[i, j] = points[j] - points[i]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # Cross product z-component for all pairs
    # cross[i, j] = diff[i, j, 0] * diff[i, j, 1] ... wait, cross product of (p_j - p_i) and (p_k - p_i)
    # It is easier to compute area directly for all triples using broadcasting or loops over indices if N is small.
    # For N=11, 165 triples. Vectorizing the whole 3D array is overkill and memory heavy.
    # Instead, we can just compute all areas in a flat vectorized way.
    
    # Indices for i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    
    return np.min(areas)


def _project_to_triangle(p):
    """Project point p to the closest point inside the triangle."""
    # Convert to barycentric coordinates
    # Solve: p = w0*V0 + w1*V1 + w2*V2, w0+w1+w2=1, wi>=0
    # Using the fact that for equilateral triangle we can use distances to sides
    
    # Distance from point to each side (using cross product)
    # Side 0: V1->V2, Side 1: V2->V0, Side 2: V0->V1
    
    def dist_to_side(p, a, b):
        # Signed distance from p to line through a,b
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / np.sqrt(dx * dx + dy * dy)
    
    # For the triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    # Side 0: (1,0) to (0.5, sqrt(3)/2) - right side
    # Side 1: (0.5, sqrt(3)/2) to (0,0) - left side
    # Side 2: (0,0) to (1,0) - bottom side
    
    # Let's use a simpler approach: check if point is inside, if not project to nearest edge or vertex
    
    # Barycentric coordinates
    d0 = (V[1][0] - V[0][0]) * (V[2][1] - V[0][1]) - (V[1][1] - V[0][1]) * (V[2][0] - V[0][0])
    
    w0 = ((V[1][0] - V[2][0]) * (p[1] - V[2][1]) - (V[1][1] - V[2][1]) * (p[0] - V[2][0])) / d0
    w1 = ((V[2][0] - V[0][0]) * (p[1] - V[0][1]) - (V[2][1] - V[0][1]) * (p[0] - V[0][0])) / d0
    w2 = 1 - w0 - w1
    
    if w0 >= 0 and w1 >= 0 and w2 >= 0:
        return p
    
    # Point is outside, project to nearest point on triangle
    # Check each edge
    best_p = None
    best_dist = float('inf')
    
    for i in range(3):
        a = V[i]
        b = V[(i + 1) % 3]
        # Project p onto line segment ab
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best_p = proj
    
    return best_p


def _objective_logsumexp(x, beta=50.0):
    """
    Objective: minimize negative log-sum-exp of triangle areas.
    x is a flat array of shape (2*N,).
    """
    points = x.reshape(N, 2)
    n = N
    
    # Vectorized area calculation
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-12)
    
    # Log-sum-exp: log(sum(exp(-beta * area)))
    # We want to maximize the minimum area, so we minimize a smooth approximation
    # Using stable log-sum-exp
    max_neg = -np.max(areas)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(-beta * areas - max_neg)))
    
    # We want to minimize this, so return it
    return log_sum_exp


def _objective_exact(x):
    """Objective: minimize negative of minimum triangle area."""
    points = x.reshape(N, 2)
    min_a = _min_triangle_area(points)
    return -min_a


def _constraints(x):
    """Ensure points are inside the triangle."""
    points = x.reshape(N, 2)
    # For each point, check barycentric coordinates
    for i in range(N):
        p = points[i]
        # Distance to each side should be non-negative (inside)
        # Side 0: line through V[1] and V[2]
        # Side 1: line through V[2] and V[0]
        # Side 2: line through V[0] and V[1]
        
        # Using cross product method
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        c0 = cross(V[1], V[2], p)
        c1 = cross(V[2], V[0], p)
        c2 = cross(V[0], V[1], p)
        
        # For the triangle oriented counterclockwise, all cross products should be non-negative
        # V[0]=(0,0), V[1]=(1,0), V[2]=(0.5, sqrt(3)/2)
        # cross(V[1], V[2], p) = (V[2][0]-V[1][0])*(p[1]-V[1][1]) - (V[2][1]-V[1][1])*(p[0]-V[1][0])
        # = (-0.5)*(p[1]) - (sqrt(3)/2)*(p[0]-1)
        # = -0.5*p[1] - sqrt(3)/2*p[0] + sqrt(3)/2
        # This should be >= 0 for points inside
        
        # Let's just check if point is inside using barycentric
        d0 = (V[1][0] - V[0][0]) * (V[2][1] - V[0][1]) - (V[1][1] - V[0][1]) * (V[2][0] - V[0][0])
        w0 = ((V[1][0] - V[2][0]) * (p[1] - V[2][1]) - (V[1][1] - V[2][1]) * (p[0] - V[2][0])) / d0
        w1 = ((V[2][0] - V[0][0]) * (p[1] - V[0][1]) - (V[2][1] - V[0][1]) * (p[0] - V[0][0])) / d0
        w2 = 1 - w0 - w1
        
        if w0 < 0 or w1 < 0 or w2 < 0:
            return False
    
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds, leaving 10s margin for safety
    
    np.random.seed(42)
    best_points = None
    best_score = -float('inf')
    
    # Define bounds for Differential Evolution
    # Points must be inside the triangle.
    # x in [0, 1], y in [0, sqrt(3)/2]
    # But we need to ensure they are inside.
    # DE with constraints is tricky. We can use bounds and then project, or use a penalty.
    # However, DE works best with box constraints.
    # Let's use the box [0, 1] x [0, 0.866] and project after each evaluation?
    # No, DE doesn't call a projection callback easily.
    # Instead, we can map DE's output to the triangle using the random point generation method,
    # but DE expects a continuous space.
    
    # Alternative: Use DE in barycentric coordinates? No, sum=1 constraint is hard for DE.
    
    # Best approach for DE:
    # Define bounds as the bounding box of the triangle.
    # Inside the objective function, we check if points are inside.
    # If not, we penalize heavily or project.
    # Projecting inside the objective is robust.
    
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    def objective_de(x):
        points = x.reshape(N, 2)
        # Project to triangle to ensure validity
        for i in range(N):
            points[i] = _project_to_triangle(points[i])
        
        # Use log-sum-exp for smoothness during DE
        return _objective_logsumexp(points.flatten(), beta=20.0)
    
    try:
        # Run Differential Evolution
        # maxiter is number of generations. popsize is factor * N
        # We limit time by checking in the callback or just setting reasonable params.
        # DE can be slow. Let's set a moderate popsize and maxiter.
        res = minimize(
            objective_de,
            np.random.uniform(0, 1, 2*N),
            method='differential_evolution',
            bounds=bounds,
            maxiter=100,
            popsize=10,
            tol=1e-6,
            seed=42,
            polish=False # We will polish manually
        )
        
        if res.success or res.x is not None:
            opt_points = res.x.reshape(N, 2)
            for i in range(N):
                opt_points[i] = _project_to_triangle(opt_points[i])
            
            score = _min_triangle_area(opt_points)
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
        
    except Exception:
        pass
    
    # If DE didn't find anything or failed, use a random fallback
    if best_points is None:
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = np.random.random()
                v = np.random.random()
                if u + v <= 1:
                    points[i] = u * V[1] + v * V[2]
                    break
        best_points = points
        best_score = _min_triangle_area(points)
    
    # Refine with Nelder-Mead on the exact objective
    if time.time() < deadline - 10:
        try:
            # Nelder-Mead on the exact objective (maximizing min area)
            # We need to minimize -min_area
            def objective_nm(x):
                points = x.reshape(N, 2)
                # Project to keep inside
                for i in range(N):
                    points[i] = _project_to_triangle(points[i])
                return -_min_triangle_area(points)
            
            res2 = minimize(
                objective_nm,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            nm_points = res2.x.reshape(N, 2)
            for i in range(N):
                nm_points[i] = _project_to_triangle(nm_points[i])
            
            nm_score = _min_triangle_area(nm_points)
            
            if nm_score > best_score:
                best_points = nm_points.copy()
                best_score = nm_score
        
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-START
# (The construct function above is the entry point)
# EVOLVE-BLOCK-END
