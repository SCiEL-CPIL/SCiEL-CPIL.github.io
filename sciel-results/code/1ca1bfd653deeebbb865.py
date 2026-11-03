"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

N = 50
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s

def _covering_radius(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in the configuration.
    
    This is equivalent to finding the largest circumradius of the facets of the convex hull.
    """
    if not np.all(np.isfinite(pts)):
        return np.pi  # invalid
    if pts.shape != (N, 3):
        return np.pi
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-8:
                return np.pi
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere iff their convex hull does not contain the origin
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    # Check if origin is inside the convex hull
    # If the origin is inside, the points are NOT in a single hemisphere (good)
    # If the origin is outside, they ARE in a single hemisphere (bad)
    # We can check this by seeing if the origin is a vertex of the dual or by checking distances
    # A simpler check: if the minimum distance from origin to any facet is negative, origin is inside
    # Actually, let's use a different approach:
    # The points are in a hemisphere iff there exists a vector v such that v·p_i >= 0 for all i
    # This is equivalent to the origin being outside or on the boundary of the convex hull of the points
    
    # Compute the distance from origin to each facet
    # If all distances are positive, origin is outside (hemisphere)
    # If any distance is negative, origin is inside (not hemisphere)
    # Actually, for a convex polytope containing the origin, the distance from origin to each facet is positive
    # Let's check if the origin is inside the convex hull
    
    # Use the fact that origin is inside iff the convex hull has the origin in its interior
    # We can check this by solving for barycentric coordinates or using linear programming
    # Simpler: if the minimum singular value of the matrix of points is > 0 and the origin is inside
    
    # Alternative: compute the convex hull and check if origin is inside
    # The origin is inside the convex hull iff for every facet, the origin is on the same side as the interior
    # This is equivalent to: the signed distance from origin to each facet is <= 0 (assuming outward normals)
    
    # Let's use a robust method:
    # The points span R^3 (not in a hemisphere) iff the convex hull contains the origin
    # We can check this by seeing if the origin is a convex combination of the points
    # This is a linear programming problem: find lambda_i >= 0 such that sum lambda_i * p_i = 0, sum lambda_i = 1
    
    # For 50 points, we can use scipy's linear programming
    from scipy.optimize import linprog
    
    # Minimize 0 subject to:
    # P @ lambda = 0
    # sum(lambda) = 1
    # lambda >= 0
    
    A_eq = np.vstack([pts, np.ones(N)])
    b_eq = np.append(np.zeros(3), 1.0)
    
    res = linprog(c=np.zeros(N), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)]*N, method='highs')
    
    if res.status != 0:
        # Origin is not in the convex hull -> points are in a hemisphere -> invalid
        return np.pi
    
    # Now compute the covering radius
    # The covering radius is the maximum over all facets of the angular radius of the empty cap
    # For each triangular facet (p1, p2, p3), the circumcenter of the spherical triangle
    # gives the center of the empty cap, and the angular distance from this center to the vertices
    # is the cap radius
    
    # Get the facets from the convex hull
    max_radius = 0.0
    for simplex in hull.simplices:
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute the circumcenter of the spherical triangle
        # This is the normalized cross product of the normals to the great circles
        # Or: the point equidistant from p1, p2, p3 on the sphere
        
        # The circumcenter c satisfies c·p1 = c·p2 = c·p3 = cos(theta)
        # where theta is the angular radius
        
        # Solve for c: c is in the span of p1, p2, p3
        # c = a*p1 + b*p2 + d*p3
        # c·p1 = c·p2 = c·p3
        
        # Let M be the 3x3 matrix with rows p1, p2, p3
        # We want c such that M @ c = [cos(theta)] * 1
        # But c is also in the column space of M
        
        # Actually, the circumcenter is proportional to the vector that is equidistant from all three
        # c = (p1 x p2) x (p1 x p3) ... no, that's not right
        
        # The circumcenter of a spherical triangle is the normalization of:
        # c = (sin(A) * p1 + sin(B) * p2 + sin(C) * p3) / |...|
        # where A, B, C are the angles opposite to p1, p2, p3
        
        # Simpler: solve the linear system
        # Let c = alpha * p1 + beta * p2 + gamma * p3
        # c·p1 = c·p2 = c·p3
        # This gives:
        # alpha*(p1·p1) + beta*(p2·p1) + gamma*(p3·p1) = alpha*(p1·p2) + beta*(p2·p2) + gamma*(p3·p2)
        # etc.
        
        # Even simpler: the circumcenter is the normalized vector in the plane of the triangle
        # that is equidistant from all three vertices
        
        # Let's use the formula: the circumcenter c satisfies
        # |c - p1| = |c - p2| = |c - p3| (Euclidean distance, but since all on unit sphere, this is equivalent to angular distance)
        
        # Actually, on the unit sphere, the point equidistant from p1, p2, p3 is:
        # c = normalize(p1 + p2 + p3) is NOT generally correct
        
        # The correct formula: c is the normalization of the vector
        # v = (p2 - p1) x (p3 - p1) cross ... no
        
        # Let's solve it properly:
        # c·p1 = c·p2 = c·p3 = cos(r)
        # |c| = 1
        
        # c is in the span of {p1, p2, p3}
        # Let G be the Gram matrix: G[i][j] = p_i · p_j
        # Let c = sum_i w_i * p_i
        # c · p_j = sum_i w_i * G[i][j] = cos(r) for all j
        # So G @ w = cos(r) * ones(3)
        
        G = np.array([[np.dot(p1, p1), np.dot(p1, p2), np.dot(p1, p3)],
                      [np.dot(p2, p1), np.dot(p2, p2), np.dot(p2, p3)],
                      [np.dot(p3, p1), np.dot(p3, p2), np.dot(p3, p3)]])
        
        # G @ w = cos(r) * [1, 1, 1]
        # So w = cos(r) * G^{-1} @ [1, 1, 1]
        # And |c|^2 = w^T @ G @ w = cos(r)^2 * [1,1,1]^T @ G^{-1} @ G @ G^{-1} @ [1,1,1]
        # = cos(r)^2 * [1,1,1]^T @ G^{-1} @ [1,1,1]
        # = 1
        
        # So cos(r)^2 = 1 / ([1,1,1]^T @ G^{-1} @ [1,1,1])
        
        try:
            G_inv = np.linalg.inv(G)
            ones = np.ones(3)
            val = ones @ G_inv @ ones
            if val <= 0:
                # Degenerate
                continue
            cos_r = 1.0 / np.sqrt(val)
            r = np.arccos(np.clip(cos_r, -1.0, 1.0))
            max_radius = max(max_radius, r)
        except np.linalg.LinAlgError:
            continue
    
    return np.degrees(max_radius)


def _fibonacci_points():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    """Normalize each row of pts to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    global START_TIME, TIME_LIMIT
    
    # Start with Fibonacci spiral
    pts = _fibonacci_points()
    current_radius = _covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    # Gradient descent on the sphere
    # Objective: maximize the minimum distance from any point to the nearest configuration point
    # This is equivalent to minimizing the covering radius
    
    # We use a smooth surrogate: the soft-min of distances
    # Or we can use the epigraph formulation with t
    
    # For now, let's do a simple local search
    lr = 0.01
    num_iter = 2000
    
    for it in range(num_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Compute the covering radius and identify the worst facet
        # For each point, find the farthest point on the sphere from all configuration points
        # This is the circumcenter of the worst facet
        
        # Compute all pairwise distances
        # For each configuration point, compute the distance to all others
        # The covering radius is determined by the facets
        
        # Let's use a different approach:
        # The gradient of the covering radius with respect to each point
        # is determined by the worst facet
        
        # For simplicity, let's use a stochastic approach:
        # Randomly perturb each point and keep the perturbation if it improves the radius
        
        for i in range(N):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Random perturbation in the tangent space
            tangent = np.random.randn(3)
            tangent -= np.dot(tangent, pts[i]) * pts[i]  # project to tangent space
            tangent /= (np.linalg.norm(tangent) + 1e-12)
            
            # Try the perturbation
            old_pt = pts[i].copy()
            pts[i] = _normalize(pts[i] + lr * tangent)
            
            new_radius = _covering_radius(pts)
            
            if new_radius < current_radius:
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()
            else:
                pts[i] = old_pt
        
        # Decay learning rate
        lr *= 0.999
    
    return _normalize(best_pts)


# Run construct at module level to ensure it's defined
if __name__ == "__main__":
    result = construct()
    print(f"Covering radius: {_covering_radius(result)} degrees")
