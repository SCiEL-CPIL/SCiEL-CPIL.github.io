"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_deg(pts):
    """
    Compute covering radius in degrees by checking hull facets.
    Returns 180.0 if invalid.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 3):
        return 180.0
    if not np.all(np.isfinite(pts)):
        return 180.0
    
    # Check for coincident points (vectorized for speed)
    # For N=130, O(N^2) is fine, but let's be careful
    # Actually, if points are too close, hull might fail or be degenerate.
    # Let's just try to build the hull.
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # Check if origin is inside convex hull (hemisphere check)
    # If all points are in a hemisphere, the origin is not in the hull.
    # We can check if the minimum distance from origin to any facet is > 0?
    # Actually, if the origin is outside, there is a separating plane.
    # A simpler check: if the sum of areas of facets visible from origin covers the sphere?
    # Let's use the property: if origin is in hull, then for any vector v, 
    # max_i v·p_i > 0.
    # Or simpler: check if there exists a direction u such that u·p_i <= 0 for all i.
    # This is equivalent to checking if the convex hull of points contains the origin.
    # We can solve: min_t t s.t. sum_i lambda_i p_i = 0, lambda_i >= 0, sum lambda_i = 1.
    # If such lambda exists, origin is in hull.
    # Alternatively, use the hull equations.
    # SciPy ConvexHull equations: dot(points, normals) + offsets = 0.
    # The interior is dot(points, normals) + offsets <= 0.
    # Check origin (0,0,0): offsets <= 0?
    # If any offset > 0, the origin is on the "outside" of that facet, meaning the hull does not contain the origin.
    # Wait, the normal points OUT. So the interior is n·x + d <= 0.
    # If d > 0, then 0 + d > 0, so origin is outside.
    # If all d <= 0, origin is inside (or on boundary).
    offsets = hull.equations[:, -1]
    if np.any(offsets > 1e-10):
        return 180.0

    max_ang = 0.0
    for simplex in hull.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute circumcenter of the spherical triangle
        # Direction is v0×v1 + v1×v2 + v2×v0
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            # Degenerate, use centroid
            c = (v0 + v1 + v2) / 3.0
            c = c / np.linalg.norm(c)
        else:
            c = n / n_norm
        
        # Check if c is inside the spherical triangle
        # c is inside if it is on the same side of each edge as the opposite vertex
        n01 = np.cross(v0, v1)
        n12 = np.cross(v1, v2)
        n20 = np.cross(v2, v0)
        
        s1 = np.dot(v2, n01)
        s2 = np.dot(v0, n12)
        s3 = np.dot(v1, n20)
        
        t1 = np.dot(c, n01)
        t2 = np.dot(c, n12)
        t3 = np.dot(c, n20)
        
        inside = (s1 * t1 >= -1e-12) and (s2 * t2 >= -1e-12) and (s3 * t3 >= -1e-12)
        
        if inside:
            # Covering radius for this facet is angular distance from c to v0
            cos_ang = np.clip(np.dot(c, v0), -1.0, 1.0)
            ang = np.arccos(cos_ang) * 180.0 / np.pi
        else:
            # Circumcenter is outside, so the farthest point is one of the edge midpoints
            # The covering radius is determined by the edge with the largest angular length
            # or the midpoint of the longest edge?
            # Actually, for a spherical triangle, if the circumcenter is outside,
            # the farthest point from the vertices within the triangle is one of the vertices?
            # No, the covering radius is the radius of the smallest cap covering the triangle.
            # If circumcenter is outside, the smallest cap is centered at the vertex opposite the longest edge?
            # Or is it the midpoint of the longest edge?
            # Let's calculate the distance from the midpoint of each edge to the vertices.
            # The covering radius is the max of these.
            
            mid01 = (v0 + v1) / np.linalg.norm(v0 + v1)
            mid12 = (v1 + v2) / np.linalg.norm(v1 + v2)
            mid20 = (v2 + v0) / np.linalg.norm(v2 + v0)
            
            # Distance from midpoint to its adjacent vertices
            ang01 = np.arccos(np.clip(np.dot(mid01, v0), -1.0, 1.0)) * 180.0 / np.pi
            ang12 = np.arccos(np.clip(np.dot(mid12, v1), -1.0, 1.0)) * 180.0 / np.pi
            ang20 = np.arccos(np.clip(np.dot(mid20, v2), -1.0, 1.0)) * 180.0 / np.pi
            
            ang = max(ang01, ang12, ang20)
        
        max_ang = max(max_ang, ang)
    
    return max_ang

def _lloyd_step(pts):
    """
    Perform one Lloyd iteration: move each point towards the circumcenter of its Voronoi cell (hull facet).
    Returns new points and the covering radius.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts, 180.0
        
    new_pts = pts.copy()
    max_ang = 0.0
    
    for simplex in hull.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute circumcenter
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            c = (v0 + v1 + v2) / 3.0
            c = c / np.linalg.norm(c)
        else:
            c = n / n_norm
            
        # Check if inside
        n01 = np.cross(v0, v1)
        n12 = np.cross(v1, v2)
        n20 = np.cross(v2, v0)
        
        s1 = np.dot(v2, n01)
        s2 = np.dot(v0, n12)
        s3 = np.dot(v1, n20)
        
        t1 = np.dot(c, n01)
        t2 = np.dot(c, n12)
        t3 = np.dot(c, n20)
        
        inside = (s1 * t1 >= -1e-12) and (s2 * t2 >= -1e-12) and (s3 * t3 >= -1e-12)
        
        if inside:
            # Move all three vertices towards the circumcenter
            # This is the Lloyd step: update vertex to be the centroid of its cell?
            # Actually, in standard Lloyd for Voronoi, you move the point to the centroid of its cell.
            # Here, the "cell" for a vertex is not just this facet.
            # However, a common heuristic for sphere covering is to move points towards the circumcenters of the facets they belong to.
            # Since a point belongs to multiple facets, we can average the target directions.
            
            # Let's accumulate the target direction for each point
            # We'll do this by averaging the circumcenters of all facets containing that point.
            # For now, let's just move each vertex towards this specific circumcenter by a small step.
            # This is a simplified "gradient" step towards reducing the local covering radius.
            
            # Better: For each point, find the "worst" facet it belongs to and move towards that circumcenter.
            # Or simply: move towards the average of all circumcenters of facets it touches.
            
            # To keep it simple and effective, let's compute the "target" for each point as the average of the circumcenters of all its adjacent facets.
            # We need to map facet -> points.
            pass
        
        # For this specific facet, the covering radius contribution is:
        if inside:
            cos_ang = np.clip(np.dot(c, v0), -1.0, 1.0)
            ang = np.arccos(cos_ang) * 180.0 / np.pi
        else:
            mid01 = (v0 + v1) / np.linalg.norm(v0 + v1)
            mid12 = (v1 + v2) / np.linalg.norm(v1 + v2)
            mid20 = (v2 + v0) / np.linalg.norm(v2 + v0)
            
            ang01 = np.arccos(np.clip(np.dot(mid01, v0), -1.0, 1.0)) * 180.0 / np.pi
            ang12 = np.arccos(np.clip(np.dot(mid12, v1), -1.0, 1.0)) * 180.0 / np.pi
            ang20 = np.arccos(np.clip(np.dot(mid20, v2), -1.0, 1.0)) * 180.0 / np.pi
            ang = max(ang01, ang12, ang20)
            
        max_ang = max(max_ang, ang)
        
    # The simple "move towards circumcenter" logic above was skipped in the loop to avoid complexity.
    # Let's implement a proper Lloyd-like update:
    # For each point i, sum up the circumcenters of all facets containing i.
    # Normalize the sum and move point i towards that direction.
    
    targets = np.zeros((N, 3))
    counts = np.zeros(N)
    
    for simplex in hull.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            c = (v0 + v1 + v2) / 3.0
            c = c / np.linalg.norm(c)
        else:
            c = n / n_norm
            
        # Check if inside
        n01 = np.cross(v0, v1)
        n12 = np.cross(v1, v2)
        n20 = np.cross(v2, v0)
        
        s1 = np.dot(v2, n01)
        s2 = np.dot(v0, n12)
        s3 = np.dot(v1, n20)
        
        t1 = np.dot(c, n01)
        t2 = np.dot(c, n12)
        t3 = np.dot(c, n20)
        
        inside = (s1 * t1 >= -1e-12) and (s2 * t2 >= -1e-12) and (s3 * t3 >= -1e-12)
        
        if inside:
            # This circumcenter is a valid "worst case" direction for the three points
            # We add this direction to the target for each of the three points
            # Weighted by 1/sin(angle) or just 1? Let's use 1 for simplicity.
            targets[simplex[0]] += c
            counts[simplex[0]] += 1
            targets[simplex[1]] += c
            counts[simplex[1]] += 1
            targets[simplex[2]] += c
            counts[simplex[2]] += 1
        else:
            # If outside, the worst case is on an edge.
            # We can't easily assign a single "center" to move towards for the vertex.
            # We'll skip these facets for the target calculation, or use the midpoint of the longest edge.
            # Let's just skip for now, relying on the "inside" facets to guide the points.
            # If a point is only in "outside" facets, it might not move, which is okay if it's already good.
            pass

    # Update points
    step_size = 0.1
    for i in range(N):
        if counts[i] > 0:
            target_dir = targets[i] / np.linalg.norm(targets[i])
            # Move towards target
            new_pts[i] = pts[i] + step_size * target_dir
            new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
        else:
            new_pts[i] = pts[i]
            
    return _normalize(new_pts), max_ang

def construct():
    """
    Generate 130 points on the unit sphere minimizing covering radius.
    Uses multi-restart Lloyd-type optimization.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_pts = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    candidates.append(fib_pts)
    
    # 2. Random points with different seeds
    for seed in range(10):
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)
        
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline - 10.0:
            break
        try:
            score = _covering_radius_deg(pts)
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
            
    # Local optimization: Lloyd iterations
    if best_pts is not None:
        current_pts = best_pts.copy()
        current_score = best_score
        
        # Run Lloyd iterations
        # We will run a fixed number of iterations or until time runs out
        # Each iteration involves building a hull and updating points.
        # Building hull for N=130 is fast (~ms).
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5.0 and no_improve < max_no_improve:
            new_pts, new_score = _lloyd_step(current_pts)
            
            if new_score < current_score - 1e-7:
                current_pts = new_pts
                current_score = new_score
                no_improve = 0
                if new_score < best_score:
                    best_score = new_score
                    best_pts = current_pts.copy()
            else:
                no_improve += 1
                # If stuck, try a random perturbation to escape local minimum
                if no_improve % 10 == 0:
                    rng = np.random.RandomState()
                    noise = rng.randn(N, 3) * 0.05
                    current_pts = _normalize(current_pts + noise)
                    # Re-evaluate
                    try:
                        current_score = _covering_radius_deg(current_pts)
                    except:
                        current_score = 180.0
                    no_improve = 0 # Reset counter after perturbation
                
    # Final refinement: Gradient descent on smooth max?
    # If time permits, we can try a few steps of gradient descent using subgradients.
    # But Lloyd is usually good enough for this scale.
    
    if best_pts is None:
        # Fallback: Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_pts
# EVOLVE-BLOCK-END
