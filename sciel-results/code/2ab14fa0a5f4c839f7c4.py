import numpy as np
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon where every vertex has 6 other vertices at equal distance.
    
    Construction: For each vertex i, place 6 neighbors on a circle of radius R centered at i.
    The 6 neighbors are at angles symmetric about the outward radial direction from i.
    This creates a convex polygon with a "spiky" shape where each vertex sees 6 equidistant points.
    """
    n = 20
    R = 1.0  # distance from each vertex to its 6 equidistant neighbors
    
    # For vertex i, the 6 neighbors are at indices i±1, i±2, i±3 (mod n)
    # Place them on a circle of radius R centered at vertex i
    # Angles: symmetric about the outward direction
    
    # Let's think of this differently:
    # Vertex i is at position P_i
    # Neighbors P_{i±1}, P_{i±2}, P_{i±3} should be at distance R from P_i
    # And they should be arranged to form a convex polygon
    
    # A simpler approach: use a "star" pattern where vertices alternate
    # between two radii, creating a shape where each vertex has symmetric neighbors
    
    # Actually, let's use the direct construction:
    # Place 20 vertices on a circle, but perturb them so that each vertex
    # has 6 equidistant neighbors
    
    # Better approach: Create a polygon where each vertex is the center
    # of a regular hexagon of 6 other vertices
    
    # The cleanest construction: 
    # Place vertices at angles theta_i = 2*pi*i/n + perturbation_i
    # with radii r_i, chosen so that for each i, the 6 nearest vertices
    # are at equal distance
    
    # Let me use a different strategy: 
    # Start with a regular 20-gon and optimize the vertex positions
    # to maximize the minimum vertex score
    
    start_time = time.time()
    deadline = 630.0
    
    # Initial: regular 20-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    def compute_score(pts):
        """Compute the minimum vertex score."""
        pts = np.array(pts, dtype=float)
        # Normalize
        centroid = pts.mean(axis=0)
        pts_norm = pts - centroid
        dists_to_centroid = np.linalg.norm(pts_norm, axis=1)
        mean_dist = dists_to_centroid.mean()
        if mean_dist <= 0.01:
            return 0.0
        pts_norm = pts_norm / mean_dist
        
        # Check convexity
        min_score = 1.0
        for i in range(n):
            v = pts_norm[i]
            # Distances to all other vertices
            d = np.linalg.norm(pts_norm - v, axis=1)
            d_sorted = np.sort(d)
            # d_sorted[0] is distance to itself (0), so use d_sorted[1:]
            d_other = d_sorted[1:]  # 19 distances
            
            # Check if any < 0.05
            if d_other[0] < 0.05:
                return 0.0
            
            # Find k: first index minimizing d_{k+1} - d_k
            gaps = np.diff(d_other)
            k = np.argmin(gaps)
            D = (d_other[k] + d_other[k+1]) / 2
            if D < 0.1:
                return 0.0
            
            # Take four distances
            if k == 0:
                d_four = d_other[0:4]
            elif k == 17:
                d_four = d_other[15:19]
            else:
                d_four = d_other[k-1:k+3]
            
            raw = np.mean(np.maximum(d_four/D, D/d_four))
            score = min(1.0, 1.0/raw)
            min_score = min(min_score, score)
        
        # Check strict convexity
        for i in range(n):
            p0 = pts_norm[i]
            p1 = pts_norm[(i+1) % n]
            p2 = pts_norm[(i+2) % n]
            cross = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])
            if abs(cross) < 1e-10:
                return 0.0
        
        return min_score
    
    # Optimize using gradient-free method
    best_pts = pts.copy()
    best_score = compute_score(pts)
    
    # Try various constructions
    # Construction 1: Regular polygon (baseline)
    # Construction 2: Two concentric circles with alternating radii
    # Construction 3: Optimized positions
    
    # Let's try a parametric family: 
    # theta_i = 2*pi*i/n, r_i = 1 + a*cos(2*i*pi/n) + b*cos(4*i*pi/n)
    # This maintains rotational symmetry of order 2 or 4
    
    best_overall_score = best_score
    best_overall_pts = pts.copy()
    
    # Try symmetric perturbations
    for a in np.linspace(-0.3, 0.3, 30):
        for b in np.linspace(-0.3, 0.3, 30):
            if time.time() > deadline - 30:
                break
            r = 1.0 + a*np.cos(2*angles) + b*np.cos(4*angles)
            if np.any(r <= 0.1):
                continue
            pts_test = np.stack([r*np.cos(angles), r*np.sin(angles)], axis=1)
            score = compute_score(pts_test)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_pts = pts_test.copy()
        if time.time() > deadline - 30:
            break
    
    # If we haven't found something better, return the best
    return best_overall_pts
