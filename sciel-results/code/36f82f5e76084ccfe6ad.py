import numpy as np
import time
import itertools

# Constants
N_HEX = 11
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin

def construct():
    start_t = time.time()
    
    # Lattice vectors for triangular lattice of hexagons with side 1
    # Distance between centers is sqrt(3)
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 3*np.sqrt(3)/2])
    
    # Generate candidate lattice points
    # We need to cover a region large enough for 11 hexes
    # Max radius approx 4 units
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            points.append(p)
    points = np.array(points)
    
    # Function to check if a point is valid (non-overlapping) with existing set
    def is_valid(new_p, existing_centers):
        for c in existing_centers:
            dist = np.linalg.norm(new_p - c)
            if dist < np.sqrt(3) - 1e-9:
                return False
        return True

    # Function to get vertices of a unit hexagon at center c
    def get_hex_vertices(c):
        # Vertices at angles 0, 60, ... 300
        angles = np.deg2rad(np.arange(0, 360, 60))
        vertices = []
        for a in angles:
            v = c + np.array([np.cos(a), np.sin(a)])
            vertices.append(v)
        return np.array(vertices)

    # Function to calculate minimal enclosing hexagon side length
    # Given a set of vertices (flattened), and outer hex center C, angle theta
    def calc_min_side(vertices, C, theta):
        # Normals to the sides of the outer hexagon
        # Side normals are at theta, theta+60, theta+120
        # The distance from center to side is S * sqrt(3) / 2
        # So S = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
        
        # To be efficient, we project all vertices onto the 3 normal directions
        # But we can just check the 3 pairs of opposite sides.
        # Actually, we need the max absolute projection on the 3 normal vectors.
        
        n0 = np.array([np.cos(theta), np.sin(theta)])
        n1 = np.array([np.cos(theta + np.pi/3), np.sin(theta + np.pi/3)])
        n2 = np.array([np.cos(theta + 2*np.pi/3), np.sin(theta + 2*np.pi/3)])
        
        # Project vertices relative to C
        rel = vertices - C
        p0 = np.abs(rel @ n0)
        p1 = np.abs(rel @ n1)
        p2 = np.abs(rel @ n2)
        
        max_proj = np.max(np.concatenate([p0, p1, p2]))
        return (2.0 / np.sqrt(3)) * max_proj

    # Function to find best theta for a fixed set of vertices and center
    def optimize_theta(vertices, C):
        # Sample theta
        best_s = float('inf')
        best_theta = 0.0
        # The function is periodic and symmetric. Sample 0 to pi/2 with fine step
        thetas = np.linspace(0, np.pi/2, 100)
        for t in thetas:
            s = calc_min_side(vertices, C, t)
            if s < best_s:
                best_s = s
                best_theta = t
        # Refine with scipy if available, otherwise return best sampled
        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(lambda t: calc_min_side(vertices, C, t), bounds=(best_theta - 0.1, best_theta + 0.1), method='bounded')
            if res.success:
                best_s = res.fun
                best_theta = res.x
        except:
            pass
        return best_s, best_theta

    # Greedy Packing
    best_config = None
    best_score = float('inf')
    
    # Try a few restarts
    for restart in range(5):
        if time.time() - start_t > TIME_LIMIT:
            break
            
        centers = [np.array([0.0, 0.0])]
        
        for _ in range(N_HEX - 1):
            if time.time() - start_t > TIME_LIMIT:
                break
                
            best_p = None
            best_s_candidate = float('inf')
            
            # Filter valid points
            valid_points = []
            for p in points:
                if is_valid(p, centers):
                    valid_points.append(p)
            
            if not valid_points:
                break
                
            # Evaluate each valid point
            # To speed up, we can use a heuristic metric first, then refine top K
            # Metric: max distance from current centroid
            current_centroid = np.mean(centers, axis=0)
            
            candidates = []
            for p in valid_points:
                # Heuristic score
                dist = np.linalg.norm(p - current_centroid)
                candidates.append((dist, p))
            
            candidates.sort(key=lambda x: x[0])
            
            # Check top 10 candidates with full S calculation
            for dist, p in candidates[:10]:
                new_centers = centers + [p]
                new_centroid = np.mean(new_centers, axis=0)
                
                # Get all vertices
                all_verts = []
                for c in new_centers:
                    all_verts.extend(get_hex_vertices(c))
                all_verts = np.array(all_verts)
                
                s, _ = optimize_theta(all_verts, new_centroid)
                
                if s < best_s_candidate:
                    best_s_candidate = s
                    best_p = p
            
            if best_p is not None:
                centers.append(best_p)
            else:
                # If no valid point found or all too bad, break
                break
        
        if len(centers) == N_HEX:
            # Final optimization for this configuration
            all_verts = []
            for c in centers:
                all_verts.extend(get_hex_vertices(c))
            all_verts = np.array(all_verts)
            
            centroid = np.mean(centers, axis=0)
            s, theta = optimize_theta(all_verts, centroid)
            
            # Optionally optimize centroid slightly?
            # For now, centroid is a good guess.
            
            if s < best_score:
                best_score = s
                best_config = (centers, centroid, s, theta)
    
    if best_config is None:
        # Fallback: Return a valid but large configuration
        # Line of 11 hexes
        centers = [np.array([k * np.sqrt(3), 0.0]) for k in range(N_HEX)]
        all_verts = []
        for c in centers:
            all_verts.extend(get_hex_vertices(c))
        all_verts = np.array(all_verts)
        centroid = np.mean(centers, axis=0)
        s, theta = optimize_theta(all_verts, centroid)
        best_config = (centers, centroid, s, theta)
    
    centers, centroid, s, theta = best_config
    
    # Prepare output
    inner = []
    for c in centers:
        # Angle is 0 for all inner hexes (aligned with lattice)
        inner.append([c[0], c[1], 0.0])
        
    return inner, (centroid[0], centroid[1]), s, np.rad2deg(theta)

# EVOLVE-BLOCK-END
