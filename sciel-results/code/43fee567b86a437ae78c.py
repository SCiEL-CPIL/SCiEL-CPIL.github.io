import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    
    # Generate hex lattice points
    # Axial coordinates (q, r)
    # Cartesian: x = 2q + r, y = sqrt(3)r
    # This corresponds to hexagons of side 1, distance between centers 2.
    
    points = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            x = 2*q + r
            y = np.sqrt(3)*r
            dist_sq = x*x + y*y
            points.append((dist_sq, x, y))
            
    points.sort()
    
    # Take N closest
    selected = points[:N]
    centers = np.array([(p[1], p[2]) for p in selected])
    
    # Calculate minimal enclosing regular hexagon
    # We try different rotations of the outer hexagon
    # and different centers (origin and centroid)
    
    best_S = float('inf')
    best_phi = 0.0
    best_center = (0.0, 0.0)
    
    # Candidate centers
    center_candidates = [np.array([0.0, 0.0]), np.mean(centers, axis=0)]
    
    for c in center_candidates:
        # Try 120 rotations for finer granularity
        for phi in np.linspace(0, np.pi/3, 120, endpoint=False):
            # Normals of the hexagon edges
            # A regular hexagon with side S has edge normals at 30, 90, 150, 210, 270, 330 degrees
            # Plus rotation phi
            angles = np.pi/6 + phi + np.arange(6) * np.pi/3
            normals = np.column_stack((np.cos(angles), np.sin(angles)))
            
            # Vectors from center to all inner hex centers
            vecs = centers - c
            
            # Calculate distance from each center to each edge line
            # Note: The inner hexagons have size. The centers must be such that the
            # entire hexagon is inside.
            # The condition for a unit hexagon centered at p to be inside the outer hexagon
            # is that the distance from p to each edge of the outer hexagon is >= inradius of inner hex.
            # Inradius of unit hex is sqrt(3)/2.
            # So we need dist(p, edge) >= sqrt(3)/2.
            # This is equivalent to dist(p, edge) - sqrt(3)/2 >= 0.
            # The side length S is related to the distance from center to edge (inradius R_in) by S = 2*R_in/sqrt(3).
            # So R_in = S*sqrt(3)/2.
            # We need R_in - dist(p, edge) >= 0? No.
            # The edge is at distance R_in from the center.
            # The point p is at distance d from the edge.
            # We need d >= inradius_inner = sqrt(3)/2.
            # So R_in - d <= R_in - sqrt(3)/2.
            # We want to minimize S, which minimizes R_in.
            # So we need R_in >= max_k (d_k + sqrt(3)/2).
            # S = 2/sqrt(3) * R_in.
            # So S = 2/sqrt(3) * (max_k d_k + sqrt(3)/2).
            
            dists = np.abs(vecs @ normals.T)
            max_dist = np.max(dists)
            
            R_in = max_dist + np.sqrt(3)/2
            S = 2 * R_in / np.sqrt(3)
            
            if S < best_S:
                best_S = S
                best_phi = phi
                best_center = c
                
            if time.time() > 10.0:
                break
        if time.time() > 10.0:
            break
            
    # Construct inner hexagons
    inner = []
    for i in range(N):
        x, y = centers[i]
        inner.append([float(x), float(y), 0.0])
        
    return inner, (float(best_center[0]), float(best_center[1])), float(best_S), float(best_phi * 180 / np.pi)
