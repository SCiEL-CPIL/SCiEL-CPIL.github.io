import numpy as np
import time
from scipy.optimize import minimize

N = 11
MIN_DIST = np.sqrt(3) # Distance between centers of touching unit hexagons is sqrt(3)? 
# Wait. Two unit hexagons side by side. Distance between centers is 2 * inradius = 2 * (sqrt(3)/2) = sqrt(3).
# Yes, MIN_DIST is correct for non-overlapping hexagons with same orientation.
# If rotated, min distance can be smaller? No, the inradius is the minimum distance to boundary.
# So centers must be at least 2 * inradius = sqrt(3) apart if they touch side-to-side.
# If they touch vertex-to-vertex? The distance is larger.
# So MIN_DIST = sqrt(3) is the correct lower bound for centers of disjoint unit hexagons (same orientation).

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def cost_func(params):
        # params is a 1D array of 22 elements: [x1, y1, x2, y2, ..., x11, y11]
        points = params.reshape(-1, 2)
        
        # Center the points at the origin to assume the outer hexagon is centered there
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Calculate the side length L of the smallest enclosing regular hexagon (centered at 0,0)
        # The outer hexagon has side length L. Distance from center to side is L * sqrt(3) / 2.
        # The inner hexagons have "radius" 1.0 in the direction of the normals (assuming aligned).
        # So the effective extent is max_proj + 1.0.
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L_enc = (max_proj + 1.0) / (np.sqrt(3) / 2)
        
        # Calculate overlap penalty
        # We want distance >= MIN_DIST
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        # If there are overlaps, the solution is invalid.
        # We use a large penalty to push them apart.
        return L_enc + 1000.0 * penalty

    # Random restarts to find a good local minimum
    best_L = float('inf')
    best_params = None
    
    # Generate a good initial guess: Hexagonal lattice
    # Axial coordinates (q, r) to Cartesian (x, y) for pointy-top or flat-top?
    # Let's use standard hex grid:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2.0)
    # This creates a grid with horizontal rows.
    # Distance between neighbors is 1.0?
    # (0,0) to (1,0): x=1.5, y=sqrt(3)/2. dist = sqrt(2.25+0.75)=sqrt(3). Correct.
    
    lattice_points = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            x = 1.5 * q
            y = np.sqrt(3) * (r + q/2.0)
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    
    # Sort by distance to center and take 11
    dists = np.linalg.norm(lattice_points, axis=1)
    indices = np.argsort(dists)[:N]
    base_init = lattice_points[indices]
    
    for restart in range(100):
        # Add noise to the base initialization
        noise_scale = 0.5 if restart > 0 else 0.1
        init_pts = base_init + np.random.normal(0, noise_scale, base_init.shape)
        initial_params = init_pts.flatten()
        
        # Use Powell optimization, which is robust for this kind of problem
        try:
            res = minimize(cost_func, initial_params, method='Powell', 
                           options={'maxiter': 500, 'ftol': 1e-6, 'xtol': 1e-6})
        except Exception:
            continue

        final_pts = res.x.reshape(-1, 2)
        centroid = np.mean(final_pts, axis=0)
        final_pts_centered = final_pts - centroid
        
        # Check for overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(final_pts_centered[i] - final_pts_centered[j])
                if d < MIN_DIST - 1e-5:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            projections = np.abs(final_pts_centered @ normals.T)
            max_proj = np.max(projections)
            L_enc = (max_proj + 1.0) / (np.sqrt(3) / 2)
            
            if L_enc < best_L:
                best_L = L_enc
                best_params = res.x

    # If no valid solution found, fallback to a safe large solution
    if best_params is None:
        # Return a valid but large solution
        # Place them in a line
        inner = [[float(k * MIN_DIST), 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        # Calculate required L for this line
        # Max projection is at ends.
        # x range: -5*sqrt(3) to 5*sqrt(3).
        # max_proj = 5*sqrt(3).
        # L = (5*sqrt(3) + 1) / (sqrt(3)/2)
        L_fallback = (5 * np.sqrt(3) + 1.0) / (np.sqrt(3) / 2)
        return inner, (0.0, 0.0), L_fallback, 0.0

    # Construct the output
    inner_pts = best_params.reshape(-1, 2)
    # Re-center the points so the centroid is at (0,0) to match the outer hexagon center
    centroid = np.mean(inner_pts, axis=0)
    inner_pts_centered = inner_pts - centroid
    
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts_centered]
    
    # Recalculate L_enc for the final centered points
    projections = np.abs(inner_pts_centered @ normals.T)
    max_proj = np.max(projections)
    outer_side = (max_proj + 1.0) / (np.sqrt(3) / 2)
    
    # Add a small buffer to ensure validity due to floating point errors
    outer_side += 1e-5

    return inner, (0.0, 0.0), outer_side, 0.0
