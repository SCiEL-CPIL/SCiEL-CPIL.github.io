import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Generate hexagonal grid coordinates
    # We want N hexagons closest to the origin in axial coordinates (q, r)
    # Distance in axial coords is approx (|q| + |r| + |q+r|)/2
    coords = []
    q = 0
    r = 0
    # Simple spiral or just generate a large enough range and sort
    # For N=11, a range of -3 to 3 is sufficient
    max_range = int(np.ceil(np.sqrt(N))) + 2
    candidates = []
    for q in range(-max_range, max_range + 1):
        for r in range(-max_range, max_range + 1):
            # Axial distance from origin
            dist = (abs(q) + abs(r) + abs(q + r)) / 2.0
            candidates.append((dist, q, r))
    
    # Sort by distance, then by q, r for stability
    candidates.sort()
    
    # Take first N
    selected = candidates[:N]
    
    # Convert to Cartesian
    # Flat-top hexagon, side length 1
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    points = []
    for _, q, r in selected:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        points.append([x, y])
    
    points = np.array(points)
    
    # Center the points
    center_offset = np.mean(points, axis=0)
    points_centered = points - center_offset
    
    # Calculate widths in the 6 normal directions
    # Normals for flat-top hexagon are at 30, 90, 150, 210, 270, 330 degrees
    angles = [np.deg2rad(30 + 60 * k) for k in range(6)]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Project points onto normals
    projections = points_centered @ normals.T
    
    # Calculate max width
    max_width = 0.0
    for k in range(6):
        width = np.max(projections[:, k]) - np.min(projections[:, k])
        if width > max_width:
            max_width = width
            
    # Calculate required apothem A
    # A >= 0.5 * (width + sqrt(3))
    # The inner hexagon apothem is sqrt(3)/2, so total extension is sqrt(3)
    A = 0.5 * (max_width + np.sqrt(3))
    
    # Calculate side length S
    # A = S * sqrt(3) / 2  => S = 2 * A / sqrt(3)
    S = 2.0 * A / np.sqrt(3)
    
    # The inner hexagons are at points_centered, rotation 0
    # The outer hexagon is at (0,0), side S, rotation 0
    # But we must return the original coordinates for inner hexagons?
    # The task says: "inner rows (x, y, angle_degrees)"
    # And "outer hexagon center (x, y)"
    # If we centered the points, the outer center should be the original center_offset?
    # No, we can just return the centered points and center (0,0).
    # Or return original points and center center_offset.
    # Let's return centered points and (0,0) for simplicity.
    
    inner = [[p[0], p[1], 0.0] for p in points_centered]
    outer_center = (0.0, 0.0)
    
    return inner, outer_center, S, 0.0
# EVOLVE-BLOCK-END
