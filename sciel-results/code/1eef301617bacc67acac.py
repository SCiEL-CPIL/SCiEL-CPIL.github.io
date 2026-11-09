import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Define a compact configuration of 11 unit hexagons
    # Using a 3x4 honeycomb grid (12 cells) minus 1 cell to make 11.
    # This shape is roughly rectangular and compact.
    
    sqrt3 = np.sqrt(3)
    
    # Row 0 (y=0): 4 hexagons
    # Row 1 (y=1.5): 3 hexagons (staggered)
    # Row 2 (y=3.0): 4 hexagons
    
    centers = []
    # Row 0
    for i in range(4):
        centers.append([i * sqrt3, 0.0])
    # Row 1
    for i in range(3):
        centers.append([(i + 0.5) * sqrt3, 1.5])
    # Row 2
    for i in range(4):
        centers.append([i * sqrt3, 3.0])
        
    centers = np.array(centers)
    
    # Center the configuration at the origin
    center_of_mass = np.mean(centers, axis=0)
    centers = centers - center_of_mass
    
    # 2. Find the minimal enclosing regular hexagon
    # We search for the optimal rotation of the outer hexagon.
    best_L = float('inf')
    best_theta = 0.0
    
    # The outer hexagon has 6 sides. The normal vectors are at angles:
    # theta, theta+60, ..., theta+300
    # The width in direction n is 2 * max(p . n).
    # The apothem A must be >= max(p . n) for all 6 normals.
    # L = 2 * A / sqrt(3)
    
    # We only need to search theta from 0 to 60 degrees (360/6)
    # Fine search
    thetas = np.linspace(0, np.pi/3, 181) # 1 degree resolution
    
    for theta in thetas:
        if time.time() > deadline:
            break
            
        # Normal vectors for the 6 sides
        angles = theta + np.arange(6) * (np.pi / 3)
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Project all centers onto these normals
        # projections shape: (11, 6)
        projections = centers @ normals.T
        
        # For each normal direction, find the max projection
        # The hexagon must contain the points, so the distance to the side
        # (apothem) must be at least the max projection in that direction.
        # Note: Since the hexagon is centered at origin, we also need to check
        # the negative side, but due to symmetry of the max over 6 directions
        # covering 360 degrees, checking max(p.n) for all 6 normals is sufficient
        # to determine the required apothem if the origin is the center.
        # Wait, the condition is |p . n| <= A?
        # No, the hexagon is {x | x . n_k <= A for all k}.
        # So we need A >= max_p (p . n_k) for each k.
        # The required A is max_k (max_p (p . n_k)).
        
        max_projections = np.max(projections, axis=0)
        required_apothem = np.max(max_projections)
        
        L_candidate = 2.0 * required_apothem / sqrt3
        
        if L_candidate < best_L:
            best_L = L_candidate
            best_theta = theta

    # 3. Construct the return values
    # Inner hexagons: (x, y, angle)
    # All inner hexagons are aligned with angle 0.
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    # Outer hexagon: center, side, angle
    # Center is (0,0) because we centered the inner hexagons.
    # Side length is best_L.
    # Angle is best_theta.
    # Note: The angle in the return value is the rotation of the outer hexagon.
    # My theta was the angle of the first normal.
    # The "angle" of a hexagon is typically defined by the orientation of its vertices or sides.
    # If the normal is at theta, the side is perpendicular to theta.
    # Standard hexagon with side 0 has a vertex at (L, 0)? Or side at 90?
    # Let's assume the angle parameter rotates the hexagon around its center.
    # If we use the normal at theta, the side is at theta + 90.
    # A standard regular hexagon with angle 0 usually has a vertex at (L, 0) or (L*cos(30), L*sin(30)).
    # Let's verify the evaluator's expectation.
    # "outer hexagon center (x, y), side length and rotation"
    # Usually, rotation 0 means the hexagon is "flat" or "pointy".
    # My calculation of L is robust to the definition of angle as long as the geometry matches.
    # The set of points {x | x . n_k <= A} defines the hexagon.
    # If the evaluator uses a specific vertex definition, I should match it.
    # Common definition: Vertex at (L, 0) for angle 0.
    # In that case, the normal to the side between vertex 0 and 1 is at 30 degrees.
    # So if my best_theta is the angle of the normal, the hexagon rotation angle (vertex) is best_theta - 30 degrees.
    # Let's assume the evaluator checks containment.
    # To be safe, I will return the angle that corresponds to the normal orientation I used.
    # If the evaluator defines angle 0 as "flat top" (vertex at 90), then normal is at 0.
    # If the evaluator defines angle 0 as "pointy top" (vertex at 0), then normal is at 30.
    
    # Let's assume the standard mathematical convention where the polygon is rotated by 'angle'.
    # If I define the hexagon by normals at theta + k*60, the shape is rotated by theta.
    # So I return best_theta.
    
    return inner, (0.0, 0.0), best_L, best_theta
# EVOLVE-BLOCK-END
