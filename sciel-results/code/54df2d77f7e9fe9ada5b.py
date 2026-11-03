import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice points
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3)/2 * (q + r/2), y = 3/2 * r
    # Distance squared from origin: q^2 + q*r + r^2
    points = []
    max_coord = 10
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            # Ensure we are within a reasonable radius
            dist_sq = q*q + q*r + r*r
            if dist_sq <= 25: # 5^2
                x = math.sqrt(3)/2 * (q + r/2.0)
                y = 1.5 * r
                points.append((dist_sq, x, y))
    
    # Sort by distance to origin
    points.sort(key=lambda p: p[0])
    
    # Take the first N points
    centers = []
    for _, x, y in points[:N]:
        centers.append((x, y))
        
    # Convert to numpy array
    centers = np.array(centers)
    
    # Optimize outer hexagon rotation phi
    # Outer hexagon normals are at phi + i*pi/3
    # Inner hexagons are aligned with outer (rotation 0 relative to outer)
    # Support function of inner hexagon in normal direction is sqrt(3)/2
    # Condition: max_p (p . n_i) + sqrt(3)/2 <= S * sqrt(3)/2
    # S >= (2/sqrt(3)) * max_p (p . n_i) + 1
    
    def calc_S(phi):
        max_proj = 0.0
        for i in range(6):
            angle = phi + i * math.pi / 3.0
            n = np.array([math.cos(angle), math.sin(angle)])
            projections = centers.dot(n)
            max_proj = max(max_proj, np.max(projections))
        return (2.0 / math.sqrt(3.0)) * max_proj + 1.0

    # Search for optimal phi in [0, pi/6] due to symmetry
    best_S = float('inf')
    best_phi = 0.0
    # Fine search
    for phi_deg in np.linspace(0, 30, 301):
        phi = math.radians(phi_deg)
        S = calc_S(phi)
        if S < best_S:
            best_S = S
            best_phi = phi
            
    # Refine search around best_phi
    for phi_deg in np.linspace(math.degrees(best_phi) - 1, math.degrees(best_phi) + 1, 201):
        phi = math.radians(phi_deg)
        S = calc_S(phi)
        if S < best_S:
            best_S = S
            best_phi = phi

    # Construct inner list
    # All inner hexagons have angle 0 (aligned with outer)
    # Outer hexagon has rotation best_phi
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Return inner, center, side, angle
    # Center is (0,0)
    # Side is best_S
    # Angle is best_phi in degrees
    return inner, (0.0, 0.0), best_S, math.degrees(best_phi)
# EVOLVE-BLOCK-END
