import numpy as np
import itertools
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the lattice points for the inner hexagons
    # Ring 0: Center
    ring0 = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors at distance 2
    ring1 = []
    for k in range(6):
        angle = k * math.pi / 3
        ring1.append((2.0 * math.cos(angle), 2.0 * math.sin(angle)))
        
    # Ring 2: 12 neighbors at distance 4
    # These are at angles k * 30 degrees
    ring2 = []
    for k in range(12):
        angle = k * math.pi / 6
        ring2.append((4.0 * math.cos(angle), 4.0 * math.sin(angle)))

    # We will search over all combinations of 5 points from Ring 2
    # to form a 1+6+5 cluster
    best_S = float('inf')
    best_subset = None
    best_alpha = 0.0
    best_centers = None
    
    # Precompute vertices for a unit hexagon at origin, angle 0
    # Vertices are at distance 1, angles 0, 60, ...
    hex_vertices_local = [(math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)) for k in range(6)]
    
    # Sample outer hexagon rotation angles
    # The optimal alignment will be within 0 to 60 degrees
    alpha_steps = 200
    alphas = [i * (math.pi / 3) / alpha_steps for i in range(alpha_steps)]
    
    for subset in itertools.combinations(range(12), 5):
        # Form the list of 12 centers
        centers = ring0 + ring1 + [ring2[i] for i in subset]
        
        # Generate all vertices of the 12 hexagons
        # Each hexagon contributes 6 vertices
        # Total 72 vertices
        all_vertices = []
        for cx, cy in centers:
            for vx, vy in hex_vertices_local:
                all_vertices.append((cx + vx, cy + vy))
        
        # Find the minimal enclosing hexagon side length S
        # by varying the rotation of the outer hexagon
        min_S_for_subset = float('inf')
        best_alpha_for_subset = 0.0
        
        for alpha in alphas:
            # Normals for the outer hexagon
            # The sides are perpendicular to the normals
            # For a regular hexagon, normals are at angles alpha + i * 60
            # The apothem A must be >= max dot product of any vertex with any normal
            max_dot = 0.0
            for i in range(6):
                n_angle = alpha + i * math.pi / 3
                nx = math.cos(n_angle)
                ny = math.sin(n_angle)
                for vx, vy in all_vertices:
                    dot = vx * nx + vy * ny
                    if dot > max_dot:
                        max_dot = dot
            # max_dot is the required apothem A
            S = (2.0 / math.sqrt(3.0)) * max_dot
            if S < min_S_for_subset:
                min_S_for_subset = S
                best_alpha_for_subset = alpha
                
        if min_S_for_subset < best_S:
            best_S = min_S_for_subset
            best_subset = subset
            best_alpha = best_alpha_for_subset
            best_centers = centers

    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner = []
    for cx, cy in best_centers:
        inner.append([cx, cy, 0.0])
        
    # outer: (x, y, side, angle)
    # Center is (0,0)
    # Side is best_S
    # Angle is best_alpha (in radians, convert to degrees if needed? 
    # The task says "angle_degrees" for inner, but for outer it says "rotation".
    # Usually angles are in degrees in these tasks unless specified.
    # The seed program used 0.0. Let's assume degrees for consistency with inner.
    outer_angle_deg = math.degrees(best_alpha)
    
    return inner, (0.0, 0.0), best_S, outer_angle_deg
# EVOLVE-BLOCK-END
