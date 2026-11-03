"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Pre-compute hexagon vertices for unit hexagon (angle 0)
    # Vertices are at distance 1 from center
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_verts = np.column_stack((np.cos(hex_angles), np.sin(hex_angles)))
    
    # Pre-compute normals for enclosing hexagon (angle 0)
    # Normals point to the midpoints of the sides
    # Side midpoints are at distance sqrt(3)/2 from center
    # Angles for normals: 0, 60, 120...
    enc_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    enc_normals = np.column_stack((np.cos(enc_angles), np.sin(enc_normals)))
    
    def get_min_side(centers, outer_angle):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that encloses all hexagons at 'centers'.
        """
        # Rotate enclosing normals by outer_angle
        cos_a = np.cos(outer_angle)
        sin_a = np.sin(outer_angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_normals = enc_normals @ rot_matrix.T
        
        max_proj = 0.0
        # For each inner hexagon
        for cx, cy in centers:
            # Add center offset
            # Project all vertices
            # Vertex pos = (cx, cy) + v
            # Projection = n . ((cx, cy) + v) = n . (cx, cy) + n . v
            # We can precompute n . v for the 6 vertices of the unit hex?
            # But n changes with outer_angle.
            # So we just do the dot product.
            center_vec = np.array([cx, cy])
            for v in hex_verts:
                p = center_vec + v
                # Dot products for all 6 normals
                projs = rotated_normals @ p
                current_max = np.max(projs)
                if current_max > max_proj:
                    max_proj = current_max
        
        # S * sqrt(3)/2 = max_proj
        # S = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / np.sqrt(3.0)

    def check_overlap(centers):
        """
        Check if any two hexagons overlap.
        Assumes all hexagons have the same orientation (angle 0).
        Distance between centers must be >= sqrt(3).
        """
        for i in range(N):
            xi, yi = centers[i]
            for j in range(i + 1, N):
                xj, yj = centers[j]
                dx = xi - xj
                dy = yi - yj
                if dx * dx + dy * dy < 3.0 - 1e-9:
                    return False
        return True

    # Initialization
    # Start with a rough cluster
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * np.pi / 3
        centers.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
    for i in range(5):
        centers.append((random.uniform(-4, 4), random.uniform(-4, 4)))
    
    best_score = float('inf')
    best_centers = centers[:]
    best_outer_angle = 0.0
    
    # Optimization Loop
    # Use a simple random restart / hill climbing
    num_iterations = 3000
    step_size = 1.0
    
    for _ in range(num_iterations):
        # Perturb centers
        new_centers = []
        for cx, cy in best_centers:
            new_cx = cx + random.gauss(0, step_size)
            new_cy = cy + random.gauss(0, step_size)
            new_centers.append((new_cx, new_cy))
        
        # Perturb outer angle
        new_outer_angle = best_outer_angle + random.gauss(0, 0.1)
        
        if check_overlap(new_centers):
            score = get_min_side(new_centers, new_outer_angle)
            if score < best_score:
                best_score = score
                best_centers = new_centers[:]
                best_outer_angle = new_outer_angle
                
    # Final refinement: try to tighten further
    # (Optional: run a second pass with smaller step size)
    
    inner = [[cx, cy, 0.0] for cx, cy in best_centers]
    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
