"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    def compute_min_hex_side(centers, outer_angle):
        """
        Given centers of unit hexagons, find the minimal side length of a regular hexagon
        with given orientation (outer_angle in radians) that contains all hexagons.
        The outer hexagon center is chosen optimally.
        """
        # Vertices of unit hexagon (circumradius 1)
        angles = np.arange(6) * np.pi / 3.0
        vertices = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # 6 normal directions of outer hexagon
        n_angles = outer_angle + np.arange(6) * np.pi / 3.0
        normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
        
        # Support function of unit hexagon in each normal direction
        dots = vertices @ normals.T  # shape (6, 6)
        h_unit = dots.max(axis=0)  # shape (6,)
        
        # Project centers onto each normal
        projections = centers @ normals  # shape (n, 6)
        A = projections.max(axis=0)  # shape (6,)
        B = projections.min(axis=0)  # shape (6,)
        
        # s >= (A_k - B_k + 2 * h_unit[k]) / sqrt(3) for each k
        s_lower = (A - B + 2 * h_unit) / sqrt3
        
        return s_lower.max()
    
    # Generate candidate arrangements
    candidates = []
    
    # Arrangement: 2 rows of 6
    centers = []
    for i in range(6):
        centers.append([i * sqrt3, 0.0])
        centers.append([i * sqrt3 + sqrt3 / 2.0, 1.5])
    candidates.append(np.array(centers))
    
    # Arrangement: 3 rows of 4
    centers = []
    for j in range(3):
        offset = (j % 2) * sqrt3 / 2.0
        for i in range(4):
            x = i * sqrt3 + offset
            y = j * 1.5
            centers.append([x, y])
    candidates.append(np.array(centers))
    
    # Arrangement: 4 rows of 3
    centers = []
    for j in range(4):
        offset = (j % 2) * sqrt3 / 2.0
        for i in range(3):
            x = i * sqrt3 + offset
            y = j * 1.5
            centers.append([x, y])
    candidates.append(np.array(centers))
    
    # Arrangement: rows of 2,4,4,2 (compact)
    centers = []
    for j, count in enumerate([2, 4, 4, 2]):
        offset = (j % 2) * sqrt3 / 2.0
        start_x = -((count - 1) * sqrt3) / 2.0 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            y = j * 1.5
            centers.append([x, y])
    candidates.append(np.array(centers))
    
    # Arrangement: rows of 3,3,3,3 (centered)
    centers = []
    for j in range(4):
        offset = (j % 2) * sqrt3 / 2.0
        start_x = -((3 - 1) * sqrt3) / 2.0 + offset
        for i in range(3):
            x = start_x + i * sqrt3
            y = j * 1.5
            centers.append([x, y])
    candidates.append(np.array(centers))
    
    # Arrangement: rows of 1,3,5,3 (hexagonal-ish, 12 total)
    centers = []
    for j, count in enumerate([1, 3, 5, 3]):
        offset = (j % 2) * sqrt3 / 2.0
        start_x = -((count - 1) * sqrt3) / 2.0 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            y = j * 1.5
            centers.append([x, y])
    candidates.append(np.array(centers))
    
    # Arrangement: rows of 2,3,4,3 (12 total)
    centers = []
    for j, count in enumerate([2, 3, 4, 3]):
        offset = (j % 2) * sqrt3 / 2.0
        start_x = -((count - 1) * sqrt3) / 2.0 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            y = j * 1.5
            centers.append([x, y])
    candidates.append(np.array(centers))
    
    best_side = float('inf')
    best_centers = None
    best_outer_angle = 0.0
    
    for centers in candidates:
        # Center at origin
        com = centers.mean(axis=0)
        centered = centers - com
        
        for outer_angle in np.linspace(0, np.pi / 3, 25):
            side = compute_min_hex_side(centered, outer_angle)
            if side < best_side:
                best_side = side
                best_centers = centered
                best_outer_angle = outer_angle
    
    # Build result
    inner = []
    for c in best_centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
    
    outer_angle_deg = float(np.degrees(best_outer_angle))
    
    return inner, (0.0, 0.0), float(best_side), outer_angle_deg
# EVOLVE-BLOCK-END

