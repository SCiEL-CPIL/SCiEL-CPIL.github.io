import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the inner hexagon centers in a 3x4 grid on the hexagonal lattice
    # Hex lattice basis vectors for side length 1:
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 1.5)
    
    centers = []
    for u in range(3):
        for v in range(4):
            x = u * np.sqrt(3) + v * (np.sqrt(3) / 2)
            y = v * 1.5
            centers.append((x, y))
    
    # Generate all vertices of the inner hexagons
    # Each hexagon has 6 vertices at distance 1 from its center
    # The vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # relative to the hexagon's local frame.
    # Since we assume the hexagons are aligned with the lattice, and we will rotate
    # the entire cluster, the local frame rotates with the cluster.
    # So we just need the vertices relative to the center in the un-rotated frame.
    
    angles = np.arange(6) * np.pi / 3
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    points = []
    for cx, cy in centers:
        for ca, sa in zip(cos_angles, sin_angles):
            points.append((cx + ca, cy + sa))
    
    points = np.array(points)
    
    # Function to calculate the minimum side length of the enclosing hexagon
    # given a rotation psi for the outer hexagon.
    # The outer hexagon is centered at (0,0).
    # Condition for a point (x,y) to be inside a regular hexagon of apothem A
    # rotated by psi:
    # max_{k=0..5} (x cos(psi + k*pi/3) + y sin(psi + k*pi/3)) <= A
    # Side length S = 2 * A / sqrt(3)
    
    def calc_side_length(psi):
        # Normals of the outer hexagon sides
        normal_angles = psi + np.arange(6) * np.pi / 3
        cos_norms = np.cos(normal_angles)
        sin_norms = np.sin(normal_angles)
        
        # Project all points onto all normals
        # points: (M, 2)
        # cos_norms: (6,)
        # We want max over k of (x_k cos_k + y_k sin_k) for each point?
        # No, we want max over ALL points and ALL normals.
        # proj[i, k] = points[i, 0] * cos_norms[k] + points[i, 1] * sin_norms[k]
        # A = max(proj)
        
        # Efficient calculation
        # x = points[:, 0]
        # y = points[:, 1]
        # proj = x[:, None] * cos_norms[None, :] + y[:, None] * sin_norms[None, :]
        # A = np.max(proj)
        
        # Since we just need the max, we can compute it directly.
        # Broadcasting might be memory heavy if M is large, but M=72, K=6, so 432 floats. Trivial.
        
        proj = points[:, 0:1] * cos_norms[None, :] + points[:, 1:2] * sin_norms[None, :]
        A = np.max(proj)
        S = 2.0 * A / np.sqrt(3)
        return S

    # Optimize psi in [0, pi/3)
    # The function is periodic with period pi/3.
    # It is unimodal in this interval? Likely, as it's the max of sinusoids.
    # We can use minimize_scalar.
    
    result = minimize_scalar(calc_side_length, bounds=(0, np.pi/3), method='bounded')
    optimal_psi = result.x
    min_side = result.fun
    
    # The inner hexagons are rotated by optimal_psi as well
    inner_angle_deg = np.degrees(optimal_psi)
    
    # Construct the inner list: [x, y, angle_degrees]
    # Note: The centers (x,y) are in the un-rotated frame.
    # But the return format asks for "inner rows (x, y, angle_degrees)".
    # Does (x,y) refer to the position in the global frame or local?
    # "inner rows (x, y, angle_degrees) of n unit hexagons"
    # Usually, x,y are global coordinates.
    # If we rotate the cluster, the centers' global coordinates change.
    # The centers in the un-rotated frame are `centers`.
    # To get global coordinates, we must rotate them by optimal_psi.
    
    cos_psi = np.cos(optimal_psi)
    sin_psi = np.sin(optimal_psi)
    
    inner = []
    for cx, cy in centers:
        # Rotate center (cx, cy) by optimal_psi
        gx = cx * cos_psi - cy * sin_psi
        gy = cx * sin_psi + cy * cos_psi
        inner.append([gx, gy, inner_angle_deg])
        
    # Outer hexagon center is (0,0)
    # Outer side length is min_side
    # Outer rotation is optimal_psi
    
    return inner, (0.0, 0.0), min_side, np.degrees(optimal_psi)
# EVOLVE-BLOCK-END
