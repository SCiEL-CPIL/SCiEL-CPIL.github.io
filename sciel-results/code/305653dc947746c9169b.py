import math
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 140.0
    safety_margin = 10.0

    d = math.sqrt(3)

    # Define the rigid cluster: center + ring 1 (6) + ring 2 (4 selected)
    # Using hexagonal lattice with spacing 2*sqrt(3) for touching unit hexagons
    spacing = 2 * d
    
    # Center
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 hexagons at distance spacing
    for i in range(6):
        angle = i * math.pi / 3
        x = spacing * math.cos(angle)
        y = spacing * math.sin(angle)
        centers.append((x, y))
    
    # Ring 2: 12 positions, select 4 for symmetry
    # Axial coordinates for ring 2: (q,r) with max(|q|,|r|,|q+r|)=2
    ring2_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = spacing * (q + r / 2.0)
                y = spacing * (d / 2.0) * r
                ring2_positions.append((x, y))
    
    # Select 4 ring-2 positions that are roughly symmetric
    # ring2_positions has 12 entries; pick indices 0, 3, 6, 9 for 90-degree symmetry
    ring2_selected = [ring2_positions[i] for i in [0, 3, 6, 9]]
    centers.extend(ring2_selected)
    
    assert len(centers) == N, f"Expected {N} centers, got {len(centers)}"
    
    # Unit hexagon vertices (flat-top orientation, centered at origin)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
    def get_cluster_vertices(rotation_angle):
        """Get all vertices of the rigid cluster rotated by rotation_angle around origin."""
        c = math.cos(rotation_angle)
        s = math.sin(rotation_angle)
        
        all_verts = []
        for cx, cy in centers:
            # Rotate the center
            rx = cx * c - cy * s
            ry = cx * s + cy * c
            
            # Rotate the local vertices
            rvx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
            rvy = verts_rel[:, 0] * s + verts_rel[:, 1] * c
            
            # Translate to rotated center
            verts = np.column_stack([rvx + rx, rvy + ry])
            all_verts.append(verts)
        
        return np.vstack(all_verts)
    
    def compute_outer_side(vertices, outer_alpha=0.0):
        """Compute the minimum side length of a regular hexagon enclosing the vertices."""
        centroid = np.mean(vertices, axis=0)
        centered = vertices - centroid
        
        # Side normals at angles outer_alpha + k*pi/3 for k=0..5
        k_angles = outer_alpha + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        
        # Projection extents
        dots = np.sum(centered[np.newaxis, :, :] * normals[:, np.newaxis, :], axis=-1)
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)
        
        # Side length S: distance between opposite sides = S * sqrt(3)
        return max_extent / math.sqrt(3)
    
    def objective(alpha):
        """Total objective: outer side length for cluster rotated by alpha."""
        if time.time() > deadline - safety_margin:
            return 1e10
        
        verts = get_cluster_vertices(alpha)
        return compute_outer_side(verts)
    
    # Optimize the global rotation angle
    best_S = float('inf')
    best_alpha = 0.0
    
    # Try multiple initial guesses for the rotation
    initial_angles = [i * math.pi / 18 for i in range(6)]  # 0, 10, 20, 30, 40, 50 degrees
    
    for init_alpha in initial_angles:
        if time.time() > deadline - safety_margin:
            break
        
        try:
            res = minimize_scalar(objective, bounds=(0, math.pi / 3), method='bounded',
                                   options={'xatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_alpha = res.x
        except Exception:
            continue
    
    # If no optimization succeeded, use the best initial guess
    if best_S == float('inf'):
        for init_alpha in initial_angles:
            S = objective(init_alpha)
            if S < best_S:
                best_S = S
                best_alpha = init_alpha
    
    # Construct the inner hexagons
    inner = []
    for cx, cy in centers:
        # Rotate the center by best_alpha
        c = math.cos(best_alpha)
        s = math.sin(best_alpha)
        rx = cx * c - cy * s
        ry = cx * s + cy * c
        inner.append([rx, ry, math.degrees(best_alpha)])
    
    return inner, (0.0, 0.0), best_S, math.degrees(best_alpha)
