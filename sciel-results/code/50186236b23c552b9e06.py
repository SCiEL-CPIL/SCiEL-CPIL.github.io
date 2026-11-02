import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Hexagonal lattice: unit hexagons with side length 1
    # Distance between adjacent centers = sqrt(3)
    # Using axial coordinates, convert to Cartesian for flat-top hexagons
    
    # 11 hexagons in hexagonal spiral: center + 6 neighbors + 4 outer
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (-1, 2)
    ]
    
    def axial_to_cartesian(q, r):
        """Flat-top hexagon axial to Cartesian: x = sqrt(3)*(q + r/2), y = 3/2*r"""
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    base_centers = np.array([axial_to_cartesian(q, r) for q, r in axial_coords])
    
    # For a regular hexagon with side length s, centered at origin, with rotation theta:
    # The 6 outward unit normals are at angles theta + k*pi/3 for k=0..5
    # The inradius (distance from center to each side) is s * sqrt(3)/2
    # A point p is inside iff for all 6 normals n_i: dot(p, n_i) <= s * sqrt(3)/2
    
    # For an inner hexagon centered at c with side 1 and rotation phi:
    # Its vertices are at c + cos(phi + k*pi/3), c + sin(phi + k*pi/3) for k=0..5
    # But for the enclosing constraint, we can use the support function of the inner hexagon:
    # max over its vertices of dot(vertex, n) = dot(c, n) + support_of_unit_hex(n, phi)
    # where support_of_unit_hex(n, phi) = max_k dot(e_k + phi, n) where e_k are unit hex vertices
    
    # For a unit regular hexagon (circumradius 1), the support function in direction u is:
    # h(u) = max_{k=0..5} cos(angle(u) - (phi + k*pi/3))
    # This equals cos(min_k |angle(u) - phi - k*pi/3| mod 2pi, adjusted)
    # Actually for a regular hexagon with circumradius 1, the support function is:
    # h(u) = cos(d) where d is the angular distance from u to the nearest vertex direction
    
    # Simpler: just compute all 6 vertices of each inner hexagon and check all 6*11=66 points
    
    def compute_inner_vertices(centers, inner_angle_deg):
        """Compute all vertices of all inner hexagons"""
        phi = np.radians(inner_angle_deg)
        vertex_angles = phi + np.arange(6) * np.pi / 3.0
        vertices_per_hex = np.column_stack([np.cos(vertex_angles), np.sin(vertex_angles)])
        # Each center gets 6 vertices
        all_vertices = []
        for c in centers:
            verts = c + vertices_per_hex
            all_vertices.append(verts)
        return np.vstack(all_vertices)  # shape (66, 2)
    
    def enclosing_side_length(all_vertices, outer_rotation_deg):
        """Compute minimal side length of regular hexagon (centered at origin) enclosing all vertices"""
        theta = np.radians(outer_rotation_deg)
        normal_angles = theta + np.arange(6) * np.pi / 3.0
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Max projection of each vertex onto each normal
        projections = all_vertices @ normals.T  # (66, 6)
        max_proj = np.max(projections, axis=0)  # (6,)
        
        # Constraint: max_proj[i] <= s * sqrt(3)/2 for all i
        # So s >= 2/sqrt(3) * max(max_proj)
        side = 2.0 / np.sqrt(3) * np.max(max_proj)
        return side
    
    def objective(params):
        """Minimize enclosing hexagon side length"""
        tx, ty, outer_rot, inner_rot = params
        shifted_centers = base_centers + np.array([tx, ty])
        all_vertices = compute_inner_vertices(shifted_centers, inner_rot)
        return enclosing_side_length(all_vertices, outer_rot)
    
    # Initial guesses
    com = np.mean(base_centers, axis=0)
    initial_guesses = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 30.0, 0.0),
        (0.0, 0.0, 0.0, 30.0),
        (0.0, 0.0, 30.0, 30.0),
        (0.1, 0.1, 0.0, 0.0),
        (-0.1, -0.1, 30.0, 0.0),
        (0.0, 0.0, 15.0, 0.0),
        (0.0, 0.0, 45.0, 0.0),
    ]
    
    best_side = float('inf')
    best_params = None
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, guess, method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 3000, 'maxfev': 3000})
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # If optimization didn't find anything good, use a simple placement
    if best_params is None or best_side > 10.0:
        # Just place at origin with outer rotation 0
        all_vertices = compute_inner_vertices(base_centers, 0.0)
        best_side = enclosing_side_length(all_vertices, 0.0)
        best_params = np.array([0.0, 0.0, 0.0, 0.0])
    
    tx, ty, outer_rot, inner_rot = best_params
    
    # Construct inner hexagon positions
    inner = []
    for cx, cy in base_centers:
        inner.append([float(cx + tx), float(cy + ty), float(inner_rot)])
    
    outer_center = (0.0, 0.0)
    side = float(best_side)
    angle = float(outer_rot)
    
    return inner, outer_center, side, angle

# EVOLVE-BLOCK-END
