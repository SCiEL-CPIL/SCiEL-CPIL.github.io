import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

def _get_hex_vertices(cx, cy, d, angle_deg):
    """Get vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _compute_s_for_centers(centers):
    """Compute minimum s for outer hexagon (centered at origin, angle 0) to contain all inner hexagons."""
    # Normals for the 6 sides of the outer hexagon
    # Vertices of unit hexagon at angles 0, 60, ...
    # Side normals are at 30, 90, ...
    normal_angles = np.pi/6 + np.arange(6) * np.pi/3
    normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
    vertex_angles = np.arange(6) * np.pi/3
    unit_vertices = np.array([[np.cos(a), np.sin(a)] for a in vertex_angles])
    
    max_proj = 0.0
    for c in centers:
        for u in unit_vertices:
            v = c + u
            # Project vertex v onto all normals
            projs = normals @ v
            proj = np.max(projs)
            if proj > max_proj:
                max_proj = proj
    
    # The distance from center to side is s * cos(pi/6) = s * sqrt(3)/2
    # So s = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
    s = 2.0 * max_proj / np.sqrt(3)
    return s

def _check_separation(centers, tol=1e-6):
    """Check that no two inner hexagons overlap using SAT."""
    vertex_angles = np.arange(6) * np.pi/3
    unit_vertices = np.array([[np.cos(a), np.sin(a)] for a in vertex_angles])
    
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            vi = centers[i] + unit_vertices
            vj = centers[j] + unit_vertices
            # Edge normals for hexagon i
            edges_i = np.roll(vi, -1, axis=0) - vi
            for edge in edges_i:
                normal = np.array([-edge[1], edge[0]])
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-15:
                    continue
                normal = normal / norm_len
                proj_i = np.dot(vi, normal)
                proj_j = np.dot(vj, normal)
                gap = min(proj_i.max(), proj_j.max()) - max(proj_i.min(), proj_j.min())
                if gap < -tol:
                    return False
    return True

def _check_inside(centers, s, tol=1e-6):
    """Check all inner hexagon vertices are inside outer hexagon."""
    outer_verts = _get_hex_vertices(0, 0, s, 0)
    vertex_angles = np.arange(6) * np.pi/3
    unit_vertices = np.array([[np.cos(a), np.sin(a)] for a in vertex_angles])
    
    for c in centers:
        for u in unit_vertices:
            v = c + u
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -tol:
                    return False
    return True

def _is_valid_packing(centers, s):
    """Full validity check."""
    if not np.all(np.isfinite(centers)):
        return False
    if s < np.sqrt(12) - 1e-10:
        return False
    if not _check_separation(centers):
        return False
    if not _check_inside(centers, s):
        return False
    return True

def _get_initial_grid():
    """Generate a valid initial packing using a hexagonal lattice subset."""
    sqrt3 = np.sqrt(3)
    u1 = np.array([1.5, sqrt3 / 2.0])
    u2 = np.array([-1.5, sqrt3 / 2.0])
    
    # A 3x4 grid of hexagons
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * u1 + j * u2
            centers.append(c)
    centers = np.array(centers)
    centers -= centers.mean(axis=0)
    return centers

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Initialize with a known valid grid
    best_centers = _get_initial_grid()
    best_s = _compute_s_for_centers(best_centers)
    
    # If the grid is invalid (shouldn't happen, but just in case), try a circle
    if not _check_separation(best_centers):
        # Fallback: place them in a circle
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        radius = 3.0
        best_centers = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        best_s = _compute_s_for_centers(best_centers)

    current_centers = best_centers.copy()
    current_s = best_s
    
    # Simulated Annealing Parameters
    T = 0.1  # Initial temperature
    cooling_rate = 0.9995
    min_T = 1e-4
    max_steps = 20000
    
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Generate a random move
        # Pick a random hexagon and move it
        idx = rng.integers(0, 12)
        # Random displacement
        displacement = rng.normal(0, 0.2, size=2)
        
        new_centers = current_centers.copy()
        new_centers[idx] += displacement
        
        # Recenter to keep the outer hexagon centered at origin
        new_centers -= new_centers.mean(axis=0)
        
        # Calculate new s
        new_s = _compute_s_for_centers(new_centers)
        
        # Check separation
        if not _check_separation(new_centers, tol=1e-6):
            # If separation is violated, reject immediately (or accept with very low prob)
            # For simplicity and robustness, reject if hard constraint violated
            continue
        
        # Check if inside is guaranteed by s calculation?
        # _compute_s_for_centers ensures containment by definition of s.
        # So we only need to check separation and finiteness.
        
        delta = new_s - current_s
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_centers = new_centers
            current_s = new_s
            
            if current_s < best_s:
                best_s = current_s
                best_centers = current_centers.copy()
        
        T *= cooling_rate
        
        if T < min_T:
            # Reheat if stuck? Or just break.
            # Let's try a simple reheat if we haven't improved in a while
            if step % 500 == 0 and step > 0:
                # Check if we improved recently
                # For now, just keep going with low T
                pass

    # Final validation
    if not _is_valid_packing(best_centers, best_s):
        # If something went wrong, return the initial grid
        best_centers = _get_initial_grid()
        best_s = _compute_s_for_centers(best_centers)
        
    # Construct output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s
