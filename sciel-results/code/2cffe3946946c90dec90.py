# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _hex_lattice_positions():
    """
    Generate 12 positions on a hexagonal lattice where unit hexagons don't overlap.
    Using axial coordinates: adjacent hexagons are at distance sqrt(3) apart.
    We use a cluster of 12 hexagons in a hexagonal arrangement.
    
    For unit side hexagons, the distance between centers of touching hexagons is 2*sqrt(3)/2 * sqrt(3) = 3? 
    Actually, for regular hexagons with side s, the distance between centers when they share an edge is 2*s*cos(30°) = s*sqrt(3).
    For s=1, that's sqrt(3).
    
    Let me use a simpler approach: place them in a 4-3-4 or similar pattern.
    Actually, let's use a hexagonal number arrangement.
    
    For 12 hexagons, a good arrangement is a hexagon with side 2 (which has 1+2+3+2+1=9, not 12).
    Let me use a different pattern: a 3x4 rectangular grid in hex coordinates.
    
    Actually, let me just use a known good packing:
    - Center hexagon
    - 6 surrounding it (hexagonal ring)
    - 5 more in a second partial ring
    
    Or simpler: use a parallelogram grid of hexagons.
    """
    # Use axial coordinates converted to Cartesian
    # For hexagons with side 1, center-to-center distance for edge-sharing is sqrt(3)
    d = np.sqrt(3)
    
    # Axial to Cartesian for pointy-top hexagons
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2) * r
    positions = []
    
    # A 4x3 grid in axial coordinates (q, r)
    # This gives 12 hexagons
    for r in range(-1, 2):
        for q in range(-2, 2):
            x = d * (q + r / 2.0)
            y = d * (np.sqrt(3) / 2.0) * r
            positions.append([x, y, 0.0])
    
    return np.array(positions[:12])


def _get_all_inner_vertices(inner_hex_data):
    """Get all vertices of all inner hexagons."""
    vertices = []
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, angle)
        vertices.extend(verts.tolist())
    return np.array(vertices)


def _check_separation_fast(inner_hex_data, tol=1e-6):
    """
    Check that no two inner hexagons overlap using separating axis theorem.
    For two convex polygons, they are separated if there exists an axis
    where their projections don't overlap.
    """
    n = len(inner_hex_data)
    for i in range(n):
        for j in range(i + 1, n):
            vi = _hex_vertices(*inner_hex_data[i][:2], 1.0, inner_hex_data[i][2])
            vj = _hex_vertices(*inner_hex_data[j][:2], 1.0, inner_hex_data[j][2])
            
            # Get all edge normals from both hexagons
            edges = []
            for v in [vi, vj]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k + 1) % 6]
                    edge = p2 - p1
                    # Normal (perpendicular)
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        normal /= norm
                        edges.append(normal)
            
            # Check each axis
            for axis in edges:
                proj_i = np.dot(vi, axis)
                proj_j = np.dot(vj, axis)
                
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                # Check if intervals are separated
                if max_i + tol < min_j or max_j + tol < min_i:
                    continue  # Separated on this axis, so no overlap
                else:
                    # If no axis separates them, they overlap
                    # But we need to check ALL axes - if any axis separates, they're fine
                    pass
            
            # If we got here without finding a separating axis, check more carefully
            # Actually, the SAT says: if for EVERY axis the projections overlap, then the shapes overlap
            # Let me redo this: we need to verify they DON'T overlap
            separated = False
            for axis in edges:
                proj_i = np.dot(vi, axis)
                proj_j = np.dot(vj, axis)
                
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i + tol < min_j or max_j + tol < min_i:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _min_containing_hex_side(inner_hex_data, outer_center, outer_angle_deg):
    """
    Find the minimum side length of an outer hexagon with given center and angle
    that contains all inner hexagon vertices.
    
    For a regular hexagon with center (cx,cy), side s, angle a, a point p is inside
    if for each edge, the point is on the correct side.
    
    The distance from center to each edge (apothem) is s * cos(30°) = s * sqrt(3)/2.
    So the maximum distance from center to any vertex, measured perpendicular to each edge,
    must be <= apothem.
    """
    cx, cy = outer_center
    a = np.radians(outer_angle_deg)
    
    # Edge normals for the outer hexagon
    angles = a + np.radians(60) * np.arange(6)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Get all inner vertices
    all_verts = _get_all_inner_vertices(inner_hex_data)
    
    # For each edge normal, compute the maximum projection of inner vertices
    max_proj = 0.0
    for normal in normals:
        proj = np.dot(all_verts - np.array([cx, cy]), normal)
        max_proj = max(max_proj, np.max(proj))
    
    # The apothem must be at least max_proj
    apothem = max_proj
    # apothem = s * sqrt(3)/2
    s = apothem / (np.sqrt(3) / 2.0)
    
    return s


def _objective(params, inner_hex_data_template, outer_center_template, outer_angle_template):
    """
    Objective function to minimize.
    params: flattened array of inner positions and outer parameters
    """
    # Unpack
    n_inner = 12
    # First 2*n_inner values are inner positions (x,y for each)
    # Next 2 values are outer center
    # Next 1 value is outer angle
    
    inner_xy = params[:2 * n_inner].reshape(n_inner, 2)
    outer_center = params[2 * n_inner:2 * n_inner + 2]
    outer_angle = params[2 * n_inner + 2]
    
    # Build inner hex data
    inner_hex_data = np.column_stack([inner_xy, np.zeros(n_inner)])
    
    # Check separation (fast check)
    if not _check_separation_fast(inner_hex_data):
        return 1e6
    
    # Compute minimum outer side
    s = _min_containing_hex_side(inner_hex_data, outer_center, outer_angle)
    
    if not np.isfinite(s) or s < 1.0:
        return 1e6
    
    return 1.0 / s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a hexagonal lattice as starting point and optimizes positions.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Start with hexagonal lattice positions
    inner_hex_data = _hex_lattice_positions()
    
    # Initial outer hexagon
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    s = _min_containing_hex_side(inner_hex_data, outer_center, outer_angle)
    
    best_params = np.concatenate([
        inner_hex_data[:, :2].flatten(),
        outer_center,
        [outer_angle]
    ])
    best_score = 1.0 / s
    
    # Try multiple random restarts with different initial angles
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Perturb the initial positions slightly
        perturbed_inner = inner_hex_data + rng.randn(12, 3) * 0.1
        # Keep angles small
        perturbed_inner[:, 2] = rng.randn(12) * 5
        
        # Try different outer angles
        outer_angle_try = rng.uniform(0, 60)
        outer_center_try = rng.uniform(-1, 1, size=2)
        
        # Evaluate
        s_try = _min_containing_hex_side(perturbed_inner, outer_center_try, outer_angle_try)
        
        if np.isfinite(s_try) and s_try > 0:
            score_try = 1.0 / s_try
            if score_try > best_score:
                best_score = score_try
                best_params = np.concatenate([
                    perturbed_inner[:, :2].flatten(),
                    outer_center_try,
                    [outer_angle_try]
                ])
    
    # Use scipy to refine from the best starting point
    if time.time() < deadline - 60:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(inner_hex_data, outer_center, outer_angle),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 1000}
            )
            
            if result.success or result.fun < best_score:
                refined_params = result.x
                inner_xy = refined_params[:24].reshape(12, 2)
                outer_center = refined_params[24:26]
                outer_angle = refined_params[26]
                
                inner_hex_data_refined = np.column_stack([inner_xy, np.zeros(12)])
                
                # Verify validity
                if _check_separation_fast(inner_hex_data_refined):
                    s_refined = _min_containing_hex_side(inner_hex_data_refined, outer_center, outer_angle)
                    if np.isfinite(s_refined) and s_refined > 0:
                        score_refined = 1.0 / s_refined
                        if score_refined > best_score:
                            best_score = score_refined
                            inner_hex_data = inner_hex_data_refined
                            outer_center = outer_center
                            outer_angle = outer_angle
        except Exception:
            pass
    
    # Final result
    inner_hex_data_final = np.column_stack([
        best_params[:24].reshape(12, 2),
        np.zeros(12)
    ])
    outer_center_final = best_params[24:26]
    outer_angle_final = best_params[26]
    
    s_final = _min_containing_hex_side(inner_hex_data_final, outer_center_final, outer_angle_final)
    
    outer_hex_data = np.array([outer_center_final[0], outer_center_final[1], outer_angle_final])
    
    return inner_hex_data_final, outer_hex_data, s_final


# EVOLVE-BLOCK-END
