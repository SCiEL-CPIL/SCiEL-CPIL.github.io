# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon (centered at outer_center,
    with the given angle) that contains all inner unit hexagons.
    
    Returns (s_min, valid) where valid is True if the configuration is feasible
    (no overlaps and all inner vertices can fit).
    """
    ov_cx, ov_cy = outer_center[0], outer_center[1]
    ov_a = np.deg2rad(outer_angle_deg)
    
    # The 6 outward normals to the sides of the outer hexagon
    # For a hexagon with vertices at angles (a + 60*i), the side normals point at (a + 30 + 60*i)
    normal_angles = ov_a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For each inner hexagon, find the maximum projection onto each normal
    max_projections = np.zeros(6)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        # Translate to outer center frame
        hv_rel = hv - np.array([ov_cx, ov_cy])
        projections = hv_rel @ normals.T  # shape (6, 6)
        max_projections = np.maximum(max_projections, np.max(projections, axis=0))
    
    # The distance from center to each side is s * sqrt(3)/2
    # So we need s * sqrt(3)/2 >= max_projection for all 6 directions
    s_min = np.max(max_projections) / (np.sqrt(3.0) / 2.0)
    
    return s_min


def _check_separation(inner):
    """
    Check if all pairs of inner hexagons are separated (non-overlapping) using SAT.
    Returns True if all pairs are separated.
    """
    n = len(inner)
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            # Get all edge normals from both hexagons
            edges = []
            for k in range(6):
                edges.append(v1[(k + 1) % 6] - v1[k])
                edges.append(v2[(k + 1) % 6] - v2[k])
            
            separated = False
            for edge in edges:
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                if gap1 > 1e-6 or gap2 > 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _objective(params, outer_center, outer_angle_deg):
    """
    Objective function: minimize the containing hexagon side length.
    params: 24 values = 12 * (x, y) for the 12 inner hexagons (angles fixed at 0)
    """
    inner = np.zeros((12, 3))
    inner[:, 0] = params[0::2]
    inner[:, 1] = params[1::2]
    inner[:, 2] = 0.0  # Fixed angles
    
    # Check separation
    if not _check_separation(inner):
        # Penalize overlaps heavily
        return 1000.0
    
    s = _min_containing_hex_side(inner, outer_center, outer_angle_deg)
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s limit
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    # Initial configuration: symmetric hexagonal arrangement
    # For unit hexagons, adjacent centers must be at distance 2*sqrt(3) apart
    # (the distance from center to a vertex is 1, so two touching hexagons
    #  have centers 2*1 = 2 apart along the normal to a shared edge,
    #  which corresponds to 2*sqrt(3) in the hexagonal lattice)
    
    # Actually, for regular hexagons with side 1, the distance between
    # centers of two adjacent (edge-sharing) hexagons is 2*sqrt(3)/2 * 2 = 2*sqrt(3)
    # Wait: the apothem (center to edge midpoint) is sqrt(3)/2, so two adjacent
    # hexagons sharing an edge have centers 2 * sqrt(3)/2 = sqrt(3) apart.
    # But that would mean they overlap! Let me reconsider.
    
    # For two unit hexagons to be non-overlapping and just touching:
    # The minimum distance between centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    # No wait, that's for sharing a vertex. For sharing an edge, it's 2 * apothem = sqrt(3).
    # Actually no: if two hexagons share an edge, their centers are 2*apothem apart = sqrt(3).
    # But then the shared edge would be at distance apothem from each center, so they'd overlap.
    
    # Let me think again: a unit regular hexagon has inradius (apothem) = sqrt(3)/2.
    # Two hexagons that share an edge: the edge is at distance sqrt(3)/2 from each center,
    # so the centers are 2*(sqrt(3)/2) = sqrt(3) apart. This is the minimum distance
    # for edge-sharing (touching) hexagons.
    
    # So d = sqrt(3) is correct for touching hexagons. But we need them to be separated
    # by more than 1e-6, so we should use d slightly larger than sqrt(3).
    
    d = np.sqrt(3.0) + 1e-4  # Slightly more than touching distance
    
    # 12 hexagons: center + ring of 6 + 5 more on outer ring
    # Hexagonal lattice positions (using axial coordinates converted to cartesian)
    # For a hexagon with "radius" r (in hex units), positions are:
    # Ring 0: (0,0)
    # Ring 1: 6 positions at distance d
    # Ring 2: 12 positions at distance 2d (we pick 5 of them)
    
    # Ring 1 positions (6 hexagons around center):
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    ring1 = []
    for i in range(6):
        angle = np.deg2rad(60.0 * i)
        ring1.append([d * np.cos(angle), d * np.sin(angle)])
    
    # Ring 2 positions (we need 5 more, pick from the 12 available)
    # Ring 2 has positions at:
    # (2d, 0), (d, d*sqrt(3)), (-d, d*sqrt(3)), (-2d, 0), (-d, -d*sqrt(3)), (d, -d*sqrt(3))
    # and 6 more at intermediate positions
    ring2_candidates = []
    for i in range(6):
        angle = np.deg2rad(60.0 * i)
        # Outer vertices of ring 2
        ring2_candidates.append([2*d * np.cos(angle), 2*d * np.sin(angle)])
        # Midpoints of ring 2 (these are at distance sqrt(3)*d from center)
        angle_mid = np.deg2rad(60.0 * i + 30)
        r_mid = d * np.sqrt(3)
        ring2_candidates.append([r_mid * np.cos(angle_mid), r_mid * np.sin(angle_mid)])
    
    # Pick 5 from ring 2 that give good symmetry
    # Use the 6 "outer vertex" positions and drop one for symmetry
    selected_ring2 = ring2_candidates[:6][:5]  # First 5 outer vertices
    
    initial_params = np.array([
        0.0, 0.0,          # center
    ] + [x for pos in ring1 for x in pos] + [x for pos in selected_ring2 for x in pos])
    
    # Verify initial configuration is valid
    inner_init = np.zeros((12, 3))
    inner_init[:, 0] = initial_params[0::2]
    inner_init[:, 1] = initial_params[1::2]
    
    if not _check_separation(inner_init):
        # Fallback: use a linear arrangement that's definitely valid
        # Place hexagons in a row with spacing 2*sqrt(3)
        spacing = 2.0 * np.sqrt(3.0) + 1e-3
        initial_params = np.array([])
        for i in range(12):
            initial_params = np.append(initial_params, [i * spacing, 0.0])
        
        inner_init = np.zeros((12, 3))
        inner_init[:, 0] = initial_params[0::2]
        inner_init[:, 1] = initial_params[1::2]
    
    best_params = initial_params.copy()
    best_s = _min_containing_hex_side(inner_init, outer_center, outer_angle)
    
    # Use L-BFGS-B for faster convergence with bounds
    # First, let's try a gradient-free method with better initial conditions
    
    # Try multiple optimization strategies
    strategies = [
        ('Nelder-Mead', {'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}),
        ('Powell', {'maxiter': 2000, 'ftol': 1e-8, 'xtol': 1e-8}),
    ]
    
    for method_name, options in strategies:
        if time.time() - start_time > deadline - 120.0:
            break
        try:
            result = minimize(
                _objective,
                best_params,
                args=(outer_center, outer_angle),
                method=method_name,
                options=options
            )
            
            if result.fun < best_s:
                best_params = result.x
                best_s = result.fun
                
        except Exception as e:
            pass
    
    # Try perturbed starts if time remains
    elapsed = time.time() - start_time
    if elapsed < deadline - 100.0:
        try:
            np.random.seed(42)
            for trial in range(5):
                if time.time() - start_time > deadline - 60.0:
                    break
                
                perturbed = initial_params + np.random.normal(0, 0.3, size=initial_params.shape)
                result2 = minimize(
                    _objective,
                    perturbed,
                    args=(outer_center, outer_angle),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if result2.fun < best_s:
                    best_params = result2.x
                    best_s = result2.fun
                    
        except Exception as e:
            pass
    
    # Construct final inner hexagons
    inner = np.zeros((12, 3))
    inner[:, 0] = best_params[0::2]
    inner[:, 1] = best_params[1::2]
    inner[:, 2] = 0.0
    
    # Compute final s with a small margin
    s_final = _min_containing_hex_side(inner, outer_center, outer_angle) + 1e-4
    
    # Verify validity one more time
    if not _check_separation(inner):
        # Fallback to a definitely valid configuration: linear row
        spacing = 2.0 * np.sqrt(3.0) + 1e-3
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = i * spacing
            inner[i, 1] = 0.0
            inner[i, 2] = 0.0
        s_final = _min_containing_hex_side(inner, outer_center, outer_angle) + 1e-4
    
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, s_final


# EVOLVE-BLOCK-END
