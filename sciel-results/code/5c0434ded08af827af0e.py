# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])


def _separation_violation(t1, t2, a1=0.0, a2=0.0):
    """
    Return the maximum penetration depth (positive means overlap)
    between two unit hexagons at centers t1, t2 with angles a1, a2.
    Uses separating axis test with all 12 edge normals.
    """
    v1 = _hex_vertices(t1[0], t1[1], 1.0, a1)
    v2 = _hex_vertices(t2[0], t2[1], 1.0, a2)
    
    max_pen = 0.0
    for verts in [v1, v2]:
        for k in range(6):
            p1 = verts[k]
            p2 = verts[(k + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                pen = -gap
                if pen > max_pen:
                    max_pen = pen
    return max_pen


def _min_outer_side(translations, outer_center, outer_angle_deg):
    """
    Find minimum side length s of outer hexagon centered at outer_center
    with given angle that contains all 12 unit hexagons.
    """
    outer_a = np.radians(outer_angle_deg)
    outer_normals = np.column_stack([
        np.cos(outer_a + np.radians(60) * np.arange(6)),
        np.sin(outer_a + np.radians(60) * np.arange(6))
    ])
    
    all_vertices = []
    for i in range(12):
        tx, ty = translations[i]
        verts = _hex_vertices(tx, ty, 1.0, 0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    shifted = all_vertices - outer_center
    supports = np.max(shifted @ outer_normals.T, axis=1)
    
    s_min = 2.0 * np.max(supports) / np.sqrt(3.0)
    return s_min


def _check_validity(inner_data, outer_data, s, tol=1e-6):
    """Check if the packing is valid."""
    if s < np.sqrt(12.0):
        return False
    
    if not (np.all(np.isfinite(inner_data)) and np.all(np.isfinite(outer_data))):
        return False
    
    ox, oy, oa = outer_data
    outer_center = np.array([ox, oy])
    
    # Check inner hexagons are separated
    for i in range(12):
        for j in range(i + 1, 12):
            cx1, cy1, ca1 = inner_data[i]
            cx2, cy2, ca2 = inner_data[j]
            v1 = _hex_vertices(cx1, cy1, 1.0, ca1)
            v2 = _hex_vertices(cx2, cy2, 1.0, ca2)
            
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-15)
                    
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    gap = max(min1 - max2, min2 - max1)
                    if gap < -tol:
                        return False
    
    # Check all inner vertices are inside outer hexagon
    outer_verts = _hex_vertices(ox, oy, s, oa)
    for i in range(12):
        cx, cy, ca = inner_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ca)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    
    return True


def _objective(params, outer_angle):
    """
    Objective: minimize outer side length s, with penalty for separation violations.
    params: 26-dim array = 12 translations (24) + outer center (2)
    """
    translations = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    
    s = _min_outer_side(translations, outer_center, outer_angle)
    
    # Separation penalty
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            pen = _separation_violation(translations[i], translations[j])
            if pen > 0:
                penalty += pen ** 2
    
    return s + 100.0 * penalty


def _initial_hex_cluster():
    """
    Generate initial positions for 12 unit hexagons in a hexagonal cluster.
    Hexagons with side 1 have width 2 (flat-to-flat) and 2*sqrt(3)/2*... 
    Actually for a regular hexagon with side 1:
    - Width (vertex to opposite vertex) = 2
    - Flat-to-flat distance = sqrt(3)
    
    For edge-to-edge packing of unit hexagons:
    - Center-to-center distance for adjacent hexagons sharing an edge = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    Wait, that's for the apothem. The distance between centers of two adjacent hexagons
    that share an edge is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    
    Actually let me reconsider. A regular hexagon with side length 1:
    - Apothem (center to edge midpoint) = sqrt(3)/2
    - Distance between parallel edges = 2 * sqrt(3)/2 = sqrt(3)
    - So center-to-center for adjacent hexagons = sqrt(3) ≈ 1.732
    
    But wait, the task says "unit regular hexagons" with side 1.
    The hexagons must be separated (non-overlapping).
    
    For a hexagonal lattice of unit hexagons:
    - Nearest neighbor distance = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    """
    # Hexagonal lattice positions for 12 hexagons
    # Using axial coordinates, convert to cartesian
    # In a hex grid with spacing sqrt(3) between centers:
    d = np.sqrt(3.0)  # center-to-center distance for adjacent hexagons
    
    # Pattern: 1 center, 6 in first ring, 5 in second ring
    positions = []
    positions.append([0, 0])  # center
    
    # First ring: 6 hexagons at distance d
    for i in range(6):
        angle = np.radians(60 * i)
        positions.append([d * np.cos(angle), d * np.sin(angle)])
    
    # Second ring: pick 5 positions
    # Second ring positions are at distance 2d or d*sqrt(3)
    # In hex grid, second ring has 12 positions, we need 5
    # Let's use the "outer" ones that form a nice shape
    
    # Actually, for a compact cluster of 12, a common pattern is:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    # Let me use a different approach: hexagonal numbers
    # 1 + 6 + 5 = 12 (center + first ring + partial second ring)
    
    # Second ring partial: take 5 of the 12 second-ring positions
    # Second ring positions in axial coords: (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (1,-2), (0,-3), (-1,-2)
    # Convert axial to pixel: x = d * (q + r/2), y = d * (sqrt(3)/2) * r
    
    second_ring_axial = [
        (2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0),
        (-1, -1), (0, -2), (1, -1), (2, -1), (1, -2), (0, -3), (-1, -2)
    ]
    
    # Pick 5 that make a roughly symmetric shape
    # Let's take: (2,0), (1,1), (0,2), (-2,0), (0,-2) -- not symmetric
    # Better: (2,0), (-2,0), (1,1), (-1,-1), (0,2) -- still not great
    
    # For symmetry, let's take 6 from second ring to make 13, then remove one
    # Or use a different initial arrangement
    
    # Let me just use a simple rectangular-ish grid and let optimization fix it
    positions = []
    # 4 rows of 3
    for row in range(4):
        for col in range(3):
            x = col * d
            y = row * d * np.sqrt(3) / 2
            if row % 2 == 1:
                x += d / 2
            positions.append([x, y])
    
    return np.array(positions[:12])


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length using convex optimization.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580 second limit (safety margin)
    
    def elapsed():
        return time.time() - start_time
    
    # Initial positions
    init_translations = _initial_hex_cluster()
    
    best_s = np.inf
    best_translations = init_translations.copy()
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    best_penalty = np.inf
    
    # Try a few outer angles
    outer_angles_to_try = [0.0, 30.0]
    
    for outer_angle in outer_angles_to_try:
        if elapsed() > deadline - 60.0:
            break
        
        # First, try differential evolution for global search
        if elapsed() < deadline - 120.0:
            bounds = [(-10, 10)] * 26
            try:
                result_de = differential_evolution(
                    _objective,
                    bounds,
                    args=(outer_angle,),
                    maxiter=100,
                    popsize=20,
                    tol=1e-4,
                    seed=42,
                    polish=False,
                    updating='deferred'
                )
                if result_de.fun < best_s:
                    best_s = result_de.fun
                    best_translations = result_de.x[:24].reshape(12, 2)
                    best_outer_center = result_de.x[24:26]
                    best_outer_angle = outer_angle
            except Exception:
                pass
        
        # Then refine with L-BFGS-B
        if elapsed() < deadline - 30.0:
            x0 = np.zeros(26)
            x0[:24] = best_translations.flatten()
            x0[24:26] = best_outer_center
            
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(outer_angle,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-10}
                )
                
                if result.fun < best_s:
                    best_s = result.fun
                    best_translations = result.x[:24].reshape(12, 2)
                    best_outer_center = result.x[24:26]
                    best_outer_angle = outer_angle
            except Exception:
                pass
        
        # Local refinement: push apart overlapping hexagons
        if elapsed() < deadline - 10.0:
            for iteration in range(200):
                if elapsed() > deadline - 5.0:
                    break
                
                pushed = False
                for i in range(12):
                    for j in range(i + 1, 12):
                        pen = _separation_violation(best_translations[i], best_translations[j])
                        if pen > 1e-8:
                            # Push apart along center-to-center vector
                            diff = best_translations[j] - best_translations[i]
                            norm = np.linalg.norm(diff)
                            if norm < 1e-15:
                                diff = np.array([1.0, 0.0])
                                norm = 1.0
                            diff = diff / norm
                            push = min(pen, 0.05)
                            best_translations[i] -= push * diff
                            best_translations[j] += push * diff
                            pushed = True
                            break
                    if pushed:
                        break
                
                if not pushed:
                    break
                
                # Recompute s
                s_new = _min_outer_side(best_translations, best_outer_center, best_outer_angle)
                if s_new < best_s:
                    best_s = s_new
        
        # Final polish with tighter optimization
        if elapsed() < deadline - 5.0:
            x0 = np.zeros(26)
            x0[:24] = best_translations.flatten()
            x0[24:26] = best_outer_center
            
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(best_outer_angle,),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-12}
                )
                
                if result.fun < best_s:
                    best_s = result.fun
                    best_translations = result.x[:24].reshape(12, 2)
                    best_outer_center = result.x[24:26]
                    best_outer_angle = best_outer_angle
            except Exception:
                pass
    
    # Compute final s
    best_s = _min_outer_side(best_translations, best_outer_center, best_outer_angle)
    
    # Build result arrays
    inner_data = np.column_stack([
        best_translations[:, 0],
        best_translations[:, 1],
        np.zeros(12)
    ])
    outer_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Validate
    if not _check_validity(inner_data, outer_data, best_s):
        # Fallback to seed solution
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return inner_data, outer_data, best_s


# EVOLVE-BLOCK-END
