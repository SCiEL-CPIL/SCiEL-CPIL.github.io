# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all):
    """
    Given array of all inner hexagon vertices (shape (72,2)), compute minimum outer hex side s
    (centered at origin, angle 0) that contains all vertices.
    s_min = (2/sqrt(3)) * max over all vertices v of max over 6 inward normals n_k of (n_k . v)
    """
    # Inward normals for outer hex with vertices at 0,60,...,300 degrees
    # Edge k is between vertex k and vertex k+1 (mod 6)
    # Inward normal direction: bisector of the two outward normals, pointing inward
    # For a regular hexagon, the inward normal to edge k points toward center from edge midpoint
    # Edge k midpoint is at angle (k+0.5)*60 deg from center, so inward normal is at angle (k+0.5)*60 + 180 deg
    # Actually: vertex k at angle k*60. Edge from vertex k to vertex k+1.
    # The midpoint of this edge is at distance s*sqrt(3)/2 from center, at angle (k+0.5)*60.
    # The inward normal points from edge toward center, i.e., direction = -midpoint_direction
    # So inward normal n_k = -[cos((k+0.5)*60), sin((k+0.5)*60)]
    # But we want n_k . v <= apothem for containment, where n_k points inward.
    # Let me redefine: let m_k = [cos((k+0.5)*60), sin((k+0.5)*60)] be the outward normal to edge k.
    # Then v is inside iff m_k . v <= s*sqrt(3)/2 for all k.
    # So s >= (2/sqrt(3)) * max_k (m_k . v)
    
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3  # 30, 90, 150, 210, 270, 330 degrees
    m = np.column_stack([np.cos(angles), np.sin(angles)])  # shape (6,2)
    
    # For each vertex v, compute max_k (m_k . v)
    # vertices_all shape (N, 2), m shape (6,2)
    # dot products: (N, 6)
    dots = vertices_all @ m.T  # shape (N, 6)
    max_dot = np.max(dots, axis=1)  # shape (N,)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using SAT.
    Returns True if no overlap, False otherwise.
    centers: (12,2), angles_deg: (12,)
    """
    n = len(centers)
    vertices_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            # SAT: check all 12 edge normals (6 from each hexagon)
            overlap = True
            for hex_idx, verts in enumerate([v1, v2]):
                for e in range(6):
                    p1 = verts[e]
                    p2 = verts[(e+1) % 6]
                    edge = p2 - p1
                    # Normal (perpendicular, pointing outward for this hexagon)
                    # For a CCW polygon, outward normal is (edge_y, -edge_x) normalized... 
                    # Actually let's just use the edge normal direction
                    nx, ny = edge[1], -edge[0]
                    # Project both hexagons onto this normal
                    proj1 = v1 @ np.array([nx, ny])
                    proj2 = v2 @ np.array([nx, ny])
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    # Check if projections are separated
                    if max1 + tol < min2 or max2 + tol < min1:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    return True

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all 72 vertices of 12 hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _objective(params, side=1.0):
    """
    Objective: minimize outer hex side s, with penalty for overlaps.
    params: (24,) = [x1,y1,theta1, x2,y2,theta2, ..., x12,y12,theta12]
    """
    centers = params[0::3]  # x coords... wait, let me restructure
    # Actually let's use params as (12, 3) flattened
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    # Compute min outer side
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    s = _min_outer_side(all_verts)
    
    # Check non-overlap
    if not _check_nonoverlap(centers, angles_deg, side):
        # Penalty for overlap
        # Compute minimum distance between closest pairs
        # Simple penalty: large value
        return s + 100.0
    
    return s

def _initial_honeycomb():
    """
    Create initial honeycomb-like arrangement of 12 hexagons.
    Hexagon with side 1: distance between adjacent centers in honeycomb = sqrt(3) ≈ 1.732
    for flat-top hexagons, or 2 for pointy-top... 
    
    For regular hexagons with side 1, the distance between centers of two non-overlapping
    adjacent hexagons sharing an edge is sqrt(3) * 1 = sqrt(3) ≈ 1.732.
    
    Let's use a centered arrangement:
    - 1 at center
    - 6 around it at distance sqrt(3)
    - 5 more around the outer ring
    
    For flat-top hexagons (angle=0), the 6 neighbors are at angles 0,60,120,180,240,300
    at distance sqrt(3).
    """
    d = np.sqrt(3.0)  # distance between adjacent hex centers
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Inner ring: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    # Outer ring: 5 more hexagons (to make 12 total)
    # Place them at the "gaps" of the outer ring, further out
    # The outer ring hexagons are at distance d from center.
    # Next ring would be at distance 2*d, but we only need 5 more.
    # Place them at angles 30, 90, 150, 210, 270 (between the inner ring hexagons)
    # at distance 2*d * cos(30) ... actually let me think.
    
    # In a honeycomb, the second ring has 12 hexagons. We need 5 of them.
    # Place them symmetrically. Let's put them at angles 0, 72, 144, 216, 288 at distance 2*d
    # But that might not be optimal. Let me just place them at the 5 "outer" positions.
    
    # Alternative: place 5 hexagons at distance 2*d at angles 30, 90, 150, 210, 270
    d2 = 2.0 * d
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180  # 30, 102, 174, 246, 318
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    # Start with honeycomb arrangement
    centers, angles_deg = _initial_honeycomb()
    params = np.column_stack([centers, angles_deg]).flatten()
    
    # Compute initial s
    all_verts = _compute_all_vertices(centers, angles_deg)
    best_s = _min_outer_side(all_verts)
    best_params = params.copy()
    
    # Verify initial is valid
    if _check_nonoverlap(centers, angles_deg):
        best_s = _min_outer_side(all_verts)
    else:
        # Need to adjust
        # Scale up positions
        scale = 1.1
        while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
            scale += 0.05
        centers = centers * scale
        params = np.column_stack([centers, angles_deg]).flatten()
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts)
        best_params = params.copy()
    
    # Simulated annealing
    np.random.seed(42)
    T = 0.5  # initial temperature
    cooling = 0.9995
    step_size = 0.3
    
    best_s = _min_outer_side(_compute_all_vertices(best_params.reshape(12,3)[:, :2], best_params.reshape(12,3)[:, 2]))
    
    current_params = best_params.copy()
    current_s = best_s
    
    max_iters = 500000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Perturb
        new_params = current_params + np.random.randn(36) * step_size
        
        # Compute objective
        centers_new = new_params.reshape(12, 3)[:, :2]
        angles_new = new_params.reshape(12, 3)[:, 2]
        
        all_verts_new = _compute_all_vertices(centers_new, angles_new)
        s_new = _min_outer_side(all_verts_new)
        
        # Check non-overlap
        if not _check_nonoverlap(centers_new, angles_new):
            s_new += 50.0  # heavy penalty
        
        delta = s_new - current_s
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        # Reduce step size slowly
        if it % 1000 == 0:
            step_size *= 0.95
            if step_size < 0.01:
                step_size = 0.01
    
    # Final refinement: try to minimize s more precisely
    # Use the best_params found
    centers_best = best_params.reshape(12, 3)[:, :2]
    angles_best = best_params.reshape(12, 3)[:, 2]
    
    # Try small random perturbations to see if we can improve
    for _ in range(10000):
        if time.time() > deadline:
            break
        trial = best_params + np.random.randn(36) * 0.05
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Final result
    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    # Shift so outer hexagon is centered at origin
    # The outer hexagon should be centered at origin for the formula to work correctly
    # Actually, we can center the whole configuration at the centroid of inner hexagons
    # But the outer hexagon is defined to be centered at origin with angle 0.
    # Let's shift all inner hexagons so their centroid is at origin
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    # Recompute s
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    s_final = _min_outer_side(all_verts_final)
    
    # Verify
    if not _check_nonoverlap(centers_final, angles_final):
        # Fallback to initial
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
