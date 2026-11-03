"""Hexagonal packing: 12 unit hexagons in a regular hexagon, minimize outer side."""
N = 12
import numpy as np
import time
import random

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of regular hexagon centered at (cx,cy) with rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _sat_min_inradius(verts_list, outer_angle_deg):
    """
    Compute the minimum inradius of a regular hexagon (with given rotation)
    that contains all given vertex sets. Returns the required inradius.
    """
    outer_angle_rad = np.radians(outer_angle_deg)
    max_proj = 0.0
    # The 6 face normal directions of the outer hexagon
    normals = []
    for k in range(6):
        theta = outer_angle_rad + k * np.pi / 3 + np.pi / 6
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    for verts in verts_list:
        for v in verts:
            for n in normals:
                proj = np.dot(v, n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj

def _check_no_overlap(centers, angles, tol=1e-7):
    """Check that all unit hexagons are disjoint using SAT."""
    n = len(centers)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(centers[i][0], centers[i][1], angles[i]))
    
    for i in range(n):
        for j in range(i+1, n):
            v1, v2 = verts_list[i], verts_list[j]
            # SAT: check all 12 axes (6 from each polygon)
            overlap = True
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-15:
                        continue
                    normal /= norm
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 < min2 + tol or max2 < min1 + tol:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    return True

def _axial_to_pixel(q, r):
    """Convert axial hex grid coords to pixel coords for flat-top hexagons side=1."""
    sqrt3 = np.sqrt(3)
    x = 1.5 * q
    y = sqrt3 * (r + q / 2.0)
    return x, y

def _pixel_to_axial(x, y):
    """Approximate inverse of axial_to_pixel."""
    sqrt3 = np.sqrt(3)
    q = (2.0/3.0 * x + 1.0/sqrt3 * y)
    r = (-1.0/sqrt3 * y)
    return round(q), round(r)

def _generate_lattice_positions(n_hexes):
    """Generate a compact cluster of n_hexes unit hexagons on hex lattice."""
    # Use a spiral/growth pattern to get compact shapes
    # Start with center, then add neighbors in rings
    axial = [(0, 0)]
    # Directions for 6 neighbors
    dirs = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    # BFS to grow compact cluster
    from collections import deque
    visited = {(0,0)}
    queue = deque([(0,0)])
    while len(axial) < n_hexes:
        q, r = queue.popleft()
        for dq, dr in dirs:
            nq, nr = q+dq, r+dr
            if (nq, nr) not in visited:
                visited.add((nq, nr))
                axial.append((nq, nr))
                queue.append((nq, nr))
                if len(axial) >= n_hexes:
                    break
        if len(axial) >= n_hexes:
            break
    
    # Convert to pixel and center
    pixels = []
    for q, r in axial[:n_hexes]:
        x, y = _axial_to_pixel(q, r)
        pixels.append((x, y))
    
    cx = np.mean([p[0] for p in pixels])
    cy = np.mean([p[1] for p in pixels])
    pixels = [(x-cx, y-cy) for x, y in pixels]
    
    return pixels, axial[:n_hexes]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    sqrt3 = np.sqrt(3)
    
    # Try multiple initial configurations
    best_side = np.inf
    best_result = None
    
    # Strategy: Try various outer hexagon rotations and inner arrangements
    # Key insight: for hexagonal packing, the optimal outer hexagon should be
    # aligned with the inner hexagons (same orientation) for minimal side length
    
    # Generate several candidate initial layouts
    candidate_layouts = []
    
    # Layout 1: Compact spiral cluster
    pixels1, axial1 = _generate_lattice_positions(12)
    candidate_layouts.append(pixels1)
    
    # Layout 2: 2-3-4-3 arrangement (different growth)
    axial2 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, -1), (0, -1), (1, -1),
        (0, 2), (1, 2)
    ]
    pixels2 = []
    for q, r in axial2:
        x, y = _axial_to_pixel(q, r)
        pixels2.append((x, y))
    cx = np.mean([p[0] for p in pixels2])
    cy = np.mean([p[1] for p in pixels2])
    pixels2 = [(x-cx, y-cy) for x, y in pixels2]
    candidate_layouts.append(pixels2)
    
    # Layout 3: 3-4-3-2 arrangement
    axial3 = [
        (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (0, 2), (1, 2)
    ]
    pixels3 = []
    for q, r in axial3:
        x, y = _axial_to_pixel(q, r)
        pixels3.append((x, y))
    cx = np.mean([p[0] for p in pixels3])
    cy = np.mean([p[1] for p in pixels3])
    pixels3 = [(x-cx, y-cy) for x, y in pixels3]
    candidate_layouts.append(pixels3)
    
    # Layout 4: 4-4-4 arrangement (rectangular-ish)
    axial4 = [
        (-2, -1), (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0),
        (-2, 1), (-1, 1), (0, 1), (1, 1)
    ]
    pixels4 = []
    for q, r in axial4:
        x, y = _axial_to_pixel(q, r)
        pixels4.append((x, y))
    cx = np.mean([p[0] for p in pixels4])
    cy = np.mean([p[1] for p in pixels4])
    pixels4 = [(x-cx, y-cy) for x, y in pixels4]
    candidate_layouts.append(pixels4)
    
    # Layout 5: Hexagonal number pattern (center + ring of 6 + partial ring)
    axial5 = [(0,0)]
    # Ring 1: 6 hexes
    ring1_dirs = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    for dq, dr in ring1_dirs:
        axial5.append((dq, dr))
    # Ring 2: need 5 more
    ring2 = [(2,0),(2,-1),(1,-2),(-1,-2),(-2,-1)]
    for dq, dr in ring2[:5]:
        axial5.append((dq, dr))
    pixels5 = []
    for q, r in axial5[:12]:
        x, y = _axial_to_pixel(q, r)
        pixels5.append((x, y))
    cx = np.mean([p[0] for p in pixels5])
    cy = np.mean([p[1] for p in pixels5])
    pixels5 = [(x-cx, y-cy) for x, y in pixels5]
    candidate_layouts.append(pixels5)
    
    # For each layout and each outer rotation, compute side length
    outer_angles = [0, 30, 60, 90, 120, 150]
    
    for layout in candidate_layouts:
        if time.time() > deadline - 60:
            break
        
        centers = np.array(layout)
        angles = np.zeros(N)
        
        # Verify no overlaps
        if not _check_no_overlap(centers, angles):
            continue
        
        for oa in outer_angles:
            if time.time() > deadline - 60:
                break
            
            # Compute required outer hexagon inradius
            verts_list = []
            for i in range(N):
                verts_list.append(_hex_vertices(centers[i,0], centers[i,1], angles[i]))
            
            # Center the outer hexagon at centroid of all vertices
            all_verts = np.vstack(verts_list)
            outer_cx = np.mean(all_verts[:, 0])
            outer_cy = np.mean(all_verts[:, 1])
            
            # Shift vertices relative to outer center
            verts_list_rel = [v - np.array([outer_cx, outer_cy]) for v in verts_list]
            
            inradius = _sat_min_inradius(verts_list_rel, oa)
            side = 2.0 * inradius / sqrt3
            
            if side < best_side:
                best_side = side
                best_result = {
                    'centers': centers.copy(),
                    'angles': angles.copy(),
                    'outer_cx': outer_cx,
                    'outer_cy': outer_cy,
                    'outer_angle': oa
                }
    
    # Now do a local refinement: try small perturbations of positions
    # and different outer rotations
    if best_result is not None and time.time() < deadline - 30:
        current_centers = best_result['centers'].copy()
        current_angles = best_result['angles'].copy()
        current_outer_cx = best_result['outer_cx']
        current_outer_cy = best_result['outer_cy']
        current_outer_angle = best_result['outer_angle']
        current_best = best_side
        
        # Try fine-tuning outer angle
        for oa in np.arange(0, 360, 5):
            if time.time() > deadline - 20:
                break
            verts_list = []
            for i in range(N):
                verts_list.append(_hex_vertices(current_centers[i,0], current_centers[i,1], current_angles[i]))
            all_verts = np.vstack(verts_list)
            outer_cx = np.mean(all_verts[:, 0])
            outer_cy = np.mean(all_verts[:, 1])
            verts_list_rel = [v - np.array([outer_cx, outer_cy]) for v in verts_list]
            inradius = _sat_min_inradius(verts_list_rel, oa)
            side = 2.0 * inradius / sqrt3
            if side < current_best:
                current_best = side
                current_outer_angle = oa
                current_outer_cx = outer_cx
                current_outer_cy = outer_cy
                best_side = side
                best_result = {
                    'centers': current_centers.copy(),
                    'angles': current_angles.copy(),
                    'outer_cx': outer_cx,
                    'outer_cy': outer_cy,
                    'outer_angle': oa
                }
        
        # Try small position perturbations (local search)
        if time.time() < deadline - 20:
            for _ in range(500):
                if time.time() > deadline - 15:
                    break
                
                # Pick a random hexagon and nudge it
                idx = random.randint(0, N-1)
                dx = random.uniform(-0.15, 0.15)
                dy = random.uniform(-0.15, 0.15)
                new_centers = current_centers.copy()
                new_centers[idx, 0] += dx
                new_centers[idx, 1] += dy
                
                # Check no overlaps
                if not _check_no_overlap(new_centers, current_angles):
                    continue
                
                # Compute side length
                verts_list = []
                for i in range(N):
                    verts_list.append(_hex_vertices(new_centers[i,0], new_centers[i,1], current_angles[i]))
                all_verts = np.vstack(verts_list)
                outer_cx = np.mean(all_verts[:, 0])
                outer_cy = np.mean(all_verts[:, 1])
                verts_list_rel = [v - np.array([outer_cx, outer_cy]) for v in verts_list]
                
                best_local_side = np.inf
                best_local_ao = current_outer_angle
                for oa in [current_outer_angle, current_outer_angle + 5, current_outer_angle - 5]:
                    inradius = _sat_min_inradius(verts_list_rel, oa)
                    side = 2.0 * inradius / sqrt3
                    if side < best_local_side:
                        best_local_side = side
                        best_local_ao = oa
                
                if best_local_side < current_best - 1e-9:
                    current_best = best_local_side
                    current_centers = new_centers
                    current_outer_angle = best_local_ao
                    current_outer_cx = outer_cx
                    current_outer_cy = outer_cy
                    best_side = current_best
                    best_result = {
                        'centers': current_centers.copy(),
                        'angles': current_angles.copy(),
                        'outer_cx': current_outer_cx,
                        'outer_cy': current_outer_cy,
                        'outer_angle': current_outer_angle
                    }
    
    # If we haven't found anything valid, fall back to a simple arrangement
    if best_result is None:
        pixels1, _ = _generate_lattice_positions(12)
        centers = np.array(pixels1)
        angles = np.zeros(N)
        verts_list = [_hex_vertices(centers[i,0], centers[i,1], angles[i]) for i in range(N)]
        all_verts = np.vstack(verts_list)
        outer_cx = np.mean(all_verts[:, 0])
        outer_cy = np.mean(all_verts[:, 1])
        verts_list_rel = [v - np.array([outer_cx, outer_cy]) for v in verts_list]
        inradius = _sat_min_inradius(verts_list_rel, 0)
        best_side = 2.0 * inradius / sqrt3
        best_result = {
            'centers': centers,
            'angles': angles,
            'outer_cx': outer_cx,
            'outer_cy': outer_cy,
            'outer_angle': 0.0
        }
    
    # Final: verify and return
    centers = best_result['centers']
    angles = best_result['angles']
    outer_cx = best_result['outer_cx']
    outer_cy = best_result['outer_cy']
    outer_angle = best_result['outer_angle']
    
    # Recompute side length precisely
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(centers[i,0], centers[i,1], angles[i]))
    all_verts = np.vstack(verts_list)
    outer_cx = np.mean(all_verts[:, 0])
    outer_cy = np.mean(all_verts[:, 1])
    verts_list_rel = [v - np.array([outer_cx, outer_cy]) for v in verts_list]
    inradius = _sat_min_inradius(verts_list_rel, outer_angle)
    final_side = 2.0 * inradius / sqrt3
    
    inner = [[float(centers[i,0]), float(centers[i,1]), float(angles[i])] for i in range(N)]
    
    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle)
