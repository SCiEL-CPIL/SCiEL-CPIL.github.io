"""
Combined strategy: Discrete Polyhex Search + Continuous Refinement.

1. Generate all free 11-cell polyhexes via BFS.
2. Screen them using a fast discrete bounding hexagon check.
3. Refine the top candidates using vectorized L-BFGS-B optimization.
4. Final polish with Nelder-Mead.
"""
import numpy as np
from scipy.optimize import minimize
import time
import math
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START

# --- Geometry Helpers ---

def _hex_vertices_array(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon as (6,2) array."""
    a = np.radians(angle_deg)
    # Vertices at angles a + pi/6 + k*pi/3 for flat-top alignment relative to normal
    # Actually, standard flat-top hexagon vertices are at 0, 60, 120...
    # Let's stick to the convention used in the current program:
    # angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    # This implies the "pointy-top" orientation if a=0? 
    # Let's verify: cos(pi/6)=sqrt(3)/2, sin(pi/6)=0.5. Vertex at (sqrt(3)/2, 0.5).
    # Next: cos(pi/6+pi/3)=cos(pi/2)=0, sin(pi/2)=1. Vertex at (0, 1).
    # This is a pointy-top hexagon (vertex at top).
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _all_vertices(positions, angles, side=1.0):
    """Get all vertices of all hexagons as (N*6, 2) array."""
    n = len(positions)
    verts = np.zeros((n * 6, 2))
    for i in range(n):
        verts[i*6:(i+1)*6] = _hex_vertices_array(positions[i][0], positions[i][1], side, angles[i])
    return verts

def _min_enclosing_hex_side(positions, angles, outer_center=(0.0, 0.0), outer_rotation_deg=0.0):
    """Compute minimum side length of outer hexagon centered at outer_center with given rotation
    to contain all inner hexagons."""
    verts = _all_vertices(positions, angles, 1.0)
    verts = verts - np.array(outer_center)
    phi = np.radians(outer_rotation_deg)
    # Face normals for a regular hexagon. 
    # If vertices are at pi/6 + k*pi/3, normals are at k*pi/3.
    # Wait, the current code uses: thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    # These are pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6.
    # These are normals to the edges of a "flat-top" hexagon?
    # Let's check: Edge between v0 (pi/6) and v1 (pi/2) has midpoint angle pi/4? No.
    # v0=(sqrt3/2, 1/2), v1=(0, 1). Midpoint (sqrt3/4, 3/4). Normal is (1, 1/sqrt3)?
    # Actually, for a regular polygon, the normal to the edge connecting vertices at theta_i and theta_{i+1}
    # is at angle (theta_i + theta_{i+1})/2.
    # Midpoint of angles pi/6 and pi/2 is pi/3.
    # So normals are at pi/3, 2pi/3, ... ?
    # Let's re-read the current code's logic which works.
    # It projects onto thetas = pi/6 + k*pi/3.
    # If the hexagon is "pointy top" (vertex at pi/2), the flat sides are horizontal?
    # v1=(0,1), v2=(-sqrt3/2, 1/2). Edge is not horizontal.
    # Let's trust the current code's definition of "outer_rotation" and "side".
    # The score is invariant to rotation as long as the geometry is consistent.
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    max_proj = 0.0
    for theta in thetas:
        proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return max_proj / (np.sqrt(3) / 2)

def _sat_gap(p1, a1, p2, a2, side=1.0):
    """Return the minimum gap between two hexagons using SAT."""
    v1 = _hex_vertices_array(p1[0], p1[1], side, a1)
    v2 = _hex_vertices_array(p2[0], p2[1], side, a2)
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        n_vec = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n_vec)
        if norm < 1e-12:
            continue
        n_vec = n_vec / norm
        proj1 = np.dot(v1, n_vec)
        proj2 = np.dot(v2, n_vec)
        gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_all_disjoint(positions, angles, tol=1e-9):
    """Check all pairs are disjoint (non-overlapping)."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            gap = _sat_gap(positions[i], angles[i], positions[j], angles[j])
            if gap < -tol:
                return False
    return True

def _check_inside_outer(positions, angles, outer_center, outer_side, outer_rotation_deg, tol=1e-9):
    """Check all inner hexagons are inside the outer hexagon."""
    verts = _all_vertices(positions, angles, 1.0)
    verts = verts - np.array(outer_center)
    phi = np.radians(outer_rotation_deg)
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    max_allowed = outer_side * np.sqrt(3) / 2
    for theta in thetas:
        proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
        if np.max(proj) > max_allowed + tol:
            return False
    return True

# --- Fast Vectorized Objectives for Optimizer ---

def _fast_overlap_penalty(pos_flat):
    """Vectorized overlap penalty for N hexagons with angle 0."""
    n = N
    pos = pos_flat.reshape(n, 2)
    
    # Base vertices for angle 0
    base_angles = np.pi / 6 + np.arange(6) * np.pi / 3
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    verts = pos[:, np.newaxis, :] + base_verts[np.newaxis, :, :]
    
    # SAT axes for hexagons at angle 0
    # Normals are at pi/6 + k*pi/3 ? No, edges normals.
    # Edge between v0 (pi/6) and v1 (pi/2). Midpoint angle pi/3. Normal is pi/3.
    # So axes are pi/3, 2pi/3, ... i.e., k*pi/3.
    axis_angles = np.arange(6) * np.pi / 3
    axes = np.column_stack([np.cos(axis_angles), np.sin(axis_angles)])
    
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            v1 = verts[i]
            v2 = verts[j]
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            min1 = np.min(proj1)
            max1 = np.max(proj1)
            min2 = np.min(proj2)
            max2 = np.max(proj2)
            
            gap1 = max1 - min2
            gap2 = max2 - min1
            gaps = np.minimum(gap1, gap2)
            
            min_gap = np.min(gaps)
            if min_gap < 0:
                penalty += min_gap * min_gap
                
    return penalty

def _fast_smooth_outer_side(pos_flat, outer_center, outer_rotation_deg, beta=50.0):
    """Vectorized smooth outer side length."""
    n = N
    pos = pos_flat.reshape(n, 2)
    base_angles = np.pi / 6 + np.arange(6) * np.pi / 3
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    verts = pos[:, np.newaxis, :] + base_verts[np.newaxis, :, :]
    verts = verts.reshape(-1, 2)
    verts = verts - np.array(outer_center)
    
    phi = np.radians(outer_rotation_deg)
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    
    smooth_maxes = []
    for theta in thetas:
        proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
        m = np.max(proj)
        lse = m + np.log(np.sum(np.exp(beta * (proj - m)))) / beta
        smooth_maxes.append(lse)
    
    sm = np.max(smooth_maxes)
    lse_outer = sm + np.log(np.sum(np.exp(beta * (np.array(smooth_maxes) - sm)))) / beta
    return lse_outer / (np.sqrt(3) / 2)

# --- Optimization Routines ---

def _optimize_seed(pos_seed, deadline, max_iter=200):
    """Optimize a specific seed arrangement."""
    n = N
    x0 = np.zeros(2 * n + 3)
    for i in range(n):
        x0[2*i] = pos_seed[i][0]
        x0[2*i+1] = pos_seed[i][1]
    x0[2*n] = 0.0
    x0[2*n+1] = 0.0
    x0[2*n+2] = 0.0
    
    def obj(x):
        if time.time() > deadline:
            return float('inf')
        pos_flat = x[:2*n]
        oc = (x[2*n], x[2*n+1])
        rot = math.degrees(x[2*n+2])
        
        pen = _fast_overlap_penalty(pos_flat)
        side = _fast_smooth_outer_side(pos_flat, oc, rot, beta=50.0)
        
        return side + 100.0 * pen

    try:
        result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < 1e5:
            opt_pos_flat = result.x[:2*n]
            opt_pos = [(opt_pos_flat[2*i], opt_pos_flat[2*i+1]) for i in range(n)]
            opt_oc = (result.x[2*n], result.x[2*n+1])
            opt_rot = math.degrees(result.x[2*n+2])
            
            angles = [0.0] * n
            if _check_all_disjoint(opt_pos, angles):
                actual_side = _min_enclosing_hex_side(opt_pos, angles, opt_oc, opt_rot)
                if _check_inside_outer(opt_pos, angles, opt_oc, actual_side, opt_rot):
                    return opt_pos, opt_oc, actual_side, opt_rot
    except Exception:
        pass
    return None

def _nelder_mead_polish(positions, outer_center, outer_side, outer_rotation_deg, deadline, max_iter=50):
    """Final polish using Nelder-Mead on the true (non-smooth) objective."""
    n = N
    x0 = np.zeros(2 * n + 3)
    for i in range(n):
        x0[2*i] = positions[i][0]
        x0[2*i+1] = positions[i][1]
    x0[2*n] = outer_center[0]
    x0[2*n+1] = outer_center[1]
    x0[2*n+2] = outer_rotation_deg
    
    def obj(x):
        if time.time() > deadline:
            return float('inf')
        pos = [(x[2*i], x[2*i+1]) for i in range(n)]
        oc = (x[2*n], x[2*n+1])
        rot = x[2*n+2]
        
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                gap = _sat_gap(pos[i], 0.0, pos[j], 0.0)
                if gap < 0:
                    penalty += 100.0 * gap * gap
        
        actual_side = _min_enclosing_hex_side(pos, [0.0]*n, oc, rot)
        return actual_side + penalty
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < 1e4:
            opt_pos = [(result.x[2*i], result.x[2*i+1]) for i in range(n)]
            opt_oc = (result.x[2*n], result.x[2*n+1])
            opt_rot = result.x[2*n+2]
            angles = [0.0] * n
            if _check_all_disjoint(opt_pos, angles):
                actual_side = _min_enclosing_hex_side(opt_pos, angles, opt_oc, opt_rot)
                if _check_inside_outer(opt_pos, angles, opt_oc, actual_side, opt_rot):
                    return opt_pos, opt_oc, actual_side, opt_rot
    except Exception:
        pass
    return None

# --- Polyhex Generation ---

def _axial_to_xy(q, r):
    """Convert axial coordinates to Cartesian for flat-top hexagons, side 1."""
    s3 = math.sqrt(3)
    return (1.5 * q, s3 * (r + q / 2.0))

def _generate_polyhexes(n_cells=11):
    """
    Generate all free polyhexes with n_cells cells using BFS.
    Returns a list of sets of axial coordinates.
    """
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def normalize(cells):
        """Normalize a set of axial coords so that the centroid is at (0,0) (approximately)."""
        cells = list(cells)
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        translated = [(c[0] - min_q, c[1] - min_r) for c in cells]
        return frozenset(translated)
    
    def canonical(cells):
        """Get canonical form by trying all 6 rotations and 6 reflections (12 total)."""
        cells_list = list(cells)
        forms = set()
        for rot in range(6):
            c = list(cells_list)
            for _ in range(rot):
                c = [(-r, q + r) for q, r in c]
            forms.add(normalize(c))
            c_ref = [(-q - r, r) for q, r in c]
            forms.add(normalize(c_ref))
        return min(forms)
    
    seen_canonical = set()
    polyhexes = []
    initial = frozenset([(0, 0)])
    
    def bfs_expand(current_set, depth, target):
        if depth == target:
            canon = canonical(current_set)
            if canon not in seen_canonical:
                seen_canonical.add(canon)
                polyhexes.append(set(canon))
            return
        
        boundary = set()
        for q, r in current_set:
            for dq, dr in directions:
                pos = (q + dq, r + dr)
                if pos not in current_set:
                    boundary.add(pos)
        
        for pos in boundary:
            new_set = frozenset(current_set | {pos})
            bfs_expand(new_set, depth + 1, target)
    
    bfs_expand(initial, 1, n_cells)
    return polyhexes

def _polyhex_to_positions(cells):
    """Convert a set of axial coordinates to Cartesian positions (flat-top, side 1)."""
    # Center the polyhex at origin for better optimization start
    xs = [_axial_to_xy(q, r)[0] for q, r in cells]
    ys = [_axial_to_xy(q, r)[1] for q, r in cells]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    return [(x - cx, y - cy) for x, y in zip(xs, ys)]

# --- Main Construct ---

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # 1. Generate Polyhexes
    polyhexes = _generate_polyhexes(N)
    
    # 2. Screen Polyhexes
    # For each polyhex, calculate the minimal enclosing hexagon side length
    # using a discrete set of rotation angles.
    # We assume inner hexagons are aligned (angle 0) for the discrete check.
    # The continuous optimizer will handle rotation/translation refinement.
    
    def discrete_score(arr):
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            return float('inf')
        # Check a few rotations to estimate the best orientation
        # 0, 30, 60, 90, 120, 150
        min_side = float('inf')
        for rot in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
            s = _min_enclosing_hex_side(arr, angles, (0.0, 0.0), rot)
            if s < min_side:
                min_side = s
        return min_side

    scored_polyhexes = []
    for i, ph in enumerate(polyhexes):
        if time.time() > deadline - 60.0:
            break
        arr = _polyhex_to_positions(ph)
        s = discrete_score(arr)
        if s < float('inf'):
            scored_polyhexes.append((s, arr))
            
    # Sort by score
    scored_polyhexes.sort(key=lambda x: x[0])
    
    # Take top K for continuous optimization
    K = min(30, len(scored_polyhexes))
    top_seeds = [arr for _, arr in scored_polyhexes[:K]]
    
    # Also include the standard "ring" seeds from the original program as fallback/competition
    # In case the polyhex generation misses the specific "concentric" structure or if it's the best.
    def axial_to_xy_local(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
        
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    # Add a few specific good ring configs
    for start in range(6): # Symmetric opposite pairs
        subset = [ring2[start], ring2[(start+6)%12]]
        # Need 4 more from ring2 to make 11? No, 1+6+4=11.
        # The original program used 4 from ring2.
        # Let's just add the "full ring2" if it fits? No, ring2 has 12.
        # Let's add the "consecutive 4" from ring2.
        for start in range(12):
            subset = [ring2[(start + k) % 12] for k in range(4)]
            axial_coords = [(0,0)] + ring1 + subset
            positions = [axial_to_xy_local(q, r) for q, r in axial_coords]
            # Center it
            cx = sum(p[0] for p in positions) / N
            cy = sum(p[1] for p in positions) / N
            centered = [(p[0]-cx, p[1]-cy) for p in positions]
            if centered not in top_seeds:
                top_seeds.append(centered)
                
    # Deduplicate top_seeds roughly
    unique_seeds = []
    seen = set()
    for arr in top_seeds:
        key = tuple((round(p[0], 4), round(p[1], 4)) for p in arr)
        if key not in seen:
            seen.add(key)
            unique_seeds.append(arr)
            
    top_seeds = unique_seeds[:40] # Limit to 40 to ensure time

    best_side = float('inf')
    best_positions = None
    best_outer_center = (0.0, 0.0)
    best_outer_rotation = 0.0
    
    # 3. Continuous Optimization Phase
    # Allocate time for each seed
    time_per_seed = 2.0 # 40 seeds * 2s = 80s. Leaves 60s for generation + polish.
    
    for arr in top_seeds:
        if time.time() > deadline - 30.0: # Keep 30s for polish
            break
        res = _optimize_seed(arr, deadline, max_iter=100)
        if res:
            opt_pos, opt_oc, opt_side, opt_rot = res
            if opt_side < best_side:
                best_side = opt_side
                best_positions = opt_pos
                best_outer_center = opt_oc
                best_outer_rotation = opt_rot

    # 4. Polish Phase
    if best_positions is not None and time.time() < deadline - 5.0:
        res = _nelder_mead_polish(best_positions, best_outer_center, best_side, best_outer_rotation, deadline, max_iter=100)
        if res:
            opt_pos, opt_oc, opt_side, opt_rot = res
            if opt_side < best_side:
                best_side = opt_side
                best_positions = opt_pos
                best_outer_center = opt_oc
                best_outer_rotation = opt_rot
    
    # Fallback
    if best_positions is None:
        # Use the best discrete score if no continuous result
        if scored_polyhexes:
            best_side, best_positions = scored_polyhexes[0]
            best_outer_center = (0.0, 0.0)
            best_outer_rotation = 0.0
        else:
            # Absolute fallback
            best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
            best_outer_center = (0.0, 0.0)
            best_outer_rotation = 0.0
            best_side = _min_enclosing_hex_side(best_positions, [0.0]*N, (0.0, 0.0), 0.0)
            
    # Safety margin
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], 0.0] for i in range(N)]
    return inner, best_outer_center, best_side, best_outer_rotation

# EVOLVE-BLOCK-END
