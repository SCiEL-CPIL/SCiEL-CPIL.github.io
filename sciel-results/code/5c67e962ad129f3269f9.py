"""
Discrete polyhex search + continuous refinement for packing 11 unit hexagons
in a minimal outer regular hexagon.

Strategy:
1. Generate all 11-cell polyhexes (free) via BFS on axial grid.
2. For each polyhex, compute the minimal enclosing hexagon side length
   by optimizing the outer hexagon's rotation angle.
3. Use the best few polyhexes and refine with continuous optimization
   (Nelder-Mead) over rotation and translation.
4. Final exact evaluation and return.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# ---- Geometry helpers ------------------------------------------------------

def _hex_vertices(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon (flat-top at ang=0)."""
    a = math.radians(ang_deg)
    # Flat-top: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angs = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _all_vertices(positions, angles, s=1.0):
    """Return all 6*n vertices."""
    n = len(positions)
    v = np.empty((n * 6, 2))
    for i in range(n):
        v[i*6:(i+1)*6] = _hex_vertices(positions[i][0], positions[i][1], s, angles[i])
    return v

def _min_enclosing_hex_side(positions, angles, phi_rad, center=(0.0, 0.0)):
    """
    Compute the exact side length of the smallest regular hexagon with rotation phi_rad
    and center `center` that encloses all given unit hexagons.
    
    The outer hexagon has 6 face normals at angles phi_rad + (2k+1)*pi/6 for k=0..5.
    The side length s satisfies: for each normal direction n_k,
    max_i (v_i . n_k) - min_i (v_i . n_k) <= 2 * s * cos(pi/6) = s * sqrt(3)
    Wait, let me think again.
    
    For a regular hexagon with side length S and center at origin:
    - The apothem (distance from center to each face) is S * sqrt(3)/2
    - The 6 face normals are at angles (2k+1)*pi/6 for k=0..5 (for flat-top)
    - A point (x,y) is inside if for all k: (x*cos(theta_k) + y*sin(theta_k)) <= S*sqrt(3)/2
      where theta_k = (2k+1)*pi/6
    
    So the required S = max over all vertices v and all normals n_k of (v . n_k) / (sqrt(3)/2)
    (assuming center is at origin; otherwise use (v - center) . n_k)
    """
    verts = _all_vertices(positions, angles, 1.0) - np.array(center)
    thetas = phi_rad + np.array([(2*k + 1) * np.pi / 6 for k in range(6)])
    max_proj = 0.0
    for t in thetas:
        proj = verts[:, 0] * math.cos(t) + verts[:, 1] * math.sin(t)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return max_proj / (math.sqrt(3) / 2)

def _smooth_side(positions, angles, x, beta=25.0):
    """
    Smooth surrogate for outer side length.
    x = [phi, tx, ty]: outer rotation, and translation of the cluster.
    The outer hexagon is centered at origin; inner positions are translated by (tx, ty).
    """
    phi, tx, ty = x[0], x[1], x[2]
    verts = _all_vertices(positions, angles, 1.0)
    verts = verts - np.array([tx, ty])
    thetas = phi + np.array([(2*k + 1) * np.pi / 6 for k in range(6)])
    
    # Stack all projections: (6 directions, n*6 vertices)
    projs = np.zeros((6, len(verts)))
    for k, t in enumerate(thetas):
        projs[k] = verts[:, 0] * math.cos(t) + verts[:, 1] * math.sin(t)
    
    # Smooth max over vertices for each direction
    m = np.max(projs, axis=1)  # (6,)
    lse_per_dir = m + (1.0 / beta) * np.log(np.sum(np.exp(projs - m[:, None]), axis=1))
    
    # Smooth max over directions
    m2 = np.max(lse_per_dir)
    lse_final = m2 + (1.0 / beta) * math.log(np.sum(np.exp((lse_per_dir - m2) * beta)))
    return lse_final / (math.sqrt(3) / 2)

# ---- SAT overlap check -----------------------------------------------------

def _sat_gap(p1, a1, p2, a2, s=1.0):
    """Compute minimum gap between two hexagons using SAT. Negative means overlap."""
    v1 = _hex_vertices(p1[0], p1[1], s, a1)
    v2 = _hex_vertices(p2[0], p2[1], s, a2)
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        nv = np.array([-e[1], e[0]])
        n = np.linalg.norm(nv)
        if n < 1e-12:
            continue
        nv /= n
        proj1 = v1 @ nv
        proj2 = v2 @ nv
        gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_disjoint(positions, angles, tol=1e-9):
    """Check that all hexagons are pairwise disjoint (non-overlapping)."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_gap(positions[i], angles[i], positions[j], angles[j]) < -tol:
                return False
    return True

# ---- Polyhex generation via BFS on axial grid ------------------------------

def _axial_to_xy(q, r):
    """Convert axial coordinates to Cartesian for flat-top hexagons, side 1."""
    s3 = math.sqrt(3)
    return (1.5 * q, s3 * (r + q / 2.0))

def _xy_to_axial(x, y):
    """Convert Cartesian to axial coordinates (for flat-top hexagons, side 1)."""
    s3 = math.sqrt(3)
    q = (2.0 / 3.0) * x
    r = (-1.0 / 3.0) * x + (1.0 / s3) * y
    return (round(q), round(r))

def _generate_polyhexes(n_cells=11):
    """
    Generate all free polyhexes with n_cells cells using BFS.
    Returns a list of sets of axial coordinates (normalized to center at origin).
    """
    # Axial directions for flat-top hexagons
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def normalize(cells):
        """Normalize a set of axial coords so that the centroid is at (0,0) (approximately)."""
        cells = list(cells)
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        # Translate so min is at (0,0)
        translated = [(c[0] - min_q, c[1] - min_r) for c in cells]
        return frozenset(translated)
    
    def canonical(cells):
        """Get canonical form by trying all 6 rotations and 6 reflections (12 total)."""
        cells_list = list(cells)
        forms = set()
        for rot in range(6):
            # Rotate by rot * 60 degrees
            # In axial coords, rotation by 60 degrees: (q,r) -> (-r, q+r)
            c = list(cells_list)
            for _ in range(rot):
                c = [(-r, q + r) for q, r in c]
            forms.add(normalize(c))
            # Reflection: (q,r) -> (-q-r, r)
            c_ref = [(-q - r, r) for q, r in c]
            forms.add(normalize(c_ref))
        # Return the lexicographically smallest
        return min(forms)
    
    # BFS: start with single cell, add cells
    seen_canonical = set()
    polyhexes = []
    
    # Start with 1-cell polyhex
    initial = frozenset([(0, 0)])
    
    def bfs_expand(current_set, depth, target):
        """Recursively expand a polyhex by adding one cell at a time."""
        if depth == target:
            canon = canonical(current_set)
            if canon not in seen_canonical:
                seen_canonical.add(canon)
                polyhexes.append(set(canon))
            return
        
        # Find all boundary positions (adjacent to existing cells but not occupied)
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
    return [_axial_to_xy(q, r) for q, r in cells]

# ---- Main construct --------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    s3 = math.sqrt(3)
    
    # Step 1: Generate all 11-cell polyhexes
    # This might be slow, so we'll generate a subset if needed
    print(f"Generating {N}-cell polyhexes...")
    t_gen_start = time.time()
    
    try:
        polyhexes = _generate_polyhexes(N)
        print(f"Generated {len(polyhexes)} polyhexes in {time.time() - t_gen_start:.1f}s")
    except Exception as e:
        print(f"Polyhex generation failed: {e}")
        # Fallback: use a few hand-crafted clusters
        polyhexes = []
        # Hand-crafted 3-4-3 cluster
        c1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
              (2,0), (1,1), (0,2), (-1,2)]
        polyhexes.append(set(c1))
        c2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
              (2,0), (0,2), (-2,2), (0,-2)]
        polyhexes.append(set(c2))
    
    if time.time() > deadline - 60.0:
        # Not enough time, just use a few
        polyhexes = polyhexes[:20]
    
    # Step 2: For each polyhex, compute the minimal enclosing hex side length
    # by optimizing the rotation angle
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cluster = None
    
    # Sort polyhexes by a simple heuristic: bounding box area
    def bbox_area(cells):
        xs = [_axial_to_xy(q, r)[0] for q, r in cells]
        ys = [_axial_to_xy(q, r)[1] for q, r in cells]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    
    polyhexes.sort(key=bbox_area)
    
    # Evaluate top polyhexes
    n_to_eval = min(len(polyhexes), 50)
    
    for idx, cells in enumerate(polyhexes[:n_to_eval]):
        if time.time() > deadline - 40.0:
            break
        
        positions = _polyhex_to_positions(cells)
        angles = [0.0] * N
        
        # Quick check: are they disjoint? (They should be by construction)
        if not _check_disjoint(positions, angles):
            continue
        
        # Centroid
        cx0 = np.mean([p[0] for p in positions])
        cy0 = np.mean([p[1] for p in positions])
        
        # Coarse search over rotation angle
        best_local_side = float('inf')
        best_local_phi = 0.0
        best_local_tx = -cx0
        best_local_ty = -cy0
        
        for phi_deg in np.arange(0, 60, 5.0):
            if time.time() > deadline - 35.0:
                break
            phi = math.radians(phi_deg)
            # Try a few translations
            for dx in np.arange(-0.5, 0.51, 0.25):
                for dy in np.arange(-0.5, 0.51, 0.25):
                    center = (cx0 + dx, cy0 + dy)
                    side = _min_enclosing_hex_side(positions, angles, phi, center)
                    if side < best_local_side:
                        best_local_side = side
                        best_local_phi = phi
                        best_local_tx = -cx0 + dx
                        best_local_ty = -cy0 + dy
        
        if best_local_side < best_side:
            # Refine with Nelder-Mead
            x_start = np.array([best_local_phi, best_local_tx, best_local_ty])
            
            def obj(x):
                return _smooth_side(positions, angles, x, beta=25.0)
            
            try:
                res = minimize(obj, x_start, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11,
                                        'adaptive': True})
                if res.fun < best_local_side:
                    phi_sol, tx_sol, ty_sol = res.x
                    center = (-tx_sol, -ty_sol)
                    exact = _min_enclosing_hex_side(positions, angles, phi_sol, center)
                    if exact < best_local_side:
                        best_local_side = exact
                        best_local_phi = phi_sol
                        best_local_tx = tx_sol
                        best_local_ty = ty_sol
            except Exception:
                pass
            
            if best_local_side < best_side:
                best_positions = [(p[0] + best_local_tx, p[1] + best_local_ty) for p in positions]
                best_angles = angles
                best_phi = best_local_phi
                best_side = best_local_side
                best_cluster = cells
    
    # Step 3: Fine polish on the best solution
    if best_positions is not None:
        # Fine-tune phi
        phi_center = best_phi * 180 / math.pi
        for phi_deg in np.arange(phi_center - 3, phi_center + 3.01, 0.05):
            if time.time() > deadline - 10.0:
                break
            phi = math.radians(phi_deg)
            side = _min_enclosing_hex_side(best_positions, best_angles, phi, (0.0, 0.0))
            if side < best_side:
                best_side = side
                best_phi = phi
        
        # Fine-tune center (shift all positions)
        for dx in np.arange(-0.2, 0.201, 0.01):
            for dy in np.arange(-0.2, 0.201, 0.01):
                if time.time() > deadline - 5.0:
                    break
                shifted = [(p[0] + dx, p[1] + dy) for p in best_positions]
                side = _min_enclosing_hex_side(shifted, best_angles, best_phi, (0.0, 0.0))
                if side < best_side:
                    best_side = side
                    best_positions = shifted
    
    if best_positions is None:
        # Fallback: linear arrangement
        best_positions = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        best_angles = [0.0] * N
        best_side = _min_enclosing_hex_side(best_positions, best_angles, 0.0, (0.0, 0.0))
        best_phi = 0.0
    
    # Final verification
    if not _check_disjoint(best_positions, best_angles):
        best_positions = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        best_angles = [0.0] * N
        best_side = _min_enclosing_hex_side(best_positions, best_angles, 0.0, (0.0, 0.0))
        best_phi = 0.0
    
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, math.degrees(best_phi)
