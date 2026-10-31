"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math
from itertools import combinations

def _hex_vertices_array(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon as (6,2) array."""
    a = np.radians(angle_deg)
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
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    max_proj = 0.0
    for theta in thetas:
        proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return max_proj / (np.sqrt(3) / 2)

def _smooth_enclosing_hex_side(positions, angles, outer_center=(0.0, 0.0), outer_rotation_deg=0.0, beta=50.0):
    """Smooth surrogate for outer hex side using log-sum-exp over projections."""
    verts = _all_vertices(positions, angles, 1.0)
    verts = verts - np.array(outer_center)
    phi = np.radians(outer_rotation_deg)
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    
    # For each of 6 directions, compute smooth max of projections
    smooth_maxes = []
    for theta in thetas:
        proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
        # Log-sum-exp for smooth max
        m = np.max(proj)
        lse = m + np.log(np.sum(np.exp(beta * (proj - m)))) / beta
        smooth_maxes.append(lse)
    
    # Smooth max over the 6 directions
    sm = np.max(smooth_maxes)
    lse_outer = sm + np.log(np.sum(np.exp(beta * (np.array(smooth_maxes) - sm)))) / beta
    return lse_outer / (np.sqrt(3) / 2)

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

def _optimize_inner_only(positions, angles, outer_rotation_deg, deadline, max_iter=150):
    """Phase 1: Fix outer hexagon at origin with given rotation, optimize only inner positions.
    Uses L-BFGS-B with smooth surrogate for outer constraint."""
    original_side = _min_enclosing_hex_side(positions, angles, (0.0, 0.0), outer_rotation_deg)
    x0 = np.zeros(2 * N)
    for i in range(N):
        x0[2*i] = positions[i][0]
        x0[2*i+1] = positions[i][1]
    
    def obj(x):
        if time.time() > deadline:
            return float('inf')
        pos = [(x[2*i], x[2*i+1]) for i in range(N)]
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                gap = _sat_gap(pos[i], angles[i], pos[j], angles[j])
                if gap < 0:
                    penalty += 1000.0 * gap * gap
        # Smooth outer side
        smooth_side = _smooth_enclosing_hex_side(pos, angles, (0.0, 0.0), outer_rotation_deg, beta=30.0)
        return smooth_side + penalty
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < 1e5:
            opt_pos = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
            if _check_all_disjoint(opt_pos, angles):
                opt_side = _min_enclosing_hex_side(opt_pos, angles, (0.0, 0.0), outer_rotation_deg)
                if opt_side < original_side:
                    return opt_pos, opt_side
    except Exception:
        pass
    return positions, original_side

def _optimize_joint(positions, angles, outer_center, outer_side, outer_rotation_deg, deadline, max_iter=100):
    """Phase 2: Joint optimization of inner positions and outer parameters.
    Variables: [x0,y0,...,xN-1,yN-1, ox, oy, log(side), rotation_rad]"""
    x0 = np.zeros(2 * N + 4)
    for i in range(N):
        x0[2*i] = positions[i][0]
        x0[2*i+1] = positions[i][1]
    x0[2*N] = outer_center[0]
    x0[2*N+1] = outer_center[1]
    x0[2*N+2] = math.log(max(outer_side, 0.1))
    x0[2*N+3] = math.radians(outer_rotation_deg)
    
    def obj(x):
        if time.time() > deadline:
            return float('inf')
        pos = [(x[2*i], x[2*i+1]) for i in range(N)]
        oc = (x[2*N], x[2*N+1])
        side = math.exp(x[2*N+2])
        rot = math.degrees(x[2*N+3])
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                gap = _sat_gap(pos[i], angles[i], pos[j], angles[j])
                if gap < 0:
                    penalty += 1000.0 * gap * gap
        
        # Smooth outer side
        smooth_side = _smooth_enclosing_hex_side(pos, angles, oc, rot, beta=30.0)
        return smooth_side + penalty
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < 1e5:
            opt_pos = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
            opt_oc = (result.x[2*N], result.x[2*N+1])
            opt_side = math.exp(result.x[2*N+2])
            opt_rot = math.degrees(result.x[2*N+3])
            if _check_all_disjoint(opt_pos, angles) and _check_inside_outer(opt_pos, angles, opt_oc, opt_side, opt_rot):
                actual_side = _min_enclosing_hex_side(opt_pos, angles, opt_oc, opt_rot)
                if actual_side < outer_side:
                    return opt_pos, angles, opt_oc, actual_side, opt_rot
    except Exception:
        pass
    return positions, angles, outer_center, outer_side, outer_rotation_deg

def _nelder_mead_polish(positions, angles, outer_center, outer_side, outer_rotation_deg, deadline, max_iter=50):
    """Final polish using Nelder-Mead on the true (non-smooth) objective."""
    x0 = np.zeros(2 * N + 4)
    for i in range(N):
        x0[2*i] = positions[i][0]
        x0[2*i+1] = positions[i][1]
    x0[2*N] = outer_center[0]
    x0[2*N+1] = outer_center[1]
    x0[2*N+2] = outer_side
    x0[2*N+3] = outer_rotation_deg
    
    def obj(x):
        if time.time() > deadline:
            return float('inf')
        pos = [(x[2*i], x[2*i+1]) for i in range(N)]
        oc = (x[2*N], x[2*N+1])
        side = max(x[2*N+2], 0.1)
        rot = x[2*N+3]
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                gap = _sat_gap(pos[i], angles[i], pos[j], angles[j])
                if gap < 0:
                    penalty += 100.0 * gap * gap
        
        # True outer side
        actual_side = _min_enclosing_hex_side(pos, angles, oc, rot)
        return actual_side + penalty
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < 1e4:
            opt_pos = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
            opt_oc = (result.x[2*N], result.x[2*N+1])
            opt_side = max(result.x[2*N+2], 0.1)
            opt_rot = result.x[2*N+3]
            if _check_all_disjoint(opt_pos, angles) and _check_inside_outer(opt_pos, angles, opt_oc, opt_side, opt_rot):
                actual_side = _min_enclosing_hex_side(opt_pos, angles, opt_oc, opt_rot)
                return opt_pos, opt_oc, actual_side, opt_rot
    except Exception:
        pass
    return positions, outer_center, outer_side, outer_rotation_deg

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
    
    # Generate candidate arrangements
    arrangements = []
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    # Option 1: 4 consecutive from ring2
    for start in range(12):
        subset = [ring2[(start + k) % 12] for k in range(4)]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Option 2: Symmetric selections (opposite pairs)
    for i in range(6):
        for j in range(i+1, 6):
            subset = [ring2[i], ring2[i+6], ring2[j], ring2[j+6]]
            axial_coords = [(0,0)] + ring1 + subset
            positions = [axial_to_xy(q, r) for q, r in axial_coords]
            arrangements.append(positions)
    
    # Option 3: All combinations of 4 from ring2
    for combo in combinations(range(12), 4):
        subset = [ring2[i] for i in combo]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Deduplicate
    unique_arrangements = []
    seen = set()
    for arr in arrangements:
        key = tuple((round(p[0], 6), round(p[1], 6)) for p in arr)
        if key not in seen:
            seen.add(key)
            unique_arrangements.append(arr)
    arrangements = unique_arrangements
    
    # Score and sort arrangements
    def arr_score(arr):
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            return float('inf')
        s0 = _min_enclosing_hex_side(arr, angles, (0.0, 0.0), 0.0)
        s30 = _min_enclosing_hex_side(arr, angles, (0.0, 0.0), 30.0)
        return min(s0, s30)
    
    scored = []
    for arr in arrangements:
        if time.time() > deadline - 60.0:
            break
        s = arr_score(arr)
        if s < float('inf'):
            scored.append((s, arr))
    scored.sort(key=lambda x: x[0])
    
    # Keep top seeds
    K = min(12, len(scored))
    top_seeds = [arr for _, arr in scored[:K]]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_center = (0.0, 0.0)
    best_outer_rotation = 0.0
    
    # Phase 1: Quick evaluation of all valid arrangements
    for arr in arrangements:
        if time.time() > deadline - 50.0:
            break
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            continue
        for outer_rot in [0.0, 30.0]:
            side = _min_enclosing_hex_side(arr, angles, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_positions = arr[:]
                best_angles = angles[:]
                best_outer_center = (0.0, 0.0)
                best_outer_rotation = outer_rot
    
    # Phase 2: Separable optimization on top seeds
    for arr in top_seeds:
        if time.time() > deadline - 20.0:
            break
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            continue
        for outer_rot in [0.0, 30.0]:
            opt_pos, opt_side = _optimize_inner_only(arr, angles, outer_rot, deadline, max_iter=100)
            if opt_side < best_side:
                best_side = opt_side
                best_positions = opt_pos[:]
                best_angles = angles[:]
                best_outer_center = (0.0, 0.0)
                best_outer_rotation = outer_rot
    
    # Phase 3: Joint optimization on the best result
    if best_positions is not None and time.time() < deadline - 15.0:
        opt_pos, opt_angles, opt_oc, opt_side, opt_rot = _optimize_joint(
            best_positions, best_angles, best_outer_center, best_side, best_outer_rotation, deadline, max_iter=80
        )
        if opt_side < best_side:
            best_side = opt_side
            best_positions = opt_pos
            best_angles = opt_angles
            best_outer_center = opt_oc
            best_outer_rotation = opt_rot
    
    # Phase 4: Nelder-Mead polish
    if best_positions is not None and time.time() < deadline - 5.0:
        opt_pos, opt_oc, opt_side, opt_rot = _nelder_mead_polish(
            best_positions, best_angles, best_outer_center, best_side, best_outer_rotation, deadline, max_iter=40
        )
        if opt_side < best_side:
            best_side = opt_side
            best_positions = opt_pos
            best_outer_center = opt_oc
            best_outer_rotation = opt_rot
    
    if best_positions is None:
        # Fallback
        best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        best_angles = [0.0] * N
        best_outer_center = (0.0, 0.0)
        best_outer_rotation = 0.0
        best_side = _min_enclosing_hex_side(best_positions, best_angles, (0.0, 0.0), 0.0)
    
    # Add tiny epsilon for strict containment
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, best_outer_center, best_side, best_outer_rotation
# EVOLVE-BLOCK-END
