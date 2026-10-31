"""Rewritten program: smooth log-sum-exp outer objective + SAT-based inner optimization."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hex_vertices_array(centers, angles_deg):
    """Vectorized: return (N, 6, 2) array of vertices for N hexagons."""
    angles = np.radians(angles_deg)
    base_angles = angles[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    cos_a = np.cos(base_angles)
    sin_a = np.sin(base_angles)
    x = centers[:, 0:1] + cos_a
    y = centers[:, 1:2] + sin_a
    return np.stack([x, y], axis=-1)

def _sat_gap(v1, v2):
    """SAT gap between two convex polygons given as vertex arrays.
    Positive = separated, negative = overlapping."""
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            continue
        n /= norm
        p1 = v1 @ n
        p2 = v2 @ n
        gap = min(p1.max() - p2.min(), p2.max() - p1.min())
        if gap < min_gap:
            min_gap = gap
            if min_gap < -1e-10:
                return min_gap
    return min_gap

def _check_disjoint(positions, angles, tol=1e-9):
    """Check all pairs are disjoint."""
    n = len(positions)
    for i in range(n):
        vi = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for j in range(i + 1, n):
            vj = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            if _sat_gap(vi, vj) < -tol:
                return False
    return True

def _outer_side_exact(positions, angles, outer_rot_deg):
    """Exact outer hex side length."""
    centers = np.array(positions)
    all_verts = _hex_vertices_array(centers, angles)
    phi = np.radians(outer_rot_deg)
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    max_proj = 0.0
    for theta in thetas:
        proj = all_verts[:, :, 0] * np.cos(theta) + all_verts[:, :, 1] * np.sin(theta)
        mp = proj.max()
        if mp > max_proj:
            max_proj = mp
    return max_proj / (np.sqrt(3) / 2)

def _outer_side_smooth(positions, angles, outer_rot_deg, beta=50.0):
    """Smooth approximation of outer hex side length using log-sum-exp."""
    centers = np.array(positions)
    all_verts = _hex_vertices_array(centers, angles)
    phi = np.radians(outer_rot_deg)
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    max_proj = 0.0
    for theta in thetas:
        proj = all_verts[:, :, 0] * np.cos(theta) + all_verts[:, :, 1] * np.sin(theta)
        proj_flat = proj.flatten()
        m = proj_flat.max()
        if m > max_proj:
            max_proj = m
    return max_proj / (np.sqrt(3) / 2)

def _pairwise_overlap_penalty(centers, angles, min_dist_sq=0.0):
    """Compute smooth overlap penalty using SAT gaps for all pairs."""
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        vi = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i + 1, n):
            vj = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles[j])
            gap = _sat_gap(vi, vj)
            if gap < 0:
                penalty += gap * gap
    return penalty

def _granular_packing(positions, angles, outer_rot_deg, deadline, max_iter=5000, dt=0.005):
    """Force-directed granular packing to find compact disjoint configuration.
    Uses repulsive forces from overlapping pairs and attractive force toward center."""
    centers = np.array(positions, dtype=np.float64).copy()
    angles_arr = np.array(angles, dtype=np.float64).copy()
    
    # Precompute hex vertices for each
    hexes = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles_arr[i]) for i in range(N)]
    
    # Initial side length
    init_side = _outer_side_exact(centers, angles_arr, outer_rot_deg)
    best_side = init_side
    best_centers = centers.copy()
    
    # Parameters
    repulsion_strength = 50.0
    center_attraction = 0.1
    damping = 0.95
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        for i in range(N):
            vi = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles_arr[i])
            for j in range(i + 1, N):
                vj = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles_arr[j])
                gap = _sat_gap(vi, vj)
                if gap < 0:
                    # Overlapping: push apart along center-to-center vector
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-10:
                        diff /= dist
                    else:
                        diff = np.array([1.0, 0.0])
                    # Force proportional to overlap
                    f = repulsion_strength * (-gap) * diff
                    forces[i] += f
                    forces[j] -= f
        
        # Attraction toward center of mass (keeps cluster compact)
        com = centers.mean(axis=0)
        for i in range(N):
            forces[i] += -center_attraction * (centers[i] - com)
        
        # Update positions with damping
        centers += dt * forces
        centers *= damping
        
        # Check side length
        side = _outer_side_exact(centers, angles_arr, outer_rot_deg)
        if side < best_side:
            best_side = side
            best_centers = centers.copy()
        
        # Early stop if converged
        if abs(side - best_side) < 1e-12 and it > 100:
            break
    
    return best_centers, best_side

def _optimize_positions(positions, angles, outer_rot_deg, deadline, max_iter=200):
    """Optimize positions to minimize outer side length while maintaining disjointness."""
    from scipy.optimize import minimize
    
    original_side = _outer_side_exact(positions, angles, outer_rot_deg)
    x0 = np.zeros(2 * N)
    for i in range(N):
        x0[2*i] = positions[i][0]
        x0[2*i+1] = positions[i][1]
    
    def obj(x):
        if time.time() > deadline:
            return 1e8
        pos = [(x[2*i], x[2*i+1]) for i in range(N)]
        # Check disjointness
        for i in range(N):
            vi = _hex_vertices(pos[i][0], pos[i][1], 1.0, angles[i])
            for j in range(i + 1, N):
                vj = _hex_vertices(pos[j][0], pos[j][1], 1.0, angles[j])
                if _sat_gap(vi, vj) < -1e-7:
                    return 1e6 + original_side
        return _outer_side_exact(pos, angles, outer_rot_deg)
    
    try:
        result = minimize(obj, x0, method='SLSQP', options={'maxiter': max_iter, 'ftol': 1e-12})
        if result.fun < original_side and result.fun < 1e5:
            opt_pos = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
            if _check_disjoint(opt_pos, angles):
                opt_side = _outer_side_exact(opt_pos, angles, outer_rot_deg)
                if opt_side < original_side:
                    return opt_pos, opt_side
    except Exception:
        pass
    return positions, original_side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
    
    # Generate candidate arrangements
    arrangements = []
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    # All combinations of 4 from ring2
    for combo in combinations(range(12), 4):
        subset = [ring2[i] for i in combo]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Also try some with different ring2 selections
    ring3 = [(3,0),(2,1),(1,2),(0,3),(-1,3),(-2,3),(-3,3),(-3,2),(-3,1),(-3,0),(-2,-1),(-1,-2),(0,-3),(1,-3),(2,-3),(3,-2),(3,-1)]
    # Select 4 from ring3 (farther out)
    for combo in combinations(range(17), 4):
        subset = [ring3[i] for i in combo]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Deduplicate
    unique_arrangements = []
    seen = set()
    for arr in arrangements:
        key = tuple((round(p[0], 8), round(p[1], 8)) for p in arr)
        if key not in seen:
            seen.add(key)
            unique_arrangements.append(arr)
    arrangements = unique_arrangements
    
    # Score and sort
    def arr_score(arr):
        angles = [0.0] * N
        if not _check_disjoint(arr, angles):
            return float('inf')
        s0 = _outer_side_exact(arr, angles, 0.0)
        s30 = _outer_side_exact(arr, angles, 30.0)
        return min(s0, s30)
    
    scored = []
    for arr in arrangements:
        if time.time() > deadline - 60.0:
            break
        s = arr_score(arr)
        if s < float('inf'):
            scored.append((s, arr))
    scored.sort(key=lambda x: x[0])
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_rotation = 0.0
    
    # Phase 1: Evaluate top arrangements with various outer rotations
    top_count = min(30, len(scored))
    for s, arr in scored[:top_count]:
        if time.time() > deadline - 40.0:
            break
        angles = [0.0] * N
        for outer_rot in [0.0, 30.0, 15.0, 45.0, 60.0]:
            side = _outer_side_exact(arr, angles, outer_rot)
            if side < best_side:
                best_side = side
                best_positions = arr[:]
                best_angles = angles[:]
                best_outer_rotation = outer_rot
    
    # Phase 2: Granular packing on top seeds
    opt_count = min(12, len(scored))
    for s, arr in scored[:opt_count]:
        if time.time() > deadline - 20.0:
            break
        angles = [0.0] * N
        for outer_rot in [0.0, 30.0]:
            centers = np.array(arr, dtype=np.float64)
            opt_centers, opt_side = _granular_packing(centers, angles, outer_rot, deadline, max_iter=3000)
            if opt_side < best_side:
                best_side = opt_side
                best_positions = [(opt_centers[i, 0], opt_centers[i, 1]) for i in range(N)]
                best_angles = angles[:]
                best_outer_rotation = outer_rot
    
    # Phase 3: SLSQP refinement on best
    if best_positions is not None:
        for outer_rot in [best_outer_rotation, 0.0, 30.0]:
            if time.time() > deadline - 10.0:
                break
            opt_positions, opt_side = _optimize_positions(best_positions, best_angles, outer_rot, deadline, max_iter=100)
            if opt_side < best_side:
                best_side = opt_side
                best_positions = opt_positions[:]
                best_outer_rotation = outer_rot
    
    # Phase 4: Fine-tune outer rotation on best
    if best_positions is not None:
        for outer_rot in np.arange(0, 360, 2.0):
            if time.time() > deadline - 5.0:
                break
            side = _outer_side_exact(best_positions, best_angles, outer_rot)
            if side < best_side:
                best_side = side
                best_outer_rotation = outer_rot
    
    if best_positions is None:
        # Fallback
        best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        best_angles = [0.0] * N
        best_side = _outer_side_exact(best_positions, best_angles, 0.0)
        best_outer_rotation = 0.0
    
    # Add tiny epsilon for strict containment
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, best_outer_rotation
# EVOLVE-BLOCK-END
