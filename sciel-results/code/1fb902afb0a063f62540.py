"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_rad, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation in radians."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def sat_overlap_depth(v1, v2):
    """
    Compute the minimum translation depth for two convex polygons using SAT.
    Returns 0 if they don't overlap, otherwise the minimum separation distance.
    """
    min_depth = float('inf')
    n1, n2 = len(v1), len(v2)
    
    for poly, n in [(v1, n1), (v2, n2)]:
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # Gap check
            if max1 <= min2 + 1e-9 or max2 <= min1 + 1e-9:
                return 0.0
            
            depth = min(max1, max2) - max(min1, min2)
            if depth < min_depth:
                min_depth = depth
    
    return min_depth

def compute_side_and_overlap(params, beta=50.0):
    """
    Compute a smooth objective: outer side length + soft overlap penalty.
    params: [x1, y1, a1, ..., x12, y12, a12]
    Returns (side_length, overlap_penalty)
    """
    positions = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    # Precompute vertices
    verts_list = [hex_vertices(positions[i,0], positions[i,1], angles[i]) for i in range(N)]
    
    # Compute soft overlap penalty
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap_depth(verts_list[i], verts_list[j])
            if d > 1e-8:
                # Soft penalty: use exp to make it smooth
                overlap_penalty += math.exp(-beta * d)
    
    # Compute min outer side length
    # Outer hexagon centered at (0,0), angle 0
    # Apothem = S * sqrt(3) / 2
    # For each of the 6 edge normals, compute max projection
    max_proj = 0.0
    for k in range(6):
        normal_angle = math.pi/6 + k * math.pi/3
        n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
        for i in range(N):
            proj = verts_list[i] @ n
            max_proj = max(max_proj, np.max(proj))
    
    side = max_proj / (math.sqrt(3) / 2.0)
    
    return side, overlap_penalty

def objective(params, beta=50.0):
    """Objective function: side + large penalty if overlapping."""
    side, overlap = compute_side_and_overlap(params, beta)
    # If overlap penalty is significant, add a large penalty
    if overlap > 1e-6:
        return side + 1000.0 * overlap + 100.0 * min(overlap, 1.0)
    return side

def objective_smooth(params, beta=50.0):
    """Smoother objective for gradient-based methods."""
    side, overlap = compute_side_and_overlap(params, beta)
    # Use a soft penalty that's always differentiable
    return side + 50.0 * overlap + 10.0 * overlap * overlap

def make_hex_cluster_config():
    """
    Create a valid hexagonal cluster: 1 center, 6 around, 5 more.
    This gives a valid non-overlapping configuration.
    """
    positions = []
    angles = []
    
    # Center hexagon
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 surrounding hexagons (at distance sqrt(3) from center)
    dist = math.sqrt(3)
    for k in range(6):
        angle = k * math.pi / 3
        cx = dist * math.cos(angle)
        cy = dist * math.sin(angle)
        positions.append([cx, cy])
        angles.append(0.0)
    
    # 5 more hexagons - place them at the next ring
    # The next ring has positions at distance 2*sqrt(3) or at specific offsets
    # Let's place them at angles that don't conflict
    dist2 = 2 * math.sqrt(3)
    for k in range(5):
        angle = k * 2 * math.pi / 5 + math.pi / 10
        cx = dist2 * math.cos(angle)
        cy = dist2 * math.sin(angle)
        positions.append([cx, cy])
        angles.append(0.0)
    
    pos_arr = np.array(positions).flatten()
    ang_arr = np.array(angles)
    return np.concatenate([pos_arr, ang_arr])

def make_ring_config():
    """
    Create a configuration with hexagons arranged in two rings.
    Inner ring: 6 hexagons, outer ring: 5 hexagons, plus 1 center.
    """
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Inner ring: 6 hexagons at distance sqrt(3)
    d1 = math.sqrt(3)
    for k in range(6):
        angle = k * math.pi / 3
        cx = d1 * math.cos(angle)
        cy = d1 * math.sin(angle)
        positions.append([cx, cy])
        angles.append(0.0)
    
    # Outer ring: 5 hexagons at distance ~2*sqrt(3)
    d2 = 2 * math.sqrt(3)
    for k in range(5):
        angle = k * 2 * math.pi / 5
        cx = d2 * math.cos(angle)
        cy = d2 * math.sin(angle)
        positions.append([cx, cy])
        angles.append(0.0)
    
    pos_arr = np.array(positions).flatten()
    ang_arr = np.array(angles)
    return np.concatenate([pos_arr, ang_arr])

def make_grid_config():
    """
    Create a grid-like configuration that's valid.
    Use hexagonal grid spacing.
    """
    positions = []
    angles = []
    
    # Hexagonal grid: rows offset
    # For unit hexagons, center-to-center distance is sqrt(3) for adjacent
    dx = math.sqrt(3)
    dy = 1.5  # vertical spacing for hex grid
    
    # 4 rows x 3 cols
    for row in range(4):
        for col in range(3):
            cx = col * dx
            cy = row * dy
            if row % 2 == 1:
                cx += dx / 2
            positions.append([cx, cy])
            angles.append(0.0)
    
    # Center the configuration
    pos_arr = np.array(positions)
    center = pos_arr.mean(axis=0)
    pos_arr = pos_arr - center
    pos_arr = pos_arr.flatten()
    ang_arr = np.array(angles)
    return np.concatenate([pos_arr, ang_arr])

def make_compact_hex_config():
    """
    Create a compact hexagonal arrangement.
    12 hexagons: try a 3x4 hex grid pattern.
    """
    positions = []
    angles = []
    
    # Use a hex grid with proper spacing
    # Adjacent hex centers: sqrt(3) apart
    d = math.sqrt(3)
    
    # Layout: 3 columns, 4 rows with hex offset
    for row in range(4):
        for col in range(3):
            cx = col * d
            cy = row * 1.5
            if row % 2 == 1:
                cx += d / 2
            positions.append([cx, cy])
            angles.append(0.0)
    
    pos_arr = np.array(positions)
    center = pos_arr.mean(axis=0)
    pos_arr = pos_arr - center
    pos_arr = pos_arr.flatten()
    ang_arr = np.array(angles)
    return np.concatenate([pos_arr, ang_arr])

def local_polish(params, deadline):
    """
    Do a high-precision polish using L-BFGS-B with tighter tolerances.
    """
    start = time.time()
    
    # Use smooth objective for gradient-based method
    res = minimize(
        objective_smooth,
        params,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
    )
    
    if time.time() > deadline:
        return params
    
    # Check if the polished result is valid (no overlaps)
    side, overlap = compute_side_and_overlap(res.x)
    if overlap < 1e-6:
        return res.x
    
    # If not valid, try Nelder-Mead polish
    res2 = minimize(
        objective,
        params,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
    )
    
    if time.time() > deadline:
        return params
    
    side2, overlap2 = compute_side_and_overlap(res2.x)
    if overlap2 < 1e-6:
        return res2.x
    
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    best_side = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    configs = []
    configs.append(make_hex_cluster_config())
    configs.append(make_ring_config())
    configs.append(make_grid_config())
    configs.append(make_compact_hex_config())
    
    # Add random configurations
    np.random.seed(123)
    for seed_offset in range(5):
        np.random.seed(123 + seed_offset)
        p = np.random.randn(N, 2) * 2.0
        a = np.random.rand(N) * math.pi
        configs.append(np.concatenate([p.flatten(), a]))
    
    # For each configuration, run optimization
    for cfg_idx, x0 in enumerate(configs):
        if time.time() > deadline - 15:
            break
        
        # First, check if this config is already valid
        side0, overlap0 = compute_side_and_overlap(x0)
        if overlap0 < 1e-6 and side0 < best_side:
            best_side = side0
            best_params = x0.copy()
        
        # Run Nelder-Mead optimization
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if time.time() > deadline - 10:
            break
        
        # Check result
        side_r, overlap_r = compute_side_and_overlap(res.x)
        if overlap_r < 1e-6 and side_r < best_side:
            best_side = side_r
            best_params = res.x.copy()
        
        # Also try with the smoothed objective
        if time.time() > deadline - 10:
            break
            
        res2 = minimize(
            objective_smooth,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if time.time() > deadline - 10:
            break
            
        side_r2, overlap_r2 = compute_side_and_overlap(res2.x)
        if overlap_r2 < 1e-6 and side_r2 < best_side:
            best_side = side_r2
            best_params = res2.x.copy()
    
    # Polish the best solution
    if best_params is not None and time.time() < deadline - 5:
        polished = local_polish(best_params, deadline)
        if polished is not None:
            side_p, overlap_p = compute_side_and_overlap(polished)
            if overlap_p < 1e-6 and side_p < best_side:
                best_side = side_p
                best_params = polished
    
    # Fallback: if no valid solution, use a known valid configuration
    if best_params is None or best_side >= 100.0:
        # Use the hex cluster which should be valid
        best_params = make_hex_cluster_config()
        best_side, _ = compute_side_and_overlap(best_params)
        
        # If still invalid, try to fix by spreading out
        if best_side >= 100.0:
            # Scale positions to ensure no overlap
            positions = best_params[:2*N].reshape(N, 2)
            angles = best_params[2*N:]
            
            # Increase distances
            scale = 1.5
            positions = positions * scale
            best_params = np.concatenate([positions.flatten(), angles])
            best_side, _ = compute_side_and_overlap(best_params)
    
    # Extract results
    positions = best_params[:2*N].reshape(N, 2)
    angles = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([positions[i,0], positions[i,1], math.degrees(angles[i])])
        
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
