"""
Strategy: Constrained Lattice Relaxation with Smooth SDF Objective.
Instead of generic Nelder-Mead on 39 variables, we:
1. Initialize 12 hexagons on a compact hexagonal lattice (guaranteed non-overlap).
2. Use a Smooth Signed Distance Function (SDF) for hexagon-hexagon and hexagon-boundary distances.
3. Optimize using L-BFGS-B (gradient-based) which is much faster per iteration and handles
   smooth objectives well.
4. The objective minimizes the outer hexagon side length, with a large penalty for any
   negative SDF (overlap).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _sdf_hexagon_point(px, py, cx, cy, angle_deg, side=1.0):
    """
    Smooth Signed Distance Function from a point (px, py) to a regular hexagon
    centered at (cx, cy) with rotation angle_deg and side length side.
    Positive: outside, Negative: inside.
    """
    # Transform point to hexagon local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    
    # Rotate point by -angle
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    x = dx * cos_a - dy * sin_a
    y = dx * sin_a + dy * cos_a
    
    # Hexagon in local frame (flat-top, side=1)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Inradius r = side * sqrt(3) / 2
    # Circumradius R = side
    
    # The hexagon can be defined by 6 half-planes.
    # Normal directions are 30, 90, 150, 210, 270, 330 degrees?
    # For flat-top hex:
    # Top edge is horizontal? No, flat-top means top and bottom are flat edges.
    # Vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Wait, standard math angle 0 is (1,0).
    # If vertices are at 0, 60, ... then edges are at 30, 90, ...
    # Let's check the normal of the edge between (1,0) and (0.5, sqrt(3)/2).
    # Midpoint: (0.75, sqrt(3)/4). Normal points outward.
    # Vector: (-0.5, sqrt(3)/2). Normal: (sqrt(3)/2, 0.5) -> angle 30 deg.
    # So face normals are at 30, 90, 150, 210, 270, 330.
    
    # Distance to each face line: d_i = x * n_i.x + y * n_i.y - r
    # where r = side * sqrt(3) / 2 is the inradius.
    # The SDF is the minimum over all faces of (d_i) if outside, but we need smooth min.
    
    r = side * np.sqrt(3) / 2.0
    
    # Face normals
    normals = []
    for i in range(6):
        theta = np.radians(30 + i * 60)
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # Project local point onto normals
    # dists = x * nx + y * ny - r
    dists = x * normals[:, 0] + y * normals[:, 1] - r
    
    # Smooth minimum (softmin) of dists
    # Since we want the distance to the boundary, and dists are signed distances to planes,
    # the SDF is min(dists) for convex shapes?
    # Actually, for a convex polygon defined by half-planes n_i . x <= r,
    # the signed distance is min_i (n_i . x - r).
    # This is exact for the SDF of a convex polytope?
    # Yes, for convex polytopes, the SDF is the minimum of the signed distances to the faces.
    
    # To make it differentiable, we use a softmin.
    # softmin(d) = -logsumexp(-d) / beta
    # Or simply use a smoothed min.
    # Let's use a simple smooth min:
    # m = min(dists)
    # d_smooth = m - log(sum(exp(-dists/m)) * m) ... this is tricky if m is near 0 or negative.
    
    # Alternative: Just use the exact min for the penalty, but L-BFGS needs gradients.
    # We can compute gradients numerically or analytically for the softmin.
    
    # Let's use a temperature-based softmin.
    beta = 10.0  # Higher beta = sharper min
    # To avoid overflow in exp, subtract max
    d_max = np.max(dists)
    exp_dists = np.exp(-beta * (dists - d_max))
    sum_exp = np.sum(exp_dists)
    
    # softmin = d_max - (1/beta) * log(sum_exp)
    # Wait, softmin(x) = min(x) approx.
    # If we want min(dists), and dists can be negative (inside),
    # standard softmin is -logsumexp(-x).
    # Let's use:
    # d_soft = - (1/beta) * log( sum( exp(-beta * dists) ) )
    # This is negative if all dists are positive?
    # If dists = [1, 2], -beta*dists = [-10, -20]. exp = [4e-5, 2e-9]. sum ~ 4e-5.
    # log(4e-5) ~ -10. -1/beta * (-10) = 1. Correct.
    # If dists = [-1, -2], -beta*dists = [10, 20]. exp = [2e4, 4e8]. sum ~ 4e8.
    # log(4e8) ~ 20. -1/beta * 20 = -2. Correct (closest face is at -2).
    
    # However, if dists are mixed, it works too.
    # Problem: numerical stability if beta * dists is large.
    # We shifted by d_max? No, for -logsumexp(-x), we should shift by max(x) to prevent overflow in exp(-beta * x).
    # Let m = max(dists).
    # exp(-beta * (dists - m)) = exp(-beta * dists + beta * m)
    # sum = exp(beta * m) * sum(exp(-beta * dists))
    # log(sum) = beta * m + log(sum(exp(-beta * dists)))
    # d_soft = -1/beta * (beta * m + log(...)) = -m - 1/beta * log(...)
    
    # Let's implement stable version:
    m = np.max(dists)
    exp_terms = np.exp(-beta * (dists - m))
    log_sum = np.log(np.sum(exp_terms))
    d_soft = -m - (1.0 / beta) * log_sum
    
    return d_soft

def _sdf_hex_hex(params, i, j):
    """
    Approximate SDF between two hexagons i and j.
    We sample points on the boundary of hex i and check SDF to hex j, and vice versa.
    For optimization, we can use the minimum of these samples.
    To be smooth, we take a softmin of the sampled distances.
    """
    cx1, cy1, a1 = params[3*i], params[3*i+1], params[3*i+2]
    cx2, cy2, a2 = params[3*j], params[3*j+1], params[3*j+2]
    
    # Sample 12 points on boundary of hex 1
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    # Circumradius is 1.0
    pts1 = np.array([
        [cx1 + np.cos(ang), cy1 + np.sin(ang)] for ang in angles
    ])
    
    # Calculate SDF of these points to hex 2
    dists1_to_2 = []
    for px, py in pts1:
        d = _sdf_hexagon_point(px, py, cx2, cy2, a2, 1.0)
        dists1_to_2.append(d)
    
    # Sample 12 points on boundary of hex 2
    pts2 = np.array([
        [cx2 + np.cos(ang), cy2 + np.sin(ang)] for ang in angles
    ])
    
    dists2_to_1 = []
    for px, py in pts2:
        d = _sdf_hexagon_point(px, py, cx1, cy1, a1, 1.0)
        dists2_to_1.append(d)
        
    # Combine all distances
    all_dists = np.array(dists1_to_2 + dists2_to_1)
    
    # We want the minimum distance. If they overlap, min dist is negative.
    # Use softmin to keep it differentiable.
    beta = 10.0
    m = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - m))
    log_sum = np.log(np.sum(exp_terms))
    d_soft = -m - (1.0 / beta) * log_sum
    
    return d_soft

def _outer_hex_side(params, outer_cx, outer_cy, outer_angle):
    """
    Calculate the minimum side length of the outer hexagon to contain all inner hexagons.
    """
    max_inradius = 0.0
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        verts = _hex_vertices(cx, cy, ang, 1.0)
        for vx, vy in verts:
            dx = vx - outer_cx
            dy = vy - outer_cy
            # Project onto 6 face normals of outer hex
            for k in range(6):
                theta = np.radians(outer_angle) + k * np.pi / 3 + np.pi / 6
                nx = np.cos(theta)
                ny = np.sin(theta)
                dist = dx * nx + dy * ny
                if dist > max_inradius:
                    max_inradius = dist
                    
    return 2.0 * max_inradius / np.sqrt(3)

def _objective(params):
    """
    Objective: Outer side length + Penalty for overlaps.
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, ocx, ocy, oangle]
    """
    # Extract outer params
    ocx = params[36]
    ocy = params[37]
    oangle = params[38]
    
    # Calculate outer side
    side = _outer_hex_side(params, ocx, ocy, oangle)
    
    # Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sdf_hex_hex(params, i, j)
            # If d < 0, they overlap.
            # Penalty should be large and smooth.
            # Use quadratic penalty for negative distances
            if d < 0:
                penalty += d * d * 100.0
                
    return side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Initialization: Compact hexagonal lattice
    # 12 hexagons.
    # Center hex + 6 around + 5 more?
    # Let's use a "flower" shape: 1 center, 6 ring, 5 partial ring?
    # Or a 3x4 rectangle of hexagons?
    # Axial coords for a compact cluster:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 total
    # Next ring: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # We need 5 more.
    # Let's pick: (2,0), (1,1), (0,2), (-1,2), (-2,1)
    
    axial_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    sqrt3 = np.sqrt(3)
    
    init_params = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        init_params.extend([x, y, 0.0])
        
    # Center the cluster
    xs = [init_params[i*3] for i in range(N)]
    ys = [init_params[i*3+1] for i in range(N)]
    cx_mean = np.mean(xs)
    cy_mean = np.mean(ys)
    
    for i in range(N):
        init_params[i*3] -= cx_mean
        init_params[i*3+1] -= cy_mean
        
    # Initial outer hexagon
    outer_cx = 0.0
    outer_cy = 0.0
    outer_angle = 0.0
    
    init_params.extend([outer_cx, outer_cy, outer_angle])
    
    best_params = np.array(init_params)
    best_side = _outer_hex_side(best_params, outer_cx, outer_cy, outer_angle)
    
    # Try different initial outer angles to find a better starting box
    for oa in [0, 30, 60, 90, 120, 150]:
        s = _outer_hex_side(best_params, 0, 0, oa)
        if s < best_side:
            best_side = s
            best_params[38] = oa
            
    # 2. Optimization with L-BFGS-B
    # We will run multiple restarts with slight perturbations
    
    for attempt in range(5):
        if time.time() > deadline - 20:
            break
            
        x0 = best_params.copy()
        if attempt > 0:
            # Perturb inner positions and angles slightly
            # Do not perturb outer center/angle too much initially
            for i in range(N):
                x0[i*3] += np.random.normal(0, 0.1)
                x0[i*3+1] += np.random.normal(0, 0.1)
                x0[i*3+2] += np.random.normal(0, 5) # degrees
            
        try:
            # Use L-BFGS-B
            # We need to be careful with the penalty term.
            # The SDF softmin is smooth.
            result = minimize(_objective, x0, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < best_side + 1e-5: # Allow small tolerance for penalty
                # Verify validity roughly
                # Check if any SDF is significantly negative
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = _sdf_hex_hex(result.x, i, j)
                        if d < -1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Recalculate side length without penalty
                    ocx, ocy, oa = result.x[36], result.x[37], result.x[38]
                    side = _outer_hex_side(result.x, ocx, ocy, oa)
                    if side < best_side:
                        best_side = side
                        best_params = result.x.copy()
                        
        except Exception as e:
            pass
            
    # Final Polish: Nelder-Mead on the best solution found
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_side + 1e-5:
                # Check validity
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = _sdf_hex_hex(result.x, i, j)
                        if d < -1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    ocx, ocy, oa = result.x[36], result.x[37], result.x[38]
                    side = _outer_hex_side(result.x, ocx, ocy, oa)
                    if side < best_side:
                        best_side = side
                        best_params = result.x.copy()
        except Exception:
            pass

    # Extract result
    inner = []
    for i in range(N):
        x = float(best_params[i*3])
        y = float(best_params[i*3+1])
        a = float(best_params[i*3+2])
        inner.append([x, y, a])
        
    outer_cx = float(best_params[36])
    outer_cy = float(best_params[37])
    outer_angle = float(best_params[38])
    
    side = float(best_side)
    
    return inner, (outer_cx, outer_cy), side, outer_angle

# EVOLVE-BLOCK-END
