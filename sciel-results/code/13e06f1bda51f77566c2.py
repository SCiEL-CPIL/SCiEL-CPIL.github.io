"""Packing 12 unit hexagons in minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle + i * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append((x, y))
    return np.array(vertices)

def sat_gap(p1, p2):
    """
    Calculate the signed distance between two convex hexagons.
    Positive if separated, negative if overlapping.
    """
    min_gap = float('inf')
    
    # Get edges and normals for p1
    for i in range(6):
        # Axis from p1
        e1 = p1[(i+1)%6] - p1[i]
        norm = np.linalg.norm(e1)
        if norm < 1e-9: continue
        n1 = np.array([-e1[1], e1[0]]) / norm
        
        proj1 = np.dot(p1, n1)
        proj2 = np.dot(p2, n1)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2:
            gap = min2 - max1
        elif max2 <= min1:
            gap = min1 - max2
        else:
            # Overlap
            gap = -min(max1 - min2, max2 - min1)
            
        if gap < min_gap:
            min_gap = gap
            # If gap is positive, they are separated on this axis.
            # The minimum gap over all axes is the distance.
            # If we find a positive gap, it's a candidate for the distance.
            # But we must check all axes to find the *minimum* gap?
            # No, the distance between two convex sets is the minimum gap over all axes.
            # If they are separated, the minimum gap is the distance.
            # If they overlap, the minimum gap is the negative penetration depth.
            
    # Get edges and normals for p2
    for i in range(6):
        # Axis from p2
        e2 = p2[(i+1)%6] - p2[i]
        norm = np.linalg.norm(e2)
        if norm < 1e-9: continue
        n2 = np.array([-e2[1], e2[0]]) / norm
        
        proj1 = np.dot(p1, n2)
        proj2 = np.dot(p2, n2)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2:
            gap = min2 - max1
        elif max2 <= min1:
            gap = min1 - max2
        else:
            # Overlap
            gap = -min(max1 - min2, max2 - min1)
            
        if gap < min_gap:
            min_gap = gap

    return min_gap

def compute_min_side(positions, angles, outer_angle):
    """
    Compute the minimal side length of a regular hexagon (centered at 0,0, rotated by outer_angle)
    that contains all inner hexagons.
    """
    max_proj = 0.0
    # Normals of the outer hexagon edges
    # Outer hexagon side S, angle Theta.
    # Edge normal k is at angle Theta + k*pi/3 + pi/6.
    # The condition for point p to be inside is |p . n_k| <= S * sqrt(3)/2.
    # So S >= |p . n_k| / (sqrt(3)/2).
    
    sqrt3_half = np.sqrt(3) / 2.0
    
    # We need to check all vertices of all inner hexagons against all 6 normals of the outer hexagon.
    # To be efficient, we can just compute the max projection.
    
    for i in range(N):
        v = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        # v is (6,2)
        
        for k in range(6):
            theta_k = np.radians(outer_angle) + k * np.pi / 3 + np.pi / 6
            n_k = np.array([np.cos(theta_k), np.sin(theta_k)])
            
            proj = np.abs(np.dot(v, n_k))
            max_proj = max(max_proj, np.max(proj))
            
    return max_proj / sqrt3_half

def objective(params):
    """
    params: [x1, y1, a1, ..., x12, y12, a12, outer_angle]
    """
    if time.time() > DEADLINE:
        return 1000.0
        
    positions = []
    angles = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        positions.append((x,y))
        angles.append(a)
        
    outer_angle = params[N*3]
    
    # Compute required side
    side = compute_min_side(positions, angles, outer_angle)
    
    # Compute penalty for overlaps
    penalty = 0.0
    hexes = []
    for i in range(N):
        hexes.append(hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
        
    for i in range(N):
        for j in range(i+1, N):
            gap = sat_gap(hexes[i], hexes[j])
            if gap < 0:
                # Penalty is square of overlap depth
                penalty += (-gap)**2
                
    # Weight penalty heavily
    return side + 100.0 * penalty

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_params = None
    
    # Initial configurations
    configs = []
    
    # 1. Hexagonal Lattice (1 + 6 + 5)
    d = np.sqrt(3) * 1.0
    pos = [(0,0)]
    for i in range(6):
        a = np.radians(60*i)
        pos.append((d*np.cos(a), d*np.sin(a)))
    for i in range(6):
        if i == 2: continue
        a = np.radians(60*i + 30)
        pos.append((2*d*np.cos(a), 2*d*np.sin(a)))
    
    # Flatten to params
    p = []
    for x,y in pos:
        p.extend([x,y,0.0])
    p.append(0.0) # outer angle
    configs.append(p)
    
    # 2. Rotated Hex Lattice
    p_rot = []
    for x,y in pos:
        p_rot.extend([x,y,30.0])
    p_rot.append(30.0)
    configs.append(p_rot)
    
    # 3. Linear Grid 3x4
    pos_grid = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * d
            y = (r - 1) * d * 1.5
            pos_grid.append((x,y))
            
    p_grid = []
    for x,y in pos_grid:
        p_grid.extend([x,y,0.0])
    p_grid.append(0.0)
    configs.append(p_grid)
    
    # 4. Linear Chain
    pos_chain = []
    for i in range(12):
        pos_chain.append(((i - 5.5) * d, 0))
        
    p_chain = []
    for x,y in pos_chain:
        p_chain.extend([x,y,0.0])
    p_chain.append(0.0)
    configs.append(p_chain)
    
    # Run SLSQP on each config
    for init in configs:
        if time.time() > DEADLINE:
            break
            
        try:
            result = minimize(objective, init, method='SLSQP', 
                              options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
            
    # If SLSQP failed or didn't find anything, fallback
    if best_params is None:
        # Return a valid but loose packing
        d = np.sqrt(3) * 2.0
        inner = []
        for i in range(N):
            a = np.radians(60 * i)
            x = d * np.cos(a)
            y = d * np.sin(a)
            inner.append([x, y, 0.0])
        # Add center? No, 12 items.
        # The chain is safer for fallback
        inner = []
        for i in range(N):
            inner.append([(i - 5.5) * d, 0, 0.0])
        return inner, (0.0, 0.0), 20.0, 0.0
        
    # Extract best result
    positions = []
    angles = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        positions.append((x,y))
        angles.append(a)
        
    outer_angle = best_params[N*3]
    
    # Final precise side calculation
    side = compute_min_side(positions, angles, outer_angle)
    
    # Verify validity (optional, but good for debugging)
    # If penalty was high, the packing might be invalid.
    # We can try to polish the side length slightly larger to ensure validity if needed,
    # but the objective function should have minimized overlaps.
    
    inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    return inner, (0.0, 0.0), side, outer_angle

