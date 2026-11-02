import math
import numpy as np
import time
import random

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# Precompute unit hexagon vertices (flat top? or pointy top?)
# Standard: Pointy top, center at (0,0)
# Vertices at angles 30, 90, 150, 210, 270, 330 degrees?
# Let's use flat top for easier symmetry with outer hexagon if outer is flat top.
# Outer hexagon: Flat top. Normals at 0, 60, 120, 180, 240, 300.
# Inner hexagons can rotate.
# Let's define unit hexagon vertices for angle 0 (flat top).
# Vertices:
# (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
UNIT_HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3) / 2.0],
    [-0.5, math.sqrt(3) / 2.0],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3) / 2.0],
    [0.5, -math.sqrt(3) / 2.0]
])

def get_hex_vertices(center, angle):
    """Returns 6x2 array of vertices for a unit hexagon."""
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return UNIT_HEX_VERTS @ R.T + center

def hex_distance(V1, V2):
    """
    Calculates the minimum distance between two convex polygons (hexagons).
    Returns negative value if overlapping (penetration depth).
    Uses SAT.
    """
    # Compute normals for V1
    # Normal to edge (v[i], v[i+1]) is perpendicular to (v[i+1]-v[i])
    # For a convex polygon, outward normals.
    # We need to compute support functions h(n) = max(n.dot(v))
    
    # To speed up, we can just compute the 12 normals and project.
    # Normals for V1
    normals = []
    for i in range(6):
        v1 = V1[i]
        v2 = V1[(i + 1) % 6]
        edge = v2 - v1
        # Outward normal: rotate edge 90 deg clockwise?
        # If vertices are CCW, edge is v2-v1. Normal is (edge.y, -edge.x) normalized.
        # Let's check orientation of UNIT_HEX_VERTS.
        # (1,0) -> (0.5, 0.866). Edge = (-0.5, 0.866).
        # Normal (0.866, 0.5). Dot with center (0,0) is 0.
        # Dot with vertex (1,0) is 0.866.
        # Dot with vertex (0.5, 0.866) is 0.433+0.433=0.866.
        # This normal points outward.
        nx = edge[1]
        ny = -edge[0]
        norm = math.sqrt(nx*nx + ny*ny)
        if norm > 1e-9:
            normals.append((nx/norm, ny/norm))
            
    # Normals for V2
    for i in range(6):
        v1 = V2[i]
        v2 = V2[(i + 1) % 6]
        edge = v2 - v1
        nx = edge[1]
        ny = -edge[0]
        norm = math.sqrt(nx*nx + ny*ny)
        if norm > 1e-9:
            normals.append((nx/norm, ny/norm))
            
    min_sep = float('inf')
    
    # For each normal, compute max projection of V1 and min projection of V2 (or vice versa)
    # Separation = max_proj_V1 - min_proj_V2
    # If any separation > 0, they are separated by that amount?
    # No, if they are separated, ALL separations are > 0. The distance is the MINIMUM separation.
    # Wait. If they are far apart, the minimum separation is the distance.
    # If they overlap, all separations are < 0. The "distance" is max(0, min_sep).
    # But if they overlap, min_sep is negative. We want the penetration depth.
    # Penetration depth = max(0, -min_sep)?
    # Let's check.
    # If V1 is to the right of V2. Normal (1,0).
    # max_proj_V1 = 1. min_proj_V2 = -1. Sep = 2.
    # Normal (-1,0). max_proj_V1 = -1. min_proj_V2 = 1. Sep = -2.
    # Min sep = -2.
    # Distance should be 2.
    # So Distance = -min_sep if min_sep < 0?
    # Actually, the distance is the minimum over all normals of the separation?
    # No. The distance between two convex sets is the minimum distance between any point in A and any point in B.
    # By SAT, if they are disjoint, the distance is the minimum separation along any normal?
    # Yes.
    # If they overlap, the minimum separation is negative. The magnitude is the penetration depth?
    # Not exactly, but for regular polygons, it's a good proxy.
    
    for nx, ny in normals:
        # Project V1
        max_p1 = -float('inf')
        for v in V1:
            p = v[0]*nx + v[1]*ny
            if p > max_p1: max_p1 = p
            
        # Project V2
        min_p2 = float('inf')
        for v in V2:
            p = v[0]*nx + v[1]*ny
            if p < min_p2: min_p2 = p
            
        sep = max_p1 - min_p2
        if sep < min_sep:
            min_sep = sep
            
    # If min_sep > 0, they are separated by min_sep.
    # If min_sep < 0, they overlap.
    # The "distance" (signed) is min_sep.
    # We want to penalize overlap.
    # Return min_sep.
    return min_sep

def outer_hex_penalty(V, S):
    """
    Calculate penetration depth of hexagon V into outer hexagon of side S centered at (0,0) with angle 0.
    Outer hexagon vertices are at distance S from center? No, side length S.
    Apothem = S * sqrt(3) / 2.
    Outer hexagon is defined by 6 half-planes.
    Normal k: angle k * 60 deg.
    Distance from origin to edge = Apothem.
    Condition: n_k . v <= Apothem for all v in V.
    Violation = max(0, max_v(n_k . v) - Apothem).
    """
    apothem = S * math.sqrt(3) / 2.0
    max_pen = 0.0
    for k in range(6):
        ang = k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        max_proj = -float('inf')
        for v in V:
            p = v[0]*nx + v[1]*ny
            if p > max_proj: max_proj = p
        pen = max_proj - apothem
        if pen > max_pen:
            max_pen = pen
    return max_pen

def total_penalty(positions, angles, S):
    """
    positions: list of (x,y)
    angles: list of theta
    S: outer side length
    """
    verts = []
    for i in range(N):
        x, y = positions[i]
        theta = angles[i]
        V = get_hex_vertices((x, y), theta)
        verts.append(V)
        
    pen = 0.0
    # Pairwise
    for i in range(N):
        for j in range(i+1, N):
            d = hex_distance(verts[i], verts[j])
            if d < 0:
                pen += d * d
                
    # Boundary
    for i in range(N):
        p = outer_hex_penalty(verts[i], S)
        if p > 0:
            pen += p * p
            
    return pen

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback
        return [[0.0, 0.0, 0.0] for _ in range(N)], [0.0, 0.0], 4.0, 0.0

    # Initial guess: Lattice packing
    # Center at 0,0
    # 1 in center, 6 around, 4 more?
    # Let's use a known good packing or generate one.
    # Simple lattice:
    # (0,0)
    # (1.5, 0.866), (1.5, -0.866), (0, 1.732), (-1.5, 0.866), (-1.5, -0.866), (0, -1.732)
    # This is 7 hexagons.
    # Add 4 more.
    # (3, 0), (-3, 0), (0, 3.464), (0, -3.464)?
    # Let's just start with random positions in a box and let SA find it.
    # Or better, start with a dense random cluster.
    
    random.seed(42)
    
    # Initial positions: random in circle of radius 3
    positions = []
    angles = []
    for _ in range(N):
        r = random.uniform(0, 3.0)
        theta = random.uniform(0, 2*math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        positions.append((x, y))
        angles.append(random.uniform(0, math.pi/3)) # Symmetry
        
    # Binary search for S
    S_low = 3.5
    S_high = 4.5
    
    best_valid_S = S_high
    best_positions = positions
    best_angles = angles
    
    # We will run SA for a few S values
    # To save time, we run SA for S_high first to get a good config, then refine.
    
    def run_sa(S, time_budget):
        start_t = time.time()
        T = 1.0
        T_min = 1e-6
        cooling = 0.999
        
        cur_pos = [p[:] for p in positions]
        cur_ang = angles[:]
        cur_pen = total_penalty(cur_pos, cur_ang, S)
        
        best_pen = cur_pen
        best_pos = [p[:] for p in cur_pos]
        best_ang = cur_ang[:]
        
        while time.time() - start_t < time_budget and T > T_min:
            # Generate new state
            new_pos = [p[:] for p in cur_pos]
            new_ang = cur_ang[:]
            
            # Move one hexagon
            i = random.randint(0, N-1)
            dx = random.gauss(0, 0.2)
            dy = random.gauss(0, 0.2)
            dth = random.gauss(0, 0.1)
            
            new_pos[i][0] += dx
            new_pos[i][1] += dy
            new_ang[i] += dth
            
            new_pen = total_penalty(new_pos, new_ang, S)
            
            # Acceptance
            if new_pen < cur_pen or random.random() < math.exp(-(new_pen - cur_pen) / T):
                cur_pos = new_pos
                cur_ang = new_ang
                cur_pen = new_pen
                
                if cur_pen < best_pen:
                    best_pen = cur_pen
                    best_pos = [p[:] for p in cur_pos]
                    best_ang = cur_ang[:]
                    
            T *= cooling
            
        return best_pen, best_pos, best_ang

    # Step 1: Find a valid packing for S_high
    pen, pos, ang = run_sa(S_high, 10.0)
    if pen < 1e-6:
        best_valid_S = S_high
        best_positions = pos
        best_angles = ang
        
    # Step 2: Binary search
    for _ in range(10):
        S_mid = (S_low + S_high) / 2.0
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 20.0:
            break
            
        # Use best known config as start
        pos_start = best_positions
        ang_start = best_angles
        
        pen, pos, ang = run_sa(S_mid, 8.0)
        
        if pen < 1e-6:
            S_low = S_mid
            best_valid_S = S_mid
            best_positions = pos
            best_angles = ang
        else:
            S_high = S_mid
            
    # Final polish
    # Run SA again for best_valid_S with more time
    remaining = TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)
    if remaining > 5.0:
        pen, pos, ang = run_sa(best_valid_S, remaining - 2.0)
        if pen < 1e-6:
            best_positions = pos
            best_angles = ang
            # Try to reduce S slightly?
            # Just report current S.

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), float(best_angles[i])])
        
    return inner, [0.0, 0.0], float(best_valid_S), 0.0

