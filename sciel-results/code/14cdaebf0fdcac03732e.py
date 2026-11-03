"""Packing 11 unit hexagons in a minimal outer regular hexagon."""
import time
import numpy as np

N = 11
DEADLINE = 130.0  # safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg):
    """Return the 6 vertices of a unit regular hexagon centered at (cx, cy) with rotation angle_deg."""
    a = np.deg2rad(angle_deg)
    # Vertices at angles a, a+60, ..., a+300 degrees, radius 1
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def poly_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    n1 = len(v1)
    n2 = len(v2)
    for i in range(n1):
        # Edge normal for v1
        edge = v1[(i + 1) % n1] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        # Project both polygons onto the normal
        p1 = np.dot(v1, normal)
        p2 = np.dot(v2, normal)
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
    for i in range(n2):
        edge = v2[(i + 1) % n2] - v2[i]
        normal = np.array([-edge[1], edge[0]])
        p1 = np.dot(v1, normal)
        p2 = np.dot(v2, normal)
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
    return True

def overlap_penalty(v1, v2):
    """Return a penalty if polygons overlap, 0 otherwise. Uses distance-based penalty."""
    if not poly_overlap(v1, v2):
        return 0.0
    # Compute minimum distance between vertices of one to edges of other as a proxy
    # For simplicity, use a large penalty
    return 100.0

def outer_hex_side_length(inner_positions, inner_angles):
    """Compute the minimum side length of a regular hexagon that contains all inner hexagons.
    The outer hexagon is centered at origin with rotation 0 for simplicity (we can optimize later)."""
    # For a regular hexagon of side length S centered at origin, the apothem is S * sqrt(3)/2
    # The maximum distance from center to any vertex is S
    # A point (x,y) is inside the outer hexagon if for all 6 directions, the projection is within bounds
    # Actually, let's compute the required S such that all vertices of all inner hexagons are inside.
    
    # For a regular hexagon with side S, center at origin, rotation 0:
    # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees, distance S from center.
    # The hexagon can be described by: for each of 6 directions (normals to edges), 
    # the projection of any point must be <= S * sqrt(3)/2 (apothem)
    
    # Actually, the support function of a regular hexagon with side S in direction theta is:
    # h(theta) = S * sqrt(3)/2 * max over k of cos(theta - k*pi/3) ... no that's not right.
    
    # Let me think differently. A regular hexagon with side S has inradius (apothem) r = S*sqrt(3)/2.
    # It contains a point p if and only if for all 6 edge normals n_k, dot(p, n_k) <= r.
    # The edge normals are at angles 0, 60, 120, 180, 240, 300 degrees (for a hexagon with flat top).
    
    # For our purposes, we need the minimum S such that all vertices of all inner hexagons satisfy:
    # max over k of dot(p, n_k) <= S * sqrt(3)/2
    # So S >= (2/sqrt(3)) * max over all vertices p and all k of dot(p, n_k)
    
    all_vertices = []
    for (cx, cy), ang in zip(inner_positions, inner_angles):
        verts = hex_vertices(cx, cy, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Edge normals for outer hexagon (flat-top orientation)
    normals = []
    for k in range(6):
        ang = np.deg2rad(60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)
    
    # For each vertex, compute max projection onto any normal
    # projections: (num_vertices, 6)
    projections = all_vertices @ normals.T
    max_proj = projections.max()
    
    # S * sqrt(3)/2 >= max_proj => S >= 2*max_proj/sqrt(3)
    S = 2.0 * max_proj / np.sqrt(3.0)
    return max(S, 0.0)

def objective(params):
    """params: array of shape (N*3,) = [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]"""
    positions = [(params[3*i], params[3*i+1]) for i in range(N)]
    angles = [params[3*i+2] for i in range(N)]
    
    # Compute vertices for all hexagons
    all_verts = []
    for (cx, cy), ang in zip(positions, angles):
        all_verts.append(hex_vertices(cx, cy, ang))
    
    # Check overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if poly_overlap(all_verts[i], all_verts[j]):
                penalty += 100.0
    
    # Compute outer side length
    S = outer_hex_side_length(positions, angles)
    
    return S + penalty

def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal lattice arrangement
    # 11 hexagons in a compact cluster: center + 6 around + 4 more
    # Standard hexagonal packing: center at (0,0), 6 at distance 2 in 60-degree increments, 
    # then 4 more in the next ring
    # Actually, for 11: use a 3-4-3 or similar arrangement
    
    # Let's use a simple initial arrangement:
    # Row 1: 3 hexagons, Row 2: 4 hexagons, Row 3: 3 hexagons (offset)
    # For unit hexagons, adjacent centers are 2 apart (since hexagon width is 2)
    # Actually, for regular hexagons with side 1, the distance between centers of adjacent hexagons
    # in a hexagonal lattice is 2 (since the inradius is sqrt(3)/2, and they touch when center distance = 2*inradius = sqrt(3)... no)
    
    # Wait, for regular hexagons of side 1:
    # The distance from center to a vertex is 1 (circumradius = side length = 1)
    # The distance from center to an edge midpoint is sqrt(3)/2 (inradius)
    # Two hexagons touch when the distance between centers = 2 * inradius = sqrt(3) if they share an edge
    # Or 2 * circumradius = 2 if they share a vertex... no.
    
    # Actually, for two regular hexagons of side 1 to be non-overlapping and touching:
    # If they share an edge, center distance = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    # The minimum center distance for non-overlap is sqrt(3) when aligned properly.
    
    # Let me use a simpler initial arrangement and let the optimizer fix it.
    # Start with a linear arrangement and let Nelder-Mead optimize.
    
    # Better initial guess: hexagonal cluster
    # Center hexagon at origin
    # 6 hexagons at distance sqrt(3) in 60-degree increments (edge-to-edge touching)
    # 4 more hexagons in the next ring
    
    d = np.sqrt(3.0)  # distance between adjacent centers in hex lattice
    
    init_positions = [(0.0, 0.0)]
    for k in range(6):
        ang = np.deg2rad(60 * k)
        init_positions.append((d * np.cos(ang), d * np.sin(ang)))
    # 4 more: at distance 2*d in some directions
    for k in [0, 2, 4, 5]:
        ang = np.deg2rad(60 * k)
        init_positions.append((2*d * np.cos(ang), 2*d * np.sin(ang)))
    
    # Ensure we have exactly 11
    init_positions = init_positions[:N]
    init_angles = [0.0] * N
    
    # Build initial params
    params = np.zeros(N * 3)
    for i in range(N):
        params[3*i] = init_positions[i][0]
        params[3*i+1] = init_positions[i][1]
        params[3*i+2] = init_angles[i]
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    best_params = params.copy()
    best_val = objective(params)
    
    # Try multiple restarts with different initial arrangements
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        if restart == 0:
            current_params = params.copy()
        else:
            # Perturb the initial arrangement
            np.random.seed(42 + restart)
            current_params = params + np.random.randn(N * 3) * 0.5
            # Keep angles small
            for i in range(N):
                current_params[3*i+2] = np.random.uniform(-30, 30)
        
        try:
            result = minimize(objective, current_params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract final positions and angles
    positions = [(best_params[3*i], best_params[3*i+1]) for i in range(N)]
    angles = [best_params[3*i+2] for i in range(N)]
    
    # Compute the outer hexagon side length
    S = outer_hex_side_length(positions, angles)
    
    # Build inner list
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    
    return inner, (0.0, 0.0), S, 0.0

# Test
if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Outer side length: {side:.4f}")
    print(f"Center: {center}")
    print(f"Angle: {angle}")
    for i, (x, y, a) in enumerate(inner):
        print(f"Hex {i}: ({x:.3f}, {y:.3f}), angle={a:.1f}")
