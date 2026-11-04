"""Packing 11 unit hexagons in a minimal outer regular hexagon."""
import time
import numpy as np
from scipy.optimize import minimize

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

def min_distance(v1, v2):
    """Calculate minimum distance between two polygons. Returns 0 if overlapping."""
    if not poly_overlap(v1, v2):
        # Calculate min distance between any vertex of v1 and any edge of v2, and vice versa
        # For simplicity and speed, we can use a point-to-edge distance approximation
        # Or just return a large positive number if not overlapping
        # A more robust way: check all pairs of vertices and edges.
        # But for the penalty, we just need a smooth function.
        # Let's compute the minimum distance between vertices of one and edges of the other.
        min_dist = float('inf')
        # v1 vertices to v2 edges
        for i in range(len(v1)):
            p = v1[i]
            for j in range(len(v2)):
                p1 = v2[j]
                p2 = v2[(j+1)%len(v2)]
                # Project p onto segment p1-p2
                v = p2 - p1
                w = p - p1
                c1 = np.dot(w, v)
                if c1 <= 0:
                    d = np.linalg.norm(p - p1)
                else:
                    c2 = np.dot(p2 - p, v)
                    if c2 <= 0:
                        d = np.linalg.norm(p - p2)
                    else:
                        b = np.dot(v, v)
                        t = c1 / b
                        proj = p1 + t * v
                        d = np.linalg.norm(p - proj)
                if d < min_dist:
                    min_dist = d
        # v2 vertices to v1 edges
        for i in range(len(v2)):
            p = v2[i]
            for j in range(len(v1)):
                p1 = v1[j]
                p2 = v1[(j+1)%len(v1)]
                v = p2 - p1
                w = p - p1
                c1 = np.dot(w, v)
                if c1 <= 0:
                    d = np.linalg.norm(p - p1)
                else:
                    c2 = np.dot(p2 - p, v)
                    if c2 <= 0:
                        d = np.linalg.norm(p - p2)
                    else:
                        b = np.dot(v, v)
                        t = c1 / b
                        proj = p1 + t * v
                        d = np.linalg.norm(p - proj)
                if d < min_dist:
                    min_dist = d
        return min_dist
    else:
        return 0.0

def outer_hex_side_length(inner_positions, inner_angles):
    """Compute the minimum side length of a regular hexagon that contains all inner hexagons."""
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
    projections = all_vertices @ normals.T
    max_proj = projections.max()
    
    # S * sqrt(3)/2 >= max_proj => S >= 2*max_proj/sqrt(3)
    S = 2.0 * max_proj / np.sqrt(3.0)
    return max(S, 0.0)

def objective(params):
    """params: array of shape (N*2,) = [x1, y1, x2, y2, ..., xN, yN] (angles fixed to 0)"""
    positions = [(params[2*i], params[2*i+1]) for i in range(N)]
    angles = [0.0] * N
    
    # Compute vertices for all hexagons
    all_verts = []
    for (cx, cy), ang in zip(positions, angles):
        all_verts.append(hex_vertices(cx, cy, ang))
    
    # Check overlaps with a smooth penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = min_distance(all_verts[i], all_verts[j])
            if d < 1e-6:
                # Quadratic penalty for overlap
                penalty += 100.0 * (1e-6 - d)**2
            # Even if not overlapping, we can add a small repulsive force to keep them apart
            # to avoid numerical issues, but the outer hex size will push them out anyway.
    
    # Compute outer side length
    S = outer_hex_side_length(positions, angles)
    
    return S + penalty

def construct():
    start_time = time.time()
    
    # Initial guess: Hexagonal cluster
    # 1 center, 6 around, 4 more in second ring
    d = np.sqrt(3.0) # Distance between centers of touching hexagons
    
    init_positions = [(0.0, 0.0)]
    for k in range(6):
        ang = np.deg2rad(60 * k)
        init_positions.append((d * np.cos(ang), d * np.sin(ang)))
    # 4 more in the second ring at 0, 120, 240, 300? 
    # Let's place them at distance 2*d from center, at angles 0, 60, 120, 180... 
    # Actually, to keep it compact, place them in the gaps of the first ring.
    # The first ring occupies angles 0, 60, ..., 300.
    # The gaps are at 30, 90, 150, 210, 270, 330.
    # Let's pick 4 of these gaps.
    gap_angles = [30, 90, 150, 210]
    for ang_deg in gap_angles:
        ang = np.deg2rad(ang_deg)
        # Distance to center for second ring hexagons in gaps?
        # If we place them in the gaps, they touch two first-ring hexagons.
        # The distance from center to such a hexagon is d * sqrt(3) ? No.
        # Let's just use a simple grid-like initial guess and let the optimizer fix it.
        # Or use the standard hexagonal lattice positions.
        # Lattice vectors: v1 = (d, 0), v2 = (d/2, d*sqrt(3)/2)
        # Positions:
        # (0,0)
        # (d, 0), (-d, 0)
        # (d/2, d*sqrt(3)/2), (-d/2, d*sqrt(3)/2)
        # (d/2, -d*sqrt(3)/2), (-d/2, -d*sqrt(3)/2)
        # That's 7.
        # Next ring:
        # (2d, 0), (d, d*sqrt(3)), (0, 2*d*sqrt(3)/2)... this is getting complicated.
        
        # Let's just use a random restart strategy with the 7-hexagon cluster as base.
        pass

    # Better Initial Guess:
    # Use a known good packing for 11 hexagons or just a dense random cloud.
    # Let's use a 3-4-3 grid of hexagons.
    # Row 1: 3 hexagons at y = -d*sqrt(3)/2
    # Row 2: 4 hexagons at y = 0
    # Row 3: 3 hexagons at y = d*sqrt(3)/2
    
    # This forms a parallelogram-like shape.
    # Let's construct it.
    h = d * np.sqrt(3) / 2.0
    w = d
    
    init_positions = []
    # Row 1 (y = -h)
    for i in range(3):
        x = (i - 1) * w
        init_positions.append((x, -h))
    # Row 2 (y = 0)
    for i in range(4):
        x = (i - 1.5) * w
        init_positions.append((x, 0.0))
    # Row 3 (y = h)
    for i in range(3):
        x = (i - 1) * w
        init_positions.append((x, h))
        
    # Ensure we have 11
    if len(init_positions) != N:
        # Adjust if needed
        init_positions = init_positions[:N]
        
    init_params = np.array([coord for pos in init_positions for coord in pos])
    
    best_params = init_params.copy()
    best_val = objective(init_params)
    
    # Optimize
    # Use Nelder-Mead on the 22D position space
    try:
        result = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # If we have time, try a few random restarts
    if time.time() - start_time < DEADLINE - 10:
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            np.random.seed(42 + restart)
            # Perturb initial guess
            current_params = init_params + np.random.randn(N * 2) * 0.5
            try:
                result = minimize(objective, current_params, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x
            except Exception:
                pass

    # Extract final positions
    positions = [(best_params[2*i], best_params[2*i+1]) for i in range(N)]
    angles = [0.0] * N
    
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
