import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Precompute unit hexagon vertices (side length 1, center at origin, flat-top? No, pointy-top? 
# Standard: vertices at angles 0, 60, 120, 180, 240, 300.
# This makes the hexagon "pointy" on left/right? 
# Vertex at 0 is (1,0). Vertex at 60 is (0.5, sqrt(3)/2).
# This is a standard regular hexagon.
# The "width" (x-extent) is 2. The "height" (y-extent) is sqrt(3).
# The apothem (distance to side) is sqrt(3)/2.
unit_hex_verts = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2.0],
    [-0.5, np.sqrt(3)/2.0],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2.0],
    [0.5, -np.sqrt(3)/2.0]
])

# Normals for the outer hexagon constraint
# Outer hexagon side S, center 0,0.
# Normals to sides are at angles 30, 90, 150, 210, 270, 330.
outer_normals = np.array([
    [np.cos(np.pi/6), np.sin(np.pi/6)],
    [np.cos(np.pi/2), np.sin(np.pi/2)],
    [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],
    [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],
    [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
    [np.cos(11*np.pi/6), np.sin(11*np.pi/6)]
])

def get_hex_vertices(x, y, theta):
    """Returns 6x2 array of vertices for a unit hexagon at (x,y) rotated by theta."""
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return unit_hex_verts @ R.T + np.array([x, y])

def calc_required_side(vertices_all):
    """
    vertices_all: (N, 6, 2) array of all vertices of all hexagons.
    Returns the minimum side length S of an outer hexagon (center 0,0) containing all vertices.
    """
    # For each vertex, calculate the maximum projection onto outer_normals.
    # proj = vertices_all @ outer_normals.T -> shape (N, 6, 6)
    # We need the max over all vertices and all normals?
    # No, S must satisfy u_k . v <= a for ALL k and ALL v.
    # So a >= max_k (u_k . v).
    # Then S = 2/sqrt(3) * a.
    
    # Efficiently:
    # Flatten vertices to (N*6, 2)
    verts_flat = vertices_all.reshape(-1, 2)
    # Project onto normals: (N*6, 6)
    projs = verts_flat @ outer_normals.T
    # Max projection for each normal: (6,)
    max_projs = np.max(projs, axis=0)
    # The required apothem is the max of these max projections.
    a = np.max(max_projs)
    if a < 0:
        a = 0
    return 2.0 / np.sqrt(3.0) * a

def calc_overlap_penalty(hex_data):
    """
    hex_data: list of (x, y, theta) for N hexagons.
    Returns a smooth penalty for overlaps.
    """
    N = len(hex_data)
    penalty = 0.0
    # Precompute vertices for all hexagons
    verts = [get_hex_vertices(x, y, t) for x, y, t in hex_data]
    
    # Edge normals for a unit hexagon (relative to its center/rotation)
    # The edges are between consecutive vertices.
    # Normal to edge (v_i, v_{i+1}) is perpendicular to v_{i+1}-v_i.
    # For the unit hexagon defined above, the outward normals to the edges are:
    # Edge 0-1: mid (0.75, 0.433). Normal angle 30 deg?
    # Let's just compute normals from the rotated vertices.
    
    for i in range(N):
        for j in range(i+1, N):
            vi = verts[i]
            vj = verts[j]
            
            # Get normals for hex i
            # Edge vectors
            ei = np.roll(vi, -1, axis=0) - vi
            # Normals (outward). For CCW vertices, right normal is inward? 
            # vi is CCW. ei points along edge.
            # Outward normal is (ei[1], -ei[0]) normalized?
            # Let's check. Vertex (1,0) to (0.5, 0.866). ei = (-0.5, 0.866).
            # Right normal: (0.866, 0.5). This points out. Correct.
            ni = np.column_stack((ei[:, 1], -ei[:, 0]))
            ni = ni / np.linalg.norm(ni, axis=1, keepdims=True)
            
            # Get normals for hex j
            ej = np.roll(vj, -1, axis=0) - vj
            nj = np.column_stack((ej[:, 1], -ej[:, 0]))
            nj = nj / np.linalg.norm(nj, axis=1, keepdims=True)
            
            # Check separation along all normals from i and j
            min_sep = np.inf
            for n in ni:
                # Project all vertices of i and j onto n
                pi = vi @ n
                pj = vj @ n
                # Intervals [min(pi), max(pi)] and [min(pj), max(pj)]
                # Overlap if max(mins) < min(maxs)
                # Separation distance = min(maxs) - max(mins)
                # If positive, they are separated by this amount.
                # If negative, they overlap by -separation.
                sep = min(np.max(pi), np.max(pj)) - max(np.min(pi), np.min(pj))
                if sep < min_sep:
                    min_sep = sep
            for n in nj:
                pi = vi @ n
                pj = vj @ n
                sep = min(np.max(pi), np.max(pj)) - max(np.min(pi), np.min(pj))
                if sep < min_sep:
                    min_sep = sep
            
            # min_sep is the distance between the two hexagons.
            # If min_sep < 0, they overlap.
            # We want to penalize overlap.
            # Use a smooth penalty like max(0, -min_sep)^2
            if min_sep < 0:
                penalty += (-min_sep)**2
                
    return penalty

def objective(params):
    """
    params: array of length 3*N. [x1, y1, t1, x2, y2, t2, ...]
    """
    N = len(params) // 3
    hex_data = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        t = params[3*i+2]
        hex_data.append((x, y, t))
    
    # Calculate required side length
    verts = np.array([get_hex_vertices(x, y, t) for x, y, t in hex_data])
    S = calc_required_side(verts)
    
    # Calculate overlap penalty
    pen = calc_overlap_penalty(hex_data)
    
    # Weight the penalty heavily to ensure disjointness
    # But we also want to minimize S.
    # A common strategy is to minimize S + lambda * pen.
    # If pen is 0, we just minimize S.
    # If pen > 0, we push them apart.
    return S + 10.0 * pen

def generate_initial_config():
    """
    Generate a compact initial configuration on a triangular lattice.
    """
    # Lattice vectors for side-to-side touching hexagons
    v1 = np.array([np.sqrt(3), 0.0])
    v2 = np.array([np.sqrt(3)/2.0, 1.5])
    
    points = []
    # Center
    points.append((0.0, 0.0, 0.0))
    
    # First ring (6 neighbors)
    for k in range(6):
        # Indices in lattice basis (i, j) such that pos = i*v1 + j*v2
        # Neighbors are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Let's just use angles for simplicity in initial guess, or map lattice coords.
        pass
    
    # Easier: Just define the 7 central points manually
    # (0,0)
    # (sqrt(3), 0)
    # (sqrt(3)/2, 1.5)
    # (-sqrt(3)/2, 1.5)
    # (-sqrt(3), 0)
    # (-sqrt(3)/2, -1.5)
    # (sqrt(3)/2, -1.5)
    
    pts = [
        (0.0, 0.0),
        (np.sqrt(3), 0.0),
        (np.sqrt(3)/2.0, 1.5),
        (-np.sqrt(3)/2.0, 1.5),
        (-np.sqrt(3), 0.0),
        (-np.sqrt(3)/2.0, -1.5),
        (np.sqrt(3)/2.0, -1.5)
    ]
    
    # We need 4 more. Add them in the second ring.
    # (2*sqrt(3), 0)
    # (3*sqrt(3)/2, 1.5)
    # (sqrt(3)/2, 3.0)
    # (-sqrt(3)/2, 3.0)
    # ...
    # Let's pick 4 that make a compact shape.
    # (2*sqrt(3), 0)
    # (3*sqrt(3)/2, 1.5)
    # (-3*sqrt(3)/2, 1.5)
    # (0, 3.0) ? No, (0, 3) is not on the lattice.
    # Lattice point (0,2) in basis? 
    # 2*v2 = (sqrt(3), 3).
    # 1*v1 + 2*v2 = (2.5*sqrt(3), 3).
    
    # Let's just add 4 points from the next shell arbitrarily close to the center of mass.
    extra_pts = [
        (2.0*np.sqrt(3), 0.0),
        (1.5*np.sqrt(3), 1.5),
        (-1.5*np.sqrt(3), 1.5),
        (0.0, 3.0) # Approximate
    ]
    
    all_pts = pts + extra_pts
    
    # Randomize slightly to break symmetry and help optimizer
    np.random.seed(42)
    for i in range(len(all_pts)):
        all_pts[i] = (all_pts[i][0] + np.random.uniform(-0.1, 0.1), 
                      all_pts[i][1] + np.random.uniform(-0.1, 0.1), 
                      np.random.uniform(0, np.pi))
    
    params = []
    for x, y, t in all_pts:
        params.extend([x, y, t])
    return np.array(params)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    best_params = None
    best_score = np.inf
    
    # Run multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        x0 = generate_initial_config()
        # Add some random perturbation based on restart count
        x0 += np.random.uniform(-0.5, 0.5, size=x0.shape)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
                    
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback to seed-like result if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract best result
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        t = best_params[3*i+2]
        # Convert radians to degrees
        t_deg = np.degrees(t) % 360.0
        inner.append([float(x), float(y), float(t_deg)])
    
    # Calculate final side length
    verts = np.array([get_hex_vertices(x, y, np.radians(t)) for x, y, t in 
                      [(inner[i][0], inner[i][1], inner[i][2]) for i in range(N)]])
    # Wait, get_hex_vertices expects radians.
    # Let's recompute S from best_params directly to avoid conversion errors
    hex_data = []
    for i in range(N):
        hex_data.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
    verts = np.array([get_hex_vertices(x, y, t) for x, y, t in hex_data])
    S = calc_required_side(verts)
    
    # Check for overlaps in the final result. If there are overlaps, we might need to adjust.
    # But the objective function penalizes overlaps, so it should be minimal.
    # If there is still significant overlap, the score might be invalid?
    # The task says "disjoint". If my result has overlaps, it's invalid.
    # I should verify the final result.
    
    # Verification
    verts_list = [get_hex_vertices(x, y, t) for x, y, t in hex_data]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            vi = verts_list[i]
            vj = verts_list[j]
            ei = np.roll(vi, -1, axis=0) - vi
            ni = np.column_stack((ei[:, 1], -ei[:, 0]))
            ni = ni / np.linalg.norm(ni, axis=1, keepdims=True)
            ej = np.roll(vj, -1, axis=0) - vj
            nj = np.column_stack((ej[:, 1], -ej[:, 0]))
            nj = nj / np.linalg.norm(nj, axis=1, keepdims=True)
            
            min_sep = np.inf
            for n in ni:
                pi = vi @ n
                pj = vj @ n
                sep = min(np.max(pi), np.max(pj)) - max(np.min(pi), np.min(pj))
                if sep < min_sep:
                    min_sep = sep
            for n in nj:
                pi = vi @ n
                pj = vj @ n
                sep = min(np.max(pi), np.max(pj)) - max(np.min(pi), np.min(pj))
                if sep < min_sep:
                    min_sep = sep
            
            if min_sep < -1e-6: # Tolerance
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, try to return a safer, larger configuration
        # Or just return the seed result
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return inner, (0.0, 0.0), float(S), 0.0

