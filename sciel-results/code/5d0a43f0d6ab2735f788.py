import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    # 1. Find a good cluster of 11 hexagons
    # We will use a simple local search to pack them tightly.
    # Initial guess: a rough hexagonal arrangement
    # Hex grid spacing is sqrt(3) horizontally, 1.5 vertically.
    
    # Let's try to build a cluster manually first as a seed for optimization
    # A 3x4 block is 12 hexagons. Remove one corner.
    # Coordinates for a hex grid (pointy top):
    # x = col * sqrt(3) + (row % 2) * sqrt(3)/2
    # y = row * 1.5
    
    positions = []
    # Generate a 4x4 grid of candidates and pick 11 that are close
    # Actually, let's just use a simple random restart local search on positions.
    
    best_side = float('inf')
    best_positions = None
    best_center = (0,0)
    best_angle = 0
    
    # Define collision check
    # Hexagons are regular, side 1.
    # If orientations are the same, min distance is sqrt(3).
    # We fix orientation to 0 for simplicity in the search phase.
    
    def check_collision(pos):
        # pos is array of (N, 2)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < np.sqrt(3) - 1e-6:
                    return False
        return True

    def get_bounding_hexagon_side(pos):
        # Get all vertices of all hexagons
        # Hexagon vertices relative to center (angle 0, flat top? or pointy top?)
        # Let's use pointy top: vertices at angles 30, 90, 150, 210, 270, 330?
        # Standard math: vertices at 0, 60, 120...
        # Let's stick to 0, 60, 120, 180, 240, 300.
        # Side length 1. Radius 1.
        angles = np.deg2rad(np.arange(0, 360, 60))
        verts_rel = np.column_stack([np.cos(angles), np.sin(angles)])
        
        all_verts = []
        for p in pos:
            all_verts.append(p + verts_rel)
        all_verts = np.vstack(all_verts)
        
        # Compute convex hull
        try:
            hull = ConvexHull(all_verts)
            hull_pts = all_verts[hull.vertices]
        except Exception:
            # If hull fails (e.g. all points same), return large value
            return 1000.0, (0,0), 0.0
            
        # Optimize center and angle to minimize max distance to sides
        # Objective: Max over hull points p, max over k of (n_k . (p-c))
        # We want to minimize this max value, which is S * sqrt(3)/2.
        
        def objective(params):
            cx, cy, theta = params
            c = np.array([cx, cy])
            # Normals for the 6 sides
            # Side normals are at theta + k*60
            # The distance from center to side is R_in = S * sqrt(3)/2.
            # We want to minimize S.
            # S = (1/sqrt(3)/2) * max_p max_k (n_k . (p-c))
            # Note: We must ensure p is on the correct side of the normal.
            # For a convex polygon containing the origin (in local frame), 
            # the max projection onto normals defines the support function.
            
            max_proj = 0.0
            for p in hull_pts:
                v = p - c
                for k in range(3): # 3 pairs of opposite sides
                    angle = theta + k * np.pi / 3
                    n = np.array([np.cos(angle), np.sin(angle)])
                    proj = np.dot(v, n)
                    # The hexagon is defined by |proj| <= R_in
                    # So we need max(|proj|)
                    max_proj = max(max_proj, abs(proj))
            return max_proj / (np.sqrt(3) / 2.0)
            
        # Initial guess for center: centroid of hull
        cx0 = np.mean(hull_pts[:, 0])
        cy0 = np.mean(hull_pts[:, 1])
        theta0 = 0.0
        
        # Optimize
        try:
            res = minimize(objective, [cx0, cy0, theta0], method='Nelder-Mead', options={'maxiter': 200})
            cx, cy, theta = res.x
            side = res.fun
            return side, (cx, cy), theta
        except Exception:
            return 1000.0, (cx0, cy0), theta0

    # Local Search for positions
    # Start with a random tight cluster
    np.random.seed(42)
    for restart in range(50):
        if time.time() > deadline: break
        
        # Initialize positions in a small box
        pos = np.random.uniform(-2, 2, (N, 2))
        
        # Simple local search: move one point at a time
        current_side = 1000.0
        # First, just try to make them non-overlapping and compact
        # We'll use a "repel and compress" heuristic
        # Or just random moves accepting if non-overlapping and bounding box shrinks
        
        for step in range(500):
            if time.time() > deadline: break
            
            # Pick a random index
            i = np.random.randint(N)
            # Move it randomly
            dx = np.random.uniform(-0.5, 0.5)
            dy = np.random.uniform(-0.5, 0.5)
            old_pos = pos[i].copy()
            pos[i] += np.array([dx, dy])
            
            if not check_collision(pos):
                pos[i] = old_pos
                continue
            
            # Calculate bounding box side
            side, c, a = get_bounding_hexagon_side(pos)
            
            if side < current_side:
                current_side = side
                best_side = min(best_side, side)
                best_positions = pos.copy()
                best_center = c
                best_angle = a
            else:
                # Revert if not better? Or accept with prob?
                # For speed and simplicity, let's just accept if better or with small prob
                # Actually, strict improvement is safer for avoiding local minima traps in simple search
                # But we might get stuck. Let's add a small random jitter if stuck.
                # For now, revert if worse.
                pos[i] = old_pos

    if best_positions is None:
        # Fallback to linear packing
        pos = np.array([[i * np.sqrt(3), 0] for i in range(N)])
        side, c, a = get_bounding_hexagon_side(pos)
        best_positions = pos
        best_center = c
        best_angle = a
        best_side = side

    # Format output
    inner = []
    for p in best_positions:
        inner.append([p[0], p[1], 0.0])
        
    return inner, best_center, best_side, best_angle
