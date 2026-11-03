import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute the 6 unit normals for a regular hexagon with flat-top orientation (angle=0)
    # The sides are at angles 0, 60, 120, 180, 240, 300 degrees relative to the normal?
    # Actually, for a regular hexagon with flat top, the outward normals are at angles:
    # 0, 60, 120, 180, 240, 300 degrees? No.
    # Let's define the hexagon such that its sides are horizontal/vertical?
    # Standard regular hexagon with side length s:
    # Vertices at (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # Normals to the sides:
    # Side 1: between (s,0) and (s/2, s*sqrt(3)/2). Midpoint (3s/4, s*sqrt(3)/4). Normal angle 30 deg.
    # Side 2: between (s/2, s*sqrt(3)/2) and (-s/2, s*sqrt(3)/2). Midpoint (0, s*sqrt(3)/2). Normal angle 90 deg.
    # Side 3: between (-s/2, s*sqrt(3)/2) and (-s, 0). Midpoint (-3s/4, s*sqrt(3)/4). Normal angle 150 deg.
    # Side 4: between (-s, 0) and (-s/2, -s*sqrt(3)/2). Midpoint (-3s/4, -s*sqrt(3)/4). Normal angle 210 deg.
    # Side 5: between (-s/2, -s*sqrt(3)/2) and (s/2, -s*sqrt(3)/2). Midpoint (0, -s*sqrt(3)/2). Normal angle 270 deg.
    # Side 6: between (s/2, -s*sqrt(3)/2) and (s, 0). Midpoint (3s/4, -s*sqrt(3)/4). Normal angle 330 deg.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    
    normals = np.array([
        [math.cos(math.radians(30 + 60*k)), math.sin(math.radians(30 + 60*k))]
        for k in range(6)
    ])
    
    # Objective: Minimize the side length S of the outer hexagon.
    # The outer hexagon is defined by center (cx, cy) and side length S.
    # Its apothem is A = S * sqrt(3)/2.
    # For a point p to be inside, for all normals n: |dot(p - center, n)| <= A.
    # So A >= max_k max_i |dot(p_i - center, n_k)|.
    # S = 2 * A / sqrt(3).
    
    def compute_S(pts, cx, cy):
        # pts: (N, 2)
        # cx, cy: center
        # Returns S
        # Shift points relative to center
        shifted = pts - np.array([cx, cy])
        # Project onto all normals
        # proj: (N, 6)
        proj = shifted @ normals.T
        # Max absolute projection
        max_proj = np.max(np.abs(proj))
        A = max_proj
        S = 2.0 * A / math.sqrt(3.0)
        return S

    def objective(pts):
        # Find optimal center for given pts
        # The optimal center minimizes the max projection.
        # This is equivalent to finding the Chebyshev center of the points in the hexagonal norm.
        # For a fixed set of points, the optimal center is the center of the minimal enclosing hexagon.
        # We can approximate this by taking the midpoint of the bounding box in the projected coordinates?
        # Or simply use the centroid? No, centroid is not optimal.
        # Let's use a simple local search for the center as well, or just use the centroid for speed in the inner loop?
        # Actually, for fixed points, the function f(cx, cy) = max_k max_i |dot(p_i - (cx,cy), n_k)| is convex.
        # We can solve this via linear programming or a simple subgradient method.
        # Given the time constraints, let's just use the centroid for the center in the inner loop of SA?
        # That might be suboptimal but fast.
        # Better: use the center of the bounding box in Cartesian coordinates?
        # Let's try using the average of min and max x and y?
        # Or better, use the center that minimizes the max distance in the hexagonal metric.
        # For now, let's use the centroid. It's a good starting point.
        # We will refine the center in a separate step or include it in the search.
        # Let's include center in the search variables? That doubles the dimension.
        # Alternatively, for each configuration, compute the exact optimal center using a few iterations of subgradient descent?
        # Let's stick to centroid for the main SA loop, and then do a final refinement of the center for the best solution.
        
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        return compute_S(pts, cx, cy)

    # Initial configuration: Grid-like
    # Hexagon side 1. Distance between centers for non-overlapping is 2 * apothem = sqrt(3).
    # Axial coordinates:
    # (q, r) -> x = sqrt(3) * (q + r/2), y = 1.5 * r
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate a few initial configurations
    inits = []
    
    # 1. Compact cluster
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    inits.append(axial_to_cartesian(axial_1))
    
    # 2. Linear chain
    axial_2 = [(i, 0) for i in range(N)]
    inits.append(axial_to_cartesian(axial_2))
    
    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        # Random axial coordinates
        axials = []
        for _ in range(N):
            q = np.random.randint(-3, 4)
            r = np.random.randint(-3, 4)
            axials.append((q, r))
        inits.append(axial_to_cartesian(axials))

    best_S = float('inf')
    best_pts = None
    best_center = None

    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = init.copy()
        current_S = objective(pts)
        
        if current_S < best_S:
            best_S = current_S
            best_pts = pts.copy()
            
        while T > T_min:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            # Random perturbation
            new_pts = pts.copy()
            # Move a random point
            idx = np.random.randint(N)
            # Random direction and magnitude
            angle = np.random.uniform(0, 2 * math.pi)
            dist = np.random.uniform(0, 0.5)
            new_pts[idx, 0] += dist * math.cos(angle)
            new_pts[idx, 1] += dist * math.sin(angle)
            
            new_S = objective(new_pts)
            delta = new_S - current_S
            
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                pts = new_pts
                current_S = new_S
                if current_S < best_S:
                    best_S = current_S
                    best_pts = pts.copy()
            
            T *= cooling
    
    # Refine the center for the best points
    if best_pts is not None:
        # Use Nelder-Mead to optimize the center
        from scipy.optimize import minimize
        
        def obj_center(params):
            cx, cy = params
            return compute_S(best_pts, cx, cy)
        
        cx0 = np.mean(best_pts[:, 0])
        cy0 = np.mean(best_pts[:, 1])
        
        try:
            res = minimize(obj_center, [cx0, cy0], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
            if res.success or res.fun < best_S:
                best_S = res.fun
                best_center = (res.x[0], res.x[1])
        except Exception:
            best_center = (cx0, cy0)
            
        # Final check
        if best_center is None:
            best_center = (cx0, cy0)
        
        # Construct output
        inner = []
        for x, y in best_pts:
            inner.append([float(x), float(y), 0.0])
            
        center = [float(best_center[0]), float(best_center[1])]
        side = float(best_S)
        rotation = 0.0
        
        return inner, center, side, rotation
    else:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

