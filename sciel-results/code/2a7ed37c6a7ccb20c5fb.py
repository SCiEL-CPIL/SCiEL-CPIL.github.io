import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    rng = np.random.RandomState(42)
    
    # Initial configuration: hexagonal lattice arrangement
    # Start with a compact arrangement
    def initial_config():
        # Try several initial patterns and pick the best
        patterns = []
        
        # Pattern 1: 3-4-3 arrangement (roughly)
        # Hex centers in axial coords, convert to cartesian
        def axial_to_cart(q, r):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            return x, y
        
        # 3-4-3 pattern
        p1 = [
            (0,0), (1,0), (2,0),
            (0,1), (1,1), (2,1), (3,1),
            (0,2), (1,2), (2,2)
        ]
        
        # 2-3-3-2 pattern
        p2 = [
            (0,0), (1,0),
            (0,1), (1,1), (2,1),
            (0,2), (1,2), (2,2),
            (0,3), (1,3)
        ]
        
        # 1-3-3-3-1 pattern (more spread)
        p3 = [
            (1,0),
            (0,1), (1,1), (2,1),
            (0,2), (1,2), (2,2),
            (0,3), (1,3), (2,3),
            (1,4)
        ]
        
        # 4-3-3-1 pattern
        p4 = [
            (0,0), (1,0), (2,0), (3,0),
            (0,1), (1,1), (2,1),
            (0,2), (1,2), (2,2),
            (1,3)
        ]
        
        # 2-4-3-2 pattern
        p5 = [
            (0,0), (1,0),
            (0,1), (1,1), (2,1), (3,1),
            (0,2), (1,2), (2,2),
            (0,3), (1,3)
        ]
        
        # 3-3-3-2 pattern
        p6 = [
            (0,0), (1,0), (2,0),
            (0,1), (1,1), (2,1),
            (0,2), (1,2), (2,2),
            (1,3), (2,3)
        ]
        
        for pattern in [p1, p2, p3, p4, p5, p6]:
            pts = np.array([axial_to_cart(q, r) for q, r in pattern])
            patterns.append(pts)
        
        # Evaluate each pattern and return the best
        best_pts = None
        best_score = float('inf')
        for pts in patterns:
            score = evaluate_packing(pts, np.zeros(N), 0.0)
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        
        return best_pts
    
    def evaluate_packing(centers, angles, outer_angle):
        """Calculate the outer hexagon side length needed to contain all inner hexagons."""
        # For each inner hexagon, compute its 6 vertices
        # Then find the maximum projection onto the 6 outer hexagon normal directions
        max_proj = 0.0
        
        # Outer hexagon normals
        outer_normals = []
        for k in range(6):
            ang = outer_angle + math.pi / 6.0 + k * math.pi / 3.0
            outer_normals.append((math.cos(ang), math.sin(ang)))
        
        for i in range(N):
            cx, cy = centers[i]
            theta = angles[i]
            # Vertices of unit hexagon rotated by theta
            for k in range(6):
                v_ang = theta + k * math.pi / 3.0
                vx = cx + math.cos(v_ang)
                vy = cy + math.sin(v_ang)
                # Project onto each outer normal
                for nx, ny in outer_normals:
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # The outer apothem must be at least max_proj
        # Side length = 2 * apothem / sqrt(3) = 2 * max_proj / sqrt(3)
        # Wait, for a regular hexagon, apothem = side * sqrt(3) / 2
        # So side = 2 * apothem / sqrt(3)
        # But we need the apothem to be at least max_proj
        # Actually, the outer hexagon's apothem is the distance from center to each side
        # The constraint is that all inner hexagon vertices must be within the outer hexagon
        # For a regular hexagon centered at origin with side length S, the apothem is S*sqrt(3)/2
        # A point (x,y) is inside if for all 6 normals, |dot(point, normal)| <= apothem
        # So we need apothem >= max_proj
        # Side = 2 * apothem / sqrt(3)
        side = 2.0 * max_proj / math.sqrt(3.0)
        
        return side
    
    def get_vertices(centers, angles):
        """Get all 66 vertices (11 hexagons * 6 vertices)."""
        verts = []
        for i in range(N):
            cx, cy = centers[i]
            theta = angles[i]
            for k in range(6):
                v_ang = theta + k * math.pi / 3.0
                vx = cx + math.cos(v_ang)
                vy = cy + math.sin(v_ang)
                verts.append((vx, vy))
        return np.array(verts)
    
    def compute_required_side(centers, angles, outer_angle):
        """Compute the minimum outer side length."""
        verts = get_vertices(centers, angles)
        
        max_proj = 0.0
        for k in range(6):
            ang = outer_angle + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = verts[:, 0] * nx + verts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs))
        
        side = 2.0 * max_proj / math.sqrt(3.0)
        return side
    
    # Simulated annealing
    best_centers = initial_config()
    best_angles = np.zeros(N)
    best_outer_angle = 0.0
    best_score = compute_required_side(best_centers, best_angles, best_outer_angle)
    
    # Temperature schedule
    T_start = 0.5
    T_end = 0.001
    n_iter = 50000
    
    time_budget = TIME_LIMIT - SAFETY_MARGIN
    start_sa = time.time()
    
    for it in range(n_iter):
        if time.time() - start_sa > time_budget:
            break
        
        # Temperature
        frac = it / n_iter
        T = T_start * (T_end / T_start) ** frac
        
        # Perturbation
        centers_new = best_centers.copy()
        angles_new = best_angles.copy()
        outer_angle_new = best_outer_angle
        
        # Move a random hexagon
        idx = rng.randint(N)
        dx = rng.normal(0, 0.3)
        dy = rng.normal(0, 0.3)
        centers_new[idx, 0] += dx
        centers_new[idx, 1] += dy
        angles_new[idx] += rng.normal(0, 0.2)
        
        # Occasionally perturb outer angle
        if rng.random() < 0.1:
            outer_angle_new = best_outer_angle + rng.normal(0, 0.05)
        
        # Evaluate
        score_new = compute_required_side(centers_new, angles_new, outer_angle_new)
        
        # Acceptance
        if score_new < best_score or rng.random() < math.exp((best_score - score_new) / T):
            best_centers = centers_new
            best_angles = angles_new
            best_outer_angle = outer_angle_new
            best_score = score_new
    
    # Final optimization: refine outer angle for fixed centers/angles
    def refine_outer_angle(centers, angles):
        best_ao = 0.0
        best_s = float('inf')
        for ao in np.linspace(0, math.pi/6, 100):
            s = compute_required_side(centers, angles, ao)
            if s < best_s:
                best_s = s
                best_ao = ao
        # Fine-tune
        for _ in range(20):
            improved = False
            for delta in [0.001, -0.001, 0.0001, -0.0001]:
                ao2 = best_ao + delta
                s2 = compute_required_side(centers, angles, ao2)
                if s2 < best_s:
                    best_s = s2
                    best_ao = ao2
                    improved = True
            if not improved:
                break
        return best_ao, best_s
    
    best_outer_angle, best_score = refine_outer_angle(best_centers, best_angles)
    
    # Final polish: local search on all parameters
    for _ in range(50):
        improved = False
        for i in range(N):
            for dx in [0.01, -0.01, 0.005, -0.005]:
                for dy in [0.01, -0.01, 0.005, -0.005]:
                    c_test = best_centers.copy()
                    c_test[i, 0] += dx
                    c_test[i, 1] += dy
                    s_test = compute_required_side(c_test, best_angles, best_outer_angle)
                    if s_test < best_score - 1e-12:
                        best_centers = c_test
                        best_score = s_test
                        improved = True
        if not improved:
            break
    
    # Re-refine outer angle
    best_outer_angle, best_score = refine_outer_angle(best_centers, best_angles)
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i, 0]), float(best_centers[i, 1]), float(math.degrees(best_angles[i]))])
    
    center = [0.0, 0.0]
    side = float(best_score)
    rotation = float(math.degrees(best_outer_angle))
    
    return inner, center, side, rotation
