import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    n_hex = 11
    
    # Base vertices of unit hexagon (flat-top, angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def get_vertices(center, angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return base_vertices @ rot.T + center

    def check_overlap(poly1, poly2):
        """
        Returns True if two convex polygons overlap.
        Uses Separating Axis Theorem (SAT).
        """
        # Get axes from both polygons
        axes = []
        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            edge = p2 - p1
            # Normal
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-10:
                axes.append(normal / np.linalg.norm(normal))
        
        for i in range(6):
            p1 = poly2[i]
            p2 = poly2[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-10:
                axes.append(normal / np.linalg.norm(normal))

        for axis in axes:
            proj1 = np.dot(poly1, axis)
            proj2 = np.dot(poly2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return False
        
        return True

    def compute_outer_side(centers, angles, theta_out):
        """
        Compute minimum outer hexagon side length.
        """
        # Outer hexagon face normals: at angles 30 + k*60 + theta_out
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])

        max_proj = 0.0
        for i in range(n_hex):
            verts = get_vertices(centers[i], angles[i])
            dots = verts @ normals.T
            max_proj = max(max_proj, np.max(np.abs(dots)))

        # S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)

    def objective(params):
        """
        params: [x0, y0, theta0, ..., x10, y10, theta10, theta_out]
        """
        centers = params[0:3*n_hex].reshape(n_hex, 2)
        angles = params[3*n_hex:3*n_hex+n_hex]
        theta_out = params[-1]
        
        # 1. Calculate bounding side
        side = compute_outer_side(centers, angles, theta_out)
        
        # 2. Check overlaps
        penalty = 0.0
        # Pre-compute vertices for all hexagons to avoid recomputation in pair checks
        all_verts = []
        for i in range(n_hex):
            all_verts.append(get_vertices(centers[i], angles[i]))
            
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                # Quick rejection: center distance
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < 1.732: # sqrt(3)
                    # Definitely overlapping or very close, check precisely
                    if check_overlap(all_verts[i], all_verts[j]):
                        penalty += 100.0
                elif dist < 2.0:
                    # Might be overlapping if rotated
                    if check_overlap(all_verts[i], all_verts[j]):
                        penalty += 100.0
                # If dist >= 2.0, no overlap possible for unit hexagons
        
        return side + penalty

    # Initial Guess: Compact cluster
    # 1 center, 6 around it, 4 more
    # Hex grid coordinates
    # Let's place them in a "rhombus" shape
    # (0,0)
    # (1.5, 0.866), (-1.5, 0.866), (1.5, -0.866), (-1.5, -0.866), (0, 1.732), (0, -1.732)
    # This is 7.
    # Add 4 more:
    # (3, 0), (-3, 0), (0, 3.464), (0, -3.464) ? No, too far.
    # Add to the sides:
    # (1.5, 2.598), (-1.5, 2.598), (1.5, -2.598), (-1.5, -2.598)
    
    centers_init = np.array([
        [0.0, 0.0],
        [1.5, 0.866],
        [-1.5, 0.866],
        [1.5, -0.866],
        [-1.5, -0.866],
        [0.0, 1.732],
        [0.0, -1.732],
        [3.0, 0.0],
        [-3.0, 0.0],
        [1.5, 2.598],
        [-1.5, -2.598]
    ])
    
    angles_init = np.zeros(n_hex)
    theta_out_init = 0.0
    
    x0 = np.concatenate([centers_init.ravel(), angles_init, [theta_out_init]])
    
    best_score = objective(x0)
    best_params = x0.copy()
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 15
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
            
        if i == 0:
            x_start = x0.copy()
        else:
            # Perturb initial guess
            x_start = x0 + np.random.normal(0, 0.5, x0.shape)
            # Ensure theta_out is in [0, pi/3]
            x_start[-1] = np.mod(x_start[-1], np.pi/3)
            # Ensure angles are in [0, pi/3]
            x_start[3*n_hex:3*n_hex+n_hex] = np.mod(x_start[3*n_hex:3*n_hex+n_hex], np.pi/3)
            
        try:
            res = minimize(objective, x_start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_score - 1e-6:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final polish with tighter tolerances
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score - 1e-8:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract solution
    centers = best_params[0:3*n_hex].reshape(n_hex, 2)
    angles = best_params[3*n_hex:3*n_hex+n_hex]
    theta_out = best_params[-1]
    
    # Recalculate true side length without penalty
    side = compute_outer_side(centers, angles, theta_out)
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(angles[i])])
        
    # Add tiny margin
    side += 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
