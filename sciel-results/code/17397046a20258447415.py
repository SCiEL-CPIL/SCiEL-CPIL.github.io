import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    n_hex = 11
    
    # Precompute base vertices of unit hexagon
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.deg2rad(np.arange(6) * 60)
    base_vertices = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # Precompute rotation matrices for base vertices? No, we will compute on the fly for each hexagon.
    # Or we can precompute the 6 vertex offsets for a given angle.

    def get_hex_vertices(center, angle):
        """
        Returns (6, 2) array of vertices for a unit hexagon
        with given center and rotation angle.
        """
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[c, -s], [s, c]])
        # Rotate base vertices
        verts = base_vertices @ rot_mat.T
        # Translate
        verts = verts + center
        return verts

    def compute_outer_side(params):
        """
        params: [x1, y1, theta1, ..., x11, y11, theta11, Theta_outer]
        Returns the minimum side length of the enclosing regular hexagon.
        """
        # Extract outer angle
        theta_out = params[-1]
        
        # Outer hexagon normals
        # Normals are at 30, 90, 150, 210, 270, 330 degrees + theta_out
        outer_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        n_cos = np.cos(outer_angles)
        n_sin = np.sin(outer_angles)
        
        max_proj = 0.0
        
        # Iterate over inner hexagons
        for i in range(n_hex):
            cx = params[i * 3]
            cy = params[i * 3 + 1]
            theta_i = params[i * 3 + 2]
            
            # Compute vertices
            c, s = np.cos(theta_i), np.sin(theta_i)
            # Rotated base vertices
            vx = base_vertices[:, 0] * c - base_vertices[:, 1] * s
            vy = base_vertices[:, 0] * s + base_vertices[:, 1] * c
            # Translate
            vx += cx
            vy += cy
            
            # Project onto normals
            # dots[j, k] = vx[j] * n_cos[k] + vy[j] * n_sin[k]
            dots = vx[:, None] * n_cos[None, :] + vy[:, None] * n_sin[None, :]
            
            # Max absolute projection for this hexagon
            local_max = np.max(np.abs(dots))
            if local_max > max_proj:
                max_proj = local_max
        
        # Side length S is related to max projection P by P = S * sqrt(3) / 2
        # So S = 2 * P / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)

    # Initial guesses
    # We will use lattice-based initial guesses and then optimize.
    
    # Guess 1: Flower packing
    # 1 center, 6 at dist sqrt(3), 4 at dist 3
    eps = 1e-12
    centers1 = np.array([[0.0, 0.0]])
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers1 = np.vstack([centers1, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers1 = np.vstack([centers1, [r2 * np.cos(angle), r2 * np.sin(angle)]])
    
    # Guess 2: Diamond packing (3-5-3)
    # Row 0: 3 hexagons
    # Row 1: 5 hexagons
    # Row 2: 3 hexagons
    # In hex grid, row offset is 0.5 * width
    # Let's place them manually
    centers2 = []
    # Row 0 (y=0)
    for x in [-1, 0, 1]:
        centers2.append([x * np.sqrt(3), 0])
    # Row 1 (y=sqrt(3))
    for x in [-2, -1, 0, 1, 2]:
        centers2.append([x * np.sqrt(3), np.sqrt(3)])
    # Row 2 (y=2*sqrt(3))
    for x in [-1, 0, 1]:
        centers2.append([x * np.sqrt(3), 2 * np.sqrt(3)])
    centers2 = np.array(centers2)
    # Center the shape
    centers2 -= centers2.mean(axis=0)

    # Guess 3: Bar packing (1-1-1-1-1-1-1-1-1-1-1) - likely bad but good for diversity
    centers3 = np.array([[i * np.sqrt(3), 0] for i in range(11)])
    centers3 -= centers3.mean(axis=0)

    # Guess 4: Random
    np.random.seed(42)
    centers4 = np.random.uniform(-2, 2, (11, 2))

    initial_guesses = [centers1, centers2, centers3, centers4]
    
    best_score = np.inf
    best_params = None
    
    for centers_init in initial_guesses:
        if time.time() > deadline - 10:
            break
            
        # Initialize params: [x1, y1, theta1, ..., x11, y11, theta11, Theta_outer]
        # All inner hexagons aligned with outer (theta=0) or random?
        # Let's try theta=0 and theta=30 deg for inner hexagons
        for init_theta_mode in [0, 1]:
            if time.time() > deadline - 10:
                break
                
            params_init = []
            for i in range(n_hex):
                params_init.append(centers_init[i, 0])
                params_init.append(centers_init[i, 1])
                if init_theta_mode == 0:
                    params_init.append(0.0)
                else:
                    params_init.append(np.deg2rad(30))
            # Outer angle
            params_init.append(np.deg2rad(30))
            
            x0 = np.array(params_init)
            
            try:
                # Use Powell method
                res = minimize(compute_outer_side, x0, method='Powell',
                               options={'maxiter': 200, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12})
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
                    
            except Exception:
                pass

    # If no optimization succeeded, use the best initial guess
    if best_params is None:
        # Fallback to flower packing with no optimization
        params_init = []
        for i in range(n_hex):
            params_init.append(initial_guesses[0][i, 0])
            params_init.append(initial_guesses[0][i, 1])
            params_init.append(0.0)
        params_init.append(np.deg2rad(30))
        best_params = np.array(params_init)
        best_score = compute_outer_side(best_params)

    # Extract solution
    inner = []
    for i in range(n_hex):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        angle_deg = np.rad2deg(best_params[i * 3 + 2])
        inner.append([x, y, angle_deg])
        
    side = best_score + 1e-9
    outer_angle_deg = np.rad2deg(best_params[-1])
    
    return inner, (0.0, 0.0), side, outer_angle_deg

def _fallback_solution():
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    side = 4.0 + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
