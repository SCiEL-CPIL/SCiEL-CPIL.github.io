import numpy as np
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3.0)

def _hex_vertices(cx, cy, angle_deg=0.0):
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _compute_S(centers, angles, outer_angle_deg=0.0):
    """Compute minimum outer hexagon side length."""
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k*60)), 
                         np.sin(theta_rad + np.deg2rad(30 + k*60))] for k in range(6)])
    
    max_proj = 0.0
    for i, (cx, cy) in enumerate(centers):
        verts = _hex_vertices(cx, cy, angles[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / SQRT3

def _smooth_max(values, beta=100.0):
    """Soft max approximation."""
    max_val = np.max(values)
    return max_val + np.log(np.sum(np.exp((values - max_val) / beta)) * np.exp(-max_val / beta)) * beta

def construct():
    start_time = time.time()
    deadline = 140.0
    
    def time_check():
        return time.time() > deadline - 5
    
    # Generate initial configurations
    def make_lattice_config():
        """Create a hexagonal lattice configuration."""
        A = np.array([SQRT3, 0.0])
        B = np.array([SQRT3/2, 1.5])
        
        # Flower pattern: center + 6 neighbors
        base = [
            (0.0, 0.0),
            (SQRT3, 0.0),
            (SQRT3/2, 1.5),
            (-SQRT3/2, 1.5),
            (-SQRT3, 0.0),
            (-SQRT3/2, -1.5),
            (SQRT3/2, -1.5)
        ]
        
        # Ring 2 candidates
        ring2 = [
            (2*SQRT3, 0.0),
            (SQRT3, 3.0),
            (-SQRT3, 3.0),
            (-2*SQRT3, 0.0),
            (-SQRT3, -3.0),
            (SQRT3, -3.0)
        ]
        
        configs = []
        for combo in combinations(range(6), 4):
            extra = [ring2[i] for i in combo]
            centers = base + extra
            configs.append(centers)
        
        return configs
    
    def make_alternative_config():
        """Alternative compact configurations."""
        configs = []
        
        # Try a 3-4-4 arrangement
        # Row 1: 3 hexagons
        # Row 2: 4 hexagons  
        # Row 3: 4 hexagons
        dy = 1.5
        dx = SQRT3/2
        
        for offset in [0, 1]:
            centers = []
            # Row 1 (3 hexagons)
            for i in range(3):
                centers.append((i*dx - dx, 0.0))
            # Row 2 (4 hexagons)
            for i in range(4):
                centers.append((i*dx - 1.5*dx + (offset * dx/2), dy))
            # Row 3 (4 hexagons)
            for i in range(4):
                centers.append((i*dx - 1.5*dx + (offset * dx/2), 2*dy))
            configs.append(centers)
        
        # Try 4-3-4 arrangement
        for offset in [0, 1]:
            centers = []
            # Row 1 (4 hexagons)
            for i in range(4):
                centers.append((i*dx - 1.5*dx + (offset * dx/2), 0.0))
            # Row 2 (3 hexagons)
            for i in range(3):
                centers.append((i*dx - dx, dy))
            # Row 3 (4 hexagons)
            for i in range(4):
                centers.append((i*dx - 1.5*dx + (offset * dx/2), 2*dy))
            configs.append(centers)
        
        return configs
    
    # Generate all initial configurations
    initial_configs = make_lattice_config() + make_alternative_config()
    
    best_S = float('inf')
    best_centers = None
    best_angles = None
    best_outer_angle = 0.0
    
    # For each initial config, try to optimize
    for config_idx, centers in enumerate(initial_configs):
        if time_check():
            break
        
        centers_arr = np.array(centers)
        angles_arr = np.zeros(N)
        
        # Try different outer rotations
        for outer_rot in range(0, 360, 15):
            if time_check():
                break
            
            # Center the configuration
            centroid = np.mean(centers_arr, axis=0)
            centered = centers_arr - centroid
            
            # Compute initial S
            S = _compute_S(list(centered), list(angles_arr), outer_rot)
            
            if S < best_S:
                best_S = S
                best_centers = centered.copy()
                best_angles = angles_arr.copy()
                best_outer_angle = outer_rot
        
        # Use scipy optimization for the best rotation
        try:
            from scipy.optimize import minimize
            
            # Optimize positions and angles
            def objective(params):
                # params: 22 values (11 x, 11 y) + 11 angles = 33 values
                # Actually let's just optimize positions first
                pos = params[:22].reshape(11, 2)
                angs = params[22:33]
                
                # Check overlaps - minimum distance between centers
                # For unit hexagons, min center distance is SQRT3
                min_dist = SQRT3 - 1e-8
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d < min_dist:
                            return 1e10 + (min_dist - d) * 1000
                
                S = _compute_S(list(pos), list(angs), best_outer_angle)
                return S
            
            # Initial guess
            x0 = np.concatenate([best_centers.flatten(), best_angles])
            
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if result.fun < best_S:
                new_centers = result.x[:22].reshape(11, 2)
                new_angles = result.x[22:33]
                # Verify no overlaps
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if np.linalg.norm(new_centers[i] - new_centers[j]) < SQRT3 - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_S = result.fun
                    best_centers = new_centers
                    best_angles = new_angles
                    
        except Exception as e:
            pass
    
    # Final polish: try to optimize outer angle for best config
    if best_centers is not None:
        for outer_rot in np.linspace(0, 360, 361):
            if time_check():
                break
            S = _compute_S(list(best_centers), list(best_angles), outer_rot)
            if S < best_S:
                best_S = S
                best_outer_angle = outer_rot
        
        # One more optimization pass with updated outer angle
        try:
            from scipy.optimize import minimize
            
            def objective2(params):
                pos = params[:22].reshape(11, 2)
                angs = params[22:33]
                
                min_dist = SQRT3 - 1e-8
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d < min_dist:
                            return 1e10 + (min_dist - d) * 1000
                
                S = _compute_S(list(pos), list(angs), best_outer_angle)
                return S
            
            x0 = np.concatenate([best_centers.flatten(), best_angles])
            result = minimize(objective2, x0, method='Nelder-Mead', 
                            options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            
            if result.fun < best_S:
                new_centers = result.x[:22].reshape(11, 2)
                new_angles = result.x[22:33]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if np.linalg.norm(new_centers[i] - new_centers[j]) < SQRT3 - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_S = result.fun
                    best_centers = new_centers
                    best_angles = new_angles
                    
        except:
            pass
    
    # Build output
    if best_centers is None:
        # Fallback: linear arrangement
        inner = [[k * SQRT3, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    inner = [[best_centers[i][0], best_centers[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_S, best_outer_angle
