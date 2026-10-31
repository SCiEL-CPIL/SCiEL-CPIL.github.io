import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_cartesian(pts):
    """
    Calculate areas of all triangles formed by N points in Cartesian coords.
    pts: (N, 2) array
    Returns: (165,) array of areas
    """
    p_i = pts[tri_idx[:, 0]]
    p_j = pts[tri_idx[:, 1]]
    p_k = pts[tri_idx[:, 2]]
    
    # Vector differences
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product magnitude / 2
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def soft_min_obj(params, beta=50.0):
    """
    Objective function: Negative of soft-min of triangle areas.
    params: (2N,) array of Cartesian coordinates
    """
    pts = params.reshape(N, 2)
    areas = calc_areas_cartesian(pts)
    
    # Avoid log(0)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 1e10
    
    # Soft min approximation
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    
    return -soft_min_val

def hard_min_obj(params):
    """
    Objective function: Negative of hard min of triangle areas.
    """
    pts = params.reshape(N, 2)
    areas = calc_areas_cartesian(pts)
    return -np.min(areas)

def get_min_area(pts):
    """
    Get the minimum area for a given set of points.
    """
    areas = calc_areas_cartesian(pts)
    return np.min(areas)

def constraints(pts_flat):
    """
    Constraints for SLSQP:
    1. y >= 0
    2. y <= sqrt(3) * x
    3. y <= sqrt(3) * (1 - x)
    """
    pts = pts_flat.reshape(N, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    
    sqrt3 = np.sqrt(3)
    
    c1 = y
    c2 = sqrt3 * x - y
    c3 = sqrt3 * (1 - x) - y
    
    return np.concatenate([c1, c2, c3])

def generate_lattice_init():
    """
    Generate a hexagonal lattice-like initialization.
    """
    pts = []
    # Center
    pts.append([0.5, np.sqrt(3)/6])
    
    # Ring 1 (6 points)
    r = 0.2
    for i in range(6):
        angle = i * np.pi / 3
        x = 0.5 + r * np.cos(angle)
        y = np.sqrt(3)/6 + r * np.sin(angle)
        pts.append([x, y])
        
    # Ring 2 (Need 4 more points)
    # Place them at the "corners" of the hexagon, slightly pushed out.
    # Or just pick 4 random valid points far from the center.
    # For now, let's use a specific pattern:
    # The 4 points can be placed at the midpoints of the triangle sides?
    # Midpoints: (0.5, 0), (0.25, 0.433), (0.75, 0.433)
    # These are 3 points. We need 4.
    # Let's add one more near the top vertex?
    # Or just use random for the remaining 4.
    
    rng = np.random.default_rng(42)
    while len(pts) < N:
        u = rng.random()
        v = rng.random() * (1 - u)
        x = v * 1.0 + (1 - u - v) * 0.5
        y = (1 - u - v) * (np.sqrt(3) / 2)
        pts.append([x, y])
        
    return np.array(pts)

def generate_random_init(seed):
    """
    Generate random points inside the triangle.
    """
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random() * (1 - u)
        x = v * 1.0 + (1 - u - v) * 0.5
        y = (1 - u - v) * (np.sqrt(3) / 2)
        pts.append([x, y])
    return np.array(pts)

def generate_symmetric_init(seed):
    """
    Generate a configuration with reflection symmetry across x=0.5.
    """
    rng = np.random.default_rng(seed)
    pts = []
    pts.append([0.5, np.sqrt(3)/6])
    
    for _ in range(5):
        x = 0.05 + rng.random() * 0.4
        max_y = min(np.sqrt(3)*x, np.sqrt(3)*(1-x))
        y = rng.random() * max_y * 0.9
        pts.append([x, y])
        pts.append([1-x, y])
        
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_pts = None
    
    # Initializations
    inits = []
    inits.append(generate_lattice_init())
    for s in range(5):
        inits.append(generate_symmetric_init(100 + s))
    for s in range(10):
        inits.append(generate_random_init(200 + s))
        
    # Add more random inits if time allows
    seed_counter = 300
    while time.time() < deadline - 20:
        inits.append(generate_random_init(seed_counter))
        seed_counter += 1
        if len(inits) > 50: # Cap to prevent memory issues or excessive loop overhead
            break

    # Optimization Loop
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                soft_min_obj,
                x0,
                args=(50.0,),
                method='SLSQP',
                bounds=[(0, 1)] * (2*N),
                constraints=[{'type': 'ineq', 'fun': constraints}],
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            
            if res.success:
                res_pts = res.x.reshape(N, 2)
                val = get_min_area(res_pts)
                if val > best_val:
                    best_val = val
                    best_pts = res_pts.copy()
                    
        except Exception:
            pass
            
    # Final Polish
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res_nm = minimize(
                hard_min_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_pts = res_nm.x.reshape(N, 2)
            
            x = nm_pts[:, 0]
            y = nm_pts[:, 1]
            sqrt3 = np.sqrt(3)
            
            inside = (y >= -1e-9) & (y <= sqrt3*x + 1e-9) & (y <= sqrt3*(1-x) + 1e-9)
            
            if np.all(inside):
                val_nm = get_min_area(nm_pts)
                if val_nm > best_val:
                    best_val = val_nm
                    best_pts = nm_pts.copy()
        except Exception:
            pass
            
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {get_min_area(pts)}")
