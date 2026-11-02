import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_normals():
    """Return the 6 outward normals of a unit regular hexagon (pointy-top)."""
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _support_function(points, normals):
    """Return the support function h(u) = max_i u·p_i for each normal u."""
    # points: (N, 2), normals: (6, 2)
    return np.max(points @ normals.T, axis=0)

def _min_separation(centers, normals):
    """Return the minimum separation violation across all pairs.
    For two unit hexagons with same orientation, they are disjoint iff
    for all normals u, |u·(c_i - c_j)| >= sqrt(3) for at least one u... 
    Actually, the Minkowski difference of two unit hexagons is a hexagon of side 2.
    The distance between centers must be outside this hexagon.
    The hexagon of side 2 has inradius sqrt(3). So we need:
    max_u |u · (c_i - c_j)| >= sqrt(3)
    """
    min_viol = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[i] - centers[j]
            d_dots = np.abs(diff @ normals.T)
            max_d_dot = np.max(d_dots)
            if max_d_dot < np.sqrt(3.0):
                viol = np.sqrt(3.0) - max_d_dot
                min_viol = max(min_viol, viol)
    return min_viol

def construct():
    normals = _hex_normals()
    sqrt3 = np.sqrt(3.0)
    
    best_L = float('inf')
    best_inner = None
    best_params = None
    
    # Radial Slicing: 1 center, 5 inner ring, 6 outer ring
    # Parameters: [r_inner, r_outer, global_rotation]
    # Inner ring: 5 hexagons at angles 0, 72, 144, 216, 288 + global_rot
    # Outer ring: 6 hexagons at angles 0, 60, 120, 180, 240, 300 + global_rot
    
    def build_centers(params):
        r_in, r_out, rot = params
        rot_rad = np.deg2rad(rot)
        centers = np.zeros((N, 2))
        # Center hexagon
        centers[0] = [0.0, 0.0]
        # Inner ring: 5 hexagons
        inner_angles = np.deg2rad([0, 72, 144, 216, 288]) + rot_rad
        for k in range(5):
            centers[1 + k] = [r_in * np.cos(inner_angles[k]), r_in * np.sin(inner_angles[k])]
        # Outer ring: 6 hexagons
        outer_angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + rot_rad
        for k in range(6):
            centers[6 + k] = [r_out * np.cos(outer_angles[k]), r_out * np.sin(outer_angles[k])]
        return centers
    
    def objective(params):
        centers = build_centers(params)
        # Support function of the cluster
        dots = centers @ normals.T
        max_dots = np.max(dots, axis=0)
        min_dots = np.min(dots, axis=0)
        # The outer hexagon must contain all inner hexagons.
        # For a regular hexagon centered at origin with side L, the support in direction u is L * sqrt(3)/2.
        # So we need: max_u (u · c_i) + sqrt(3)/2 <= L * sqrt(3)/2 for all i
        # => L >= max_u (u · c_i) + 1/2
        # But we also need the negative side: -min_u (u · c_i) - sqrt(3)/2 <= L * sqrt(3)/2
        # => L >= -min_u (u · c_i) + 1/2
        # So L >= max(max_dots, -min_dots) + 1/sqrt(3) * sqrt(3)/2 ... let me redo.
        
        # For a regular hexagon with side length L centered at origin, the inradius is L*sqrt(3)/2.
        # The support function in direction u is L*sqrt(3)/2.
        # For an inner hexagon centered at c with side 1, its support in direction u is u·c + sqrt(3)/2.
        # We need: u·c + sqrt(3)/2 <= L*sqrt(3)/2 for all u, c
        # => L >= (u·c + sqrt(3)/2) / (sqrt(3)/2) = (2*u·c)/sqrt(3) + 1
        # Also need: -u·c + sqrt(3)/2 <= L*sqrt(3)/2 => L >= (-2*u·c)/sqrt(3) + 1
        # So L >= max over all u, c of max(2*u·c, -2*u·c)/sqrt(3) + 1 = 2*max|u·c|/sqrt(3) + 1
        
        max_abs_dot = np.max(np.abs(centers @ normals.T))
        L_req = 2.0 * max_abs_dot / sqrt3 + 1.0
        
        # Separation constraint
        viol = _min_separation(centers, normals)
        
        return L_req + 1000.0 * viol * viol
    
    # Try multiple random starts
    for seed in range(50):
        np.random.seed(seed)
        x0 = np.array([
            np.random.uniform(1.5, 3.5),   # r_inner
            np.random.uniform(3.0, 6.0),   # r_outer
            np.random.uniform(0, 360)       # rotation
        ])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            params = result.x
            centers = build_centers(params)
            
            # Verify validity
            viol = _min_separation(centers, normals)
            if viol > 1e-5:
                continue
            
            # Compute final L
            max_abs_dot = np.max(np.abs(centers @ normals.T))
            L_final = 2.0 * max_abs_dot / sqrt3 + 1.0
            
            if L_final < best_L:
                best_L = L_final
                best_params = params.copy()
                best_inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
        except Exception:
            continue
    
    if best_inner is None:
        # Fallback: try a wider grid search
        best_L = float('inf')
        for r_in in np.linspace(1.5, 4.0, 20):
            for r_out in np.linspace(3.0, 8.0, 20):
                for rot in np.linspace(0, 60, 12):
                    centers = build_centers(np.array([r_in, r_out, rot]))
                    viol = _min_separation(centers, normals)
                    if viol > 1e-5:
                        continue
                    max_abs_dot = np.max(np.abs(centers @ normals.T))
                    L_final = 2.0 * max_abs_dot / sqrt3 + 1.0
                    if L_final < best_L:
                        best_L = L_final
                        best_inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
        
        if best_inner is None:
            # Last resort fallback
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, (0.0, 0.0), best_L, 0.0
