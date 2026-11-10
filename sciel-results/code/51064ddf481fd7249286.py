import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger
    regular hexagon, minimizing the side length of the outer hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    timeout = 600  # Safety margin for 600s limit

    n = 12
    hex_side = 1.0

    # Base honeycomb 3x4 grid positions
    e1 = np.array([1.5, np.sqrt(3) / 2])
    e2 = np.array([0, np.sqrt(3)])
    base = np.array([i * e1 + j * e2 for i in range(3) for j in range(4)])
    base -= base.mean(axis=0)

    def hex_vertices(angle, center):
        """Return 6 vertices of a unit hexagon with given angle and center."""
        angles = angle + np.arange(6) * np.pi / 3
        return center + np.column_stack([np.cos(angles), np.sin(angles)])

    def get_hexagon_data(x):
        """Extract centers and angles from the parameter vector."""
        centers = x[:24].reshape((n, 2))
        angles = x[24:36]
        return centers, angles

    def compute_s(centers, angles):
        """Compute minimal enclosing outer hexagon side length."""
        n_normals = np.array([
            [np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] 
            for k in range(6)
        ])
        
        max_proj = 0.0
        for i in range(n):
            verts = hex_vertices(angles[i], centers[i])
            proj = verts @ n_normals.T
            max_proj = max(max_proj, np.max(proj))
        
        s = max_proj / np.cos(np.pi/6)
        return s

    def check_overlap_penalty(centers, angles):
        """Calculate penalty for overlapping hexagons."""
        penalty = 0.0
        
        hexes = [hex_vertices(angles[i], centers[i]) for i in range(n)]
        normals = []
        for i in range(n):
            vi = hexes[i]
            ni = np.zeros((6, 2))
            for k in range(6):
                e_i = vi[(k+1)%6] - vi[k]
                ni[k] = np.array([-e_i[1], e_i[0]])
                ni[k] /= np.linalg.norm(ni[k])
            normals.append(ni)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > 2.0:
                    continue
                
                vi = hexes[i]
                vj = hexes[j]
                all_axes = np.vstack([normals[i], normals[j]])
                
                proj_i = vi @ all_axes.T
                proj_j = vj @ all_axes.T
                
                min_i = np.min(proj_i, axis=0)
                max_i = np.max(proj_i, axis=0)
                min_j = np.min(proj_j, axis=0)
                max_j = np.max(proj_j, axis=0)
                
                gaps = np.minimum(max_i - max_j, max_j - max_i)
                min_gap = np.min(gaps)
                
                if min_gap < 0:
                    penalty += min_gap ** 2
        
        return penalty

    def objective(x):
        """Objective function for optimization."""
        centers, angles = get_hexagon_data(x)
        s = compute_s(centers, angles)
        penalty = check_overlap_penalty(centers, angles)
        return s + 100.0 * penalty

    # Initial configuration
    centers0 = base.copy()
    angles0 = np.zeros(n)
    x0 = np.concatenate([centers0.flatten(), angles0])
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    # Random restarts with Nelder-Mead
    for i in range(50):
        if time.time() - start_time > timeout - 60:
            break
            
        if i == 0:
            x_start = x0.copy()
        else:
            # Perturb the best solution found so far
            x_start = best_x + np.random.normal(0, 0.5, size=36)
            
        res = minimize(objective, x_start, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
            print(f"Restart {i}: improved to {best_f:.6f}")

    centers = best_x[:24].reshape((n, 2))
    angles = best_x[24:36]
    
    s_final = compute_s(centers, angles)
    s_return = s_final + 0.001

    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = np.degrees(angles[i])
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_return
