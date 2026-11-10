import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, minimizing the side length of the outer hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Deadline guard
    start_time = time.time()
    timeout = 590  # Safety margin for 600s limit
    
    n = 12
    
    # Axes for separation check (normals to edges of unit hexagon with angle 0)
    # u1: normal to edge (1,0)-(0.5, sqrt(3)/2)
    u1 = np.array([np.sqrt(3)/2, 0.5])
    # u2: normal to edge (0.5, sqrt(3)/2)-(-0.5, sqrt(3)/2)
    u2 = np.array([0, 1])
    # u3: normal to edge (-0.5, sqrt(3)/2)-(-1,0)
    u3 = np.array([-np.sqrt(3)/2, 0.5])
    
    axes = np.array([u1, u2, u3])
    # The width of a unit hexagon along these axes is sqrt(3)
    min_dist = np.sqrt(3)
    
    # Objective function
    def objective(x):
        # x is flat array of 24 coords: [x0, y0, x1, y1, ...]
        centers = x.reshape((n, 2))
        
        # Calculate s required to contain all vertices
        # Vertices of a unit hexagon at center c:
        # v_k = c + (cos(k*pi/3), sin(k*pi/3))
        # We need max over all centers and vertices of the projection onto the 3 containment axes.
        # Containment axes for outer hexagon (angle 0):
        # a1 = (1, 0) -> limit s
        # a2 = (0.5, sqrt(3)/2) -> limit s
        # a3 = (-0.5, sqrt(3)/2) -> limit s
        
        a1 = np.array([1, 0])
        a2 = np.array([0.5, np.sqrt(3)/2])
        a3 = np.array([-0.5, np.sqrt(3)/2])
        
        # Max projection of any vertex
        # Instead of generating all vertices, we can compute max projection of center + max projection of offset.
        # Offsets:
        # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        
        # Max projection of offsets onto a1:
        # (1,0)->1, (0.5,..)->0.5, ... Max is 1. Min is -1.
        # So for a1, max proj is max(c_x) + 1. Min proj is min(c_x) - 1.
        # We need |proj| <= s. So s >= max(|max(c_x)+1|, |min(c_x)-1|).
        # Actually, the hexagon is centered at origin.
        # Condition: max(c_x + 1) <= s AND min(c_x - 1) >= -s.
        # So s >= max(max(c_x)+1, -min(c_x)+1).
        
        # For a2 = (0.5, sqrt(3)/2):
        # Offsets dot a2:
        # (1,0) -> 0.5
        # (0.5, sqrt(3)/2) -> 0.25 + 0.75 = 1
        # (-0.5, sqrt(3)/2) -> -0.25 + 0.75 = 0.5
        # (-1,0) -> -0.5
        # (-0.5, -sqrt(3)/2) -> -0.25 - 0.75 = -1
        # (0.5, -sqrt(3)/2) -> 0.25 - 0.75 = -0.5
        # Max offset proj: 1. Min offset proj: -1.
        # So s >= max( max(centers @ a2) + 1, -min(centers @ a2) + 1 )
        
        # For a3 = (-0.5, sqrt(3)/2):
        # Offsets dot a3:
        # (1,0) -> -0.5
        # (0.5, sqrt(3)/2) -> -0.25 + 0.75 = 0.5
        # (-0.5, sqrt(3)/2) -> 0.25 + 0.75 = 1
        # (-1,0) -> 0.5
        # (-0.5, -sqrt(3)/2) -> 0.25 - 0.75 = -0.5
        # (0.5, -sqrt(3)/2) -> -0.25 - 0.75 = -1
        # Max offset proj: 1. Min offset proj: -1.
        # So s >= max( max(centers @ a3) + 1, -min(centers @ a3) + 1 )
        
        c_a1 = centers[:, 0]
        c_a2 = centers @ a2
        c_a3 = centers @ a3
        
        s1 = max(np.max(c_a1) + 1, -np.min(c_a1) + 1)
        s2 = max(np.max(c_a2) + 1, -np.min(c_a2) + 1)
        s3 = max(np.max(c_a3) + 1, -np.min(c_a3) + 1)
        
        s = max(s1, s2, s3)
        
        # Penalty for overlaps
        penalty = 0
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                projs = np.abs(diff @ axes.T)
                max_proj = np.max(projs)
                if max_proj < min_dist:
                    penalty += (min_dist - max_proj)**2
        
        return s + 10.0 * penalty

    # Initial guess: Honeycomb 3x4 grid
    # e1 = (1.5, sqrt(3)/2), e2 = (0, sqrt(3))
    e1 = np.array([1.5, np.sqrt(3)/2])
    e2 = np.array([0, np.sqrt(3)])
    
    centers_init = []
    for i in range(3):
        for j in range(4):
            p = i * e1 + j * e2
            centers_init.append(p)
    
    centers_init = np.array(centers_init)
    # Center the grid
    centroid = np.mean(centers_init, axis=0)
    centers_init -= centroid
    
    x0 = centers_init.flatten()
    
    # Optimization
    best_x = x0
    best_f = objective(x0)
    
    # Nelder-Mead
    # maxiter depends on time
    # We have ~590s.
    # Let's do a few restarts if time permits.
    
    while time.time() - start_time < timeout - 10:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
        
        # Check if we are out of time
        if time.time() - start_time > timeout - 5:
            break
            
        # Perturb for restart
        # Only if we haven't converged to a very good solution
        if best_f > 4.0: # Arbitrary threshold
            x0 = best_x + np.random.normal(0, 0.1, size=24)
        else:
            break
            
    # Extract best result
    centers = best_x.reshape((n, 2))
    
    # Calculate final s
    a1 = np.array([1, 0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    a3 = np.array([-0.5, np.sqrt(3)/2])
    
    c_a1 = centers[:, 0]
    c_a2 = centers @ a2
    c_a3 = centers @ a3
    
    s1 = max(np.max(c_a1) + 1, -np.min(c_a1) + 1)
    s2 = max(np.max(c_a2) + 1, -np.min(c_a2) + 1)
    s3 = max(np.max(c_a3) + 1, -np.min(c_a3) + 1)
    s = max(s1, s2, s3)
    
    # Construct return arrays
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0  # Angle 0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s

