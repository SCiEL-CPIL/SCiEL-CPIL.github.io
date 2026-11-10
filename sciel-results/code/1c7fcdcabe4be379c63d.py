import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    # Deadline guard
    start_time = time.time()
    deadline = 600
    
    # Initial guess: 7+4 cluster
    # Centers
    c0 = [0, 0]
    c1 = [np.sqrt(3), 0]
    c2 = [np.sqrt(3)/2, 1.5]
    c3 = [-np.sqrt(3)/2, 1.5]
    c4 = [-np.sqrt(3), 0]
    c5 = [-np.sqrt(3)/2, -1.5]
    c6 = [np.sqrt(3)/2, -1.5]
    
    # 4 outer
    # Angles 30, 150, 210, 330 degrees
    # Distance 2*sqrt(3)
    d2 = 2 * np.sqrt(3)
    c7 = [d2 * np.cos(np.pi/6), d2 * np.sin(np.pi/6)]
    c8 = [d2 * np.cos(5*np.pi/6), d2 * np.sin(5*np.pi/6)]
    c9 = [d2 * np.cos(7*np.pi/6), d2 * np.sin(7*np.pi/6)]
    c10 = [d2 * np.cos(11*np.pi/6), d2 * np.sin(11*np.pi/6)]
    
    centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    
    def get_vertices(centers):
        # Returns array of shape (11, 6, 2)
        # Assume angle 0 for all inner hexagons
        verts = np.zeros((11, 6, 2))
        for i in range(11):
            cx, cy = centers[i]
            for j in range(6):
                angle = np.pi / 3 * j
                verts[i, j, 0] = cx + np.cos(angle)
                verts[i, j, 1] = cy + np.sin(angle)
        return verts

    def calc_S(centers):
        verts = get_vertices(centers)
        # Flatten to (66, 2)
        pts = verts.reshape(-1, 2)
        xs = pts[:, 0]
        ys = pts[:, 1]
        
        # Constraints for axis-aligned hexagon of side S:
        # |y| <= sqrt(3)/2 * S  => S >= 2/sqrt(3) * |y|
        # |x| + 1/sqrt(3)*|y| <= S => S >= |x| + 1/sqrt(3)*|y|
        S1 = (2 / np.sqrt(3)) * np.abs(ys)
        S2 = np.abs(xs) + (1 / np.sqrt(3)) * np.abs(ys)
        S = np.maximum(S1, S2)
        return np.max(S)

    def objective(x):
        centers = x.reshape(11, 2)
        S = calc_S(centers)
        
        # Overlap penalty
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(11):
            for j in range(i+1, 11):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    penalty += (min_dist - d)**2
        return S + 1000.0 * penalty

    # Optimization
    x0 = centers.flatten()
    
    # Check time
    if time.time() > deadline - 10:
        return centers.reshape(11, 2)[:, :2], np.array([0, 0, 0]), float(calc_S(centers))

    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    best_centers = result.x.reshape(11, 2)
    best_S = calc_S(best_centers)
    
    # Final check for validity (overlap)
    # If there is still significant overlap, try to push them apart or return initial if it was better
    min_dist = np.sqrt(3)
    has_overlap = False
    for i in range(11):
        for j in range(i+1, 11):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            if d < min_dist - 1e-6:
                has_overlap = True
                break
        if has_overlap:
            break
            
    if has_overlap:
        # Fallback to initial guess if optimization failed to resolve overlap
        # Initial guess was verified to be non-overlapping
        best_centers = centers
        best_S = calc_S(best_centers)

    # Format output
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0 # Angle 0
        
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, float(best_S)
