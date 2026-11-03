"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_hex_pts(rows):
    """
    Generates coordinates for a hexagonal cluster defined by row counts.
    The lattice is defined such that the nearest neighbor distance is sqrt(3).
    """
    raw_pts = []
    for j, count in enumerate(rows):
        for i in range(count):
            x = i * np.sqrt(3)
            y = j * 1.5
            if j % 2 == 1:
                x += np.sqrt(3) / 2
            raw_pts.append((x, y))
    
    if not raw_pts:
        return np.array([]).reshape(0, 2)
        
    pts = np.array(raw_pts)
    # Center the points around the origin
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    pts = pts - np.array([cx, cy])
    return pts

def _max_radius_for_pts(pts):
    """
    Finds the maximum radius r for a given set of centered lattice points
    (with nearest neighbor distance sqrt(3)) such that the bounding rectangle
    has perimeter <= 4.
    Returns (best_r, best_theta, best_rot_pts, best_W, best_H).
    """
    best_r = 0
    best_theta = 0
    best_rot_pts = None
    best_W = 0
    best_H = 0
    
    # Coarse search
    thetas = np.linspace(0, np.pi/2, 100)
    
    for theta in thetas:
        c, s = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([[c, -s], [s, c]])
        rot_pts = pts @ rot_matrix.T
        
        xs = rot_pts[:, 0]
        ys = rot_pts[:, 1]
        
        w_c = np.max(xs) - np.min(xs)
        h_c = np.max(ys) - np.min(ys)
        
        # r <= 2 / ( (2/sqrt(3))*(w_c + h_c) + 4 )
        denom = (2.0 / np.sqrt(3)) * (w_c + h_c) + 4.0
        r = 2.0 / denom
        
        if r > best_r:
            best_r = r
            best_theta = theta
            best_rot_pts = rot_pts
            best_W = r * ((2.0 / np.sqrt(3)) * w_c + 2.0)
            best_H = r * ((2.0 / np.sqrt(3)) * h_c + 2.0)
            
    # Fine search around best_theta
    if best_r > 0:
        fine_thetas = np.linspace(best_theta - 0.05, best_theta + 0.05, 100)
        for theta in fine_thetas:
            c, s = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([[c, -s], [s, c]])
            rot_pts = pts @ rot_matrix.T
            
            xs = rot_pts[:, 0]
            ys = rot_pts[:, 1]
            
            w_c = np.max(xs) - np.min(xs)
            h_c = np.max(ys) - np.min(ys)
            
            denom = (2.0 / np.sqrt(3)) * (w_c + h_c) + 4.0
            r = 2.0 / denom
            
            if r > best_r:
                best_r = r
                best_theta = theta
                best_rot_pts = rot_pts
                best_W = r * ((2.0 / np.sqrt(3)) * w_c + 2.0)
                best_H = r * ((2.0 / np.sqrt(3)) * h_c + 2.0)

    return best_r, best_theta, best_rot_pts, best_W, best_H

def construct():
    start_time = time.time()
    
    # Candidate topologies summing to 21
    topologies = [
        [4, 5, 4, 4, 4],
        [5, 4, 5, 4, 3],
        [3, 4, 5, 4, 3, 2],
        [4, 4, 5, 4, 4],
        [2, 3, 4, 5, 4, 3],
        [1, 2, 3, 4, 5, 4, 2],
        [5, 5, 5, 3, 3],
        [6, 4, 4, 3, 4],
        [3, 5, 4, 5, 4],
        [4, 5, 5, 3, 4]
    ]
    
    best_global_r = 0
    best_global_pts = None
    best_global_W = 0
    best_global_H = 0
    
    for topo in topologies:
        if sum(topo) != N:
            continue
            
        pts = _generate_hex_pts(topo)
        if pts.shape[0] != N:
            continue
            
        r, theta, rot_pts, W, H = _max_radius_for_pts(pts)
        
        if r > best_global_r:
            best_global_r = r
            best_global_pts = rot_pts
            best_global_W = W
            best_global_H = H
            
        # Deadline guard
        if time.time() - start_time > 100:
            break
            
    if best_global_pts is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final Construction
    # Shift points so that min_x = r, min_y = r
    min_x = np.min(best_global_pts[:, 0])
    min_y = np.min(best_global_pts[:, 1])
    
    shift_x = best_global_r - min_x
    shift_y = best_global_r - min_y
    
    final_pts = best_global_pts.copy()
    final_pts[:, 0] += shift_x
    final_pts[:, 1] += shift_y
    
    result = np.zeros((N, 3))
    result[:, 0] = final_pts[:, 0]
    result[:, 1] = final_pts[:, 1]
    result[:, 2] = best_global_r
    
    return result
# EVOLVE-BLOCK-END
