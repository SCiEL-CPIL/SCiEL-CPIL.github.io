import numpy as np
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0 # Safety margin

def construct():
    sqrt3 = np.sqrt(3.0)
    
    # Generate lattice points
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    points_axial = []
    for q in range(-6, 7):
        for r in range(-6, 7):
            if abs(q + r) <= 6:
                points_axial.append((q, r))
                
    points_cart = []
    for q, r in points_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        points_cart.append((x, y))
        
    points_cart = np.array(points_cart)
    axial_to_idx = {(q, r): i for i, (q, r) in enumerate(points_axial)}
    
    def hex_dist(q, r):
        return max(abs(q), abs(r), abs(q+r))

    def calc_S(points, theta_deg):
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rot_points = points @ R.T
        
        # Normals for the hexagon
        # n1: (1, 0)
        # n2: (cos 60, sin 60)
        # n3: (cos 120, sin 120)
        n1 = np.array([1.0, 0.0])
        n2 = np.array([0.5, sqrt3 / 2.0])
        n3 = np.array([-0.5, sqrt3 / 2.0])
        
        widths = []
        for n in [n1, n2, n3]:
            proj = rot_points @ n
            w = np.max(proj) - np.min(proj)
            widths.append(w)
            
        max_w = max(widths)
        S_prime = max_w / sqrt3
        S = S_prime + 1.0
        return S

    # Initial set: 7-center cluster + 4 more
    neighbors = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    init_indices = [axial_to_idx[n] for n in neighbors]
    
    candidates = [i for i, (q, r) in enumerate(points_axial) if hex_dist(q, r) == 2]
    # Sort candidates by distance to origin to pick compact ones?
    # Or just take the first 4
    init_indices.extend(candidates[:4])
    
    current_set = np.array(init_indices)
    
    # Evaluate initial set
    best_S = float('inf')
    best_theta = 0
    for t in [0, 15, 30, 45, 60, 75, 90, 105, 120]:
        s = calc_S(points_cart[current_set], t)
        if s < best_S:
            best_S = s
            best_theta = t
            
    best_set = current_set.copy()
    
    # Local Search
    while time.time() - START_TIME < TIME_LIMIT:
        improved = False
        for i in range(11):
            q, r = points_axial[current_set[i]]
            neigh_axial = [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
            for nq, nr in neigh_axial:
                if (nq, nr) in axial_to_idx:
                    new_idx = axial_to_idx[(nq, nr)]
                    if new_idx not in current_set:
                        new_set = current_set.copy()
                        new_set[i] = new_idx
                        if len(set(new_set)) < 11:
                            continue
                        new_points = points_cart[new_set]
                        # Check a few rotations
                        min_new_S = float('inf')
                        min_new_theta = 0
                        for t in [0, 15, 30, 45, 60, 75, 90, 105, 120]:
                            s = calc_S(new_points, t)
                            if s < min_new_S:
                                min_new_S = s
                                min_new_theta = t
                                
                        if min_new_S < best_S - 1e-6:
                            best_S = min_new_S
                            best_set = new_set
                            best_theta = min_new_theta
                            current_set = new_set
                            improved = True
                            break
            if improved:
                break
        if not improved:
            break
            
    # Finalize
    # Recalculate best theta for best_set just in case
    min_S = float('inf')
    final_theta = 0
    for t in [0, 15, 30, 45, 60, 75, 90, 105, 120]:
        s = calc_S(points_cart[best_set], t)
        if s < min_S:
            min_S = s
            final_theta = t
            
    theta_rad = np.radians(final_theta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    
    final_points = points_cart[best_set] @ R.T
    
    inner = []
    for p in final_points:
        inner.append([p[0], p[1], final_theta])
        
    return inner, (0.0, 0.0), min_S, 0.0
