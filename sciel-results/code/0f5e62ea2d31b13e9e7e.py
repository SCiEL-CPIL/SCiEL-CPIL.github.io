import numpy as np
import scipy.optimize as so
import time

N = 11

def hex_support(angle, rot):
    """
    Calculate the support function of a unit regular hexagon (side 1, circumradius 1)
    rotated by 'rot' in direction 'angle'.
    h(alpha) = max_k cos(alpha - rot - k*pi/3)
    """
    # The vertices are at angles rot + k*pi/3
    # The support function is the max projection of vertices on the direction 'angle'
    # proj_k = cos(angle - (rot + k*pi/3))
    # We can compute this efficiently
    # Due to symmetry, we can map the angle difference to [0, pi/3)
    diff = (angle - rot) % (np.pi / 3)
    # The max cos in the range [0, pi/3] for the set of angles k*pi/3 is cos(diff) if we align correctly?
    # Let's just compute the 6 values to be safe and simple, N is small.
    k = np.arange(6)
    angles = rot + k * np.pi / 3
    projs = np.cos(angle - angles)
    return np.max(projs)

def hex_support_vec(angles, rots):
    """
    Vectorized support function calculation.
    angles: (M,) array of directions
    rots: (N,) array of rotations
    Returns: (N, M) matrix of support values
    """
    # This might be slow if M and N are large, but here N=11, M=6 or 12
    # Let's stick to loops for clarity and correctness first, then optimize if needed.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_A = float('inf')
    best_inner = None
    
    # Precompute outer hexagon normals (apothem directions)
    # Outer hexagon centered at origin, rotation 0.
    # Normals are at pi/6 + k*pi/3
    outer_normals = np.pi/6 + np.arange(6) * np.pi / 3
    
    def objective(params):
        # params: [x0, y0, th0, x1, y1, th1, ..., x10, y10, th10, A]
        # 3 * N + 1 parameters
        if len(params) != 3 * N + 1:
            return 1e10
            
        centers = params[:3*N].reshape(N, 2)
        rots = params[3*N:3*N+N]
        A = params[-1]
        
        if A < 1.0: # Physical constraint, side length must be at least 1
            return 1e10
            
        penalty = 0.0
        
        # 1. Containment Constraints
        # For each inner hex i, for each outer normal k:
        # c_i . n_k + h_i(angle(n_k) - th_i) <= A
        for i in range(N):
            ci = centers[i]
            thi = rots[i]
            for k in range(6):
                nk_angle = outer_normals[k]
                nk_vec = np.array([np.cos(nk_angle), np.sin(nk_angle)])
                proj_center = np.dot(ci, nk_vec)
                sup = hex_support(nk_angle, thi)
                val = proj_center + sup
                if val > A:
                    penalty += (val - A) ** 2
        
        # 2. Non-overlap Constraints
        # For each pair i, j (i < j):
        # Must exist a separating axis d in D_ij such that:
        # d . (c_j - c_i) >= h_i(d) + h_j(-d)
        # If no such d exists, they overlap.
        # Penalty: sum over pairs of max(0, -min_sep_margin)^2
        for i in range(N):
            for j in range(i + 1, N):
                ci = centers[i]
                cj = centers[j]
                thi = rots[i]
                thj = rots[j]
                
                d_vec = cj - ci
                dist = np.linalg.norm(d_vec)
                
                # Candidate axes: normals of i and j
                # Normals of i: thi + pi/6 + k*pi/3
                # Normals of j: thj + pi/6 + k*pi/3
                min_margin = float('inf')
                
                for k in range(6):
                    # Axis from i
                    ang_i = thi + np.pi/6 + k * np.pi/3
                    d_i = np.array([np.cos(ang_i), np.sin(ang_i)])
                    proj_dist = np.dot(d_vec, d_i)
                    # Support of i in direction d_i
                    sup_i = hex_support(ang_i, thi)
                    # Support of j in direction -d_i
                    sup_j = hex_support(ang_i + np.pi, thj)
                    margin = proj_dist - (sup_i + sup_j)
                    if margin < min_margin:
                        min_margin = margin
                    
                    # Axis from j
                    ang_j = thj + np.pi/6 + k * np.pi/3
                    d_j = np.array([np.cos(ang_j), np.sin(ang_j)])
                    proj_dist_j = np.dot(d_vec, d_j)
                    sup_i_j = hex_support(ang_j, thi)
                    sup_j_j = hex_support(ang_j + np.pi, thj)
                    margin_j = proj_dist_j - (sup_i_j + sup_j_j)
                    if margin_j < min_margin:
                        min_margin = margin_j
                
                # If min_margin < 0, they overlap
                if min_margin < 0:
                    penalty += (-min_margin) ** 2
        
        # Objective: Minimize A + Penalty
        # We add A to the penalty to encourage smaller A
        return A + penalty

    # Initial guesses
    # We can try a few random restarts
    np.random.seed(42)
    
    num_restarts = 50
    restarts_done = 0
    
    while restarts_done < num_restarts and time.time() < deadline:
        # Random initialization
        # A initial guess: start with a large enough A to allow packing, e.g. 4.0
        A0 = 4.0
        centers0 = np.random.uniform(-1.5, 1.5, (N, 2))
        rots0 = np.random.uniform(0, np.pi/3, N)
        
        x0 = np.concatenate([centers0.flatten(), rots0, [A0]])
        
        # Optimize
        try:
            res = so.minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.success or res.fun < best_A:
                # Check if the result is valid (penalty is low)
                # The objective includes A, so we need to extract A and check constraints
                # Actually, if penalty is 0, objective is A.
                # We want the smallest A with penalty ~ 0.
                # Let's verify the penalty part separately or just trust the optimizer if fun is low.
                # To be safe, let's re-evaluate the "true" A and penalty.
                params = res.x
                A_val = params[-1]
                
                # Calculate penalty only
                centers = params[:3*N].reshape(N, 2)
                rots = params[3*N:3*N+N]
                
                pen = 0.0
                for i in range(N):
                    ci = centers[i]
                    thi = rots[i]
                    for k in range(6):
                        nk_angle = outer_normals[k]
                        nk_vec = np.array([np.cos(nk_angle), np.sin(nk_angle)])
                        proj_center = np.dot(ci, nk_vec)
                        sup = hex_support(nk_angle, thi)
                        val = proj_center + sup
                        if val > A_val:
                            pen += (val - A_val) ** 2
                            
                for i in range(N):
                    for j in range(i + 1, N):
                        ci = centers[i]
                        cj = centers[j]
                        thi = rots[i]
                        thj = rots[j]
                        d_vec = cj - ci
                        min_margin = float('inf')
                        for k in range(6):
                            ang_i = thi + np.pi/6 + k * np.pi/3
                            d_i = np.array([np.cos(ang_i), np.sin(ang_i)])
                            proj_dist = np.dot(d_vec, d_i)
                            sup_i = hex_support(ang_i, thi)
                            sup_j = hex_support(ang_i + np.pi, thj)
                            margin = proj_dist - (sup_i + sup_j)
                            if margin < min_margin:
                                min_margin = margin
                            
                            ang_j = thj + np.pi/6 + k * np.pi/3
                            d_j = np.array([np.cos(ang_j), np.sin(ang_j)])
                            proj_dist_j = np.dot(d_vec, d_j)
                            sup_i_j = hex_support(ang_j, thi)
                            sup_j_j = hex_support(ang_j + np.pi, thj)
                            margin_j = proj_dist_j - (sup_i_j + sup_j_j)
                            if margin_j < min_margin:
                                min_margin = margin_j
                        if min_margin < 0:
                            pen += (-min_margin) ** 2

                if pen < 1e-4:
                    if A_val < best_A:
                        best_A = A_val
                        best_inner = []
                        for i in range(N):
                            best_inner.append([centers[i][0], centers[i][1], np.degrees(rots[i])])
        except Exception as e:
            pass

        restarts_done += 1
        
        if time.time() > deadline - 10:
            break
            
    if best_inner is None:
        # Fallback to a known valid configuration (e.g. grid)
        # Grid of hexagons
        # Side length 1 hexagons.
        # Horizontal distance: 1.5 * 2 = 3? No, width is 2.
        # Flat top? Pointy top?
        # Let's just return a large valid box.
        inner = []
        for i in range(N):
            x = (i % 4) * 2.2
            y = (i // 4) * 1.9
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    return best_inner, (0.0, 0.0), best_A, 0.0

# EVOLVE-BLOCK-END
