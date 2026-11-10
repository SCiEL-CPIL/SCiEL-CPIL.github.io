import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform

N = 100

def construct():
    # Define lattice generators
    def get_fcc_points(L):
        pts = []
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                for k in range(-L, L + 1):
                    if (i + j + k) % 2 == 0:
                        pts.append([i, j, k])
        return np.array(pts, dtype=float)

    def get_bcc_points(L):
        pts = []
        # Corners and centers
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                for k in range(-L, L + 1):
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k + 0.5])
        return np.array(pts, dtype=float)

    def get_sc_points(L):
        pts = []
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                for k in range(-L, L + 1):
                    pts.append([i, j, k])
        return np.array(pts, dtype=float)

    lattices = {
        'fcc': get_fcc_points(4),
        'bcc': get_bcc_points(3),
        'sc': get_sc_points(4)
    }

    best_radius = -1
    best_centers = None

    for name, points in lattices.items():
        # Center the points
        center = np.mean(points, axis=0)
        points -= center

        # Select N points with smallest L-infinity norm (max coordinate)
        # This favors a cubic shape over a spherical one
        l_inf_norms = np.max(np.abs(points), axis=1)
        idx = np.argsort(l_inf_norms)
        points = points[idx[:N]]

        # Recenter to ensure symmetry
        center = np.mean(points, axis=0)
        points -= center

        # Calculate D_min
        dist_matrix = pdist(points)
        D_min = np.min(dist_matrix)
        
        if D_min == 0:
            continue

        # Define objective function
        def rot_matrix(angles):
            roll, pitch, yaw = angles
            c_r, s_r = np.cos(roll), np.sin(roll)
            c_p, s_p = np.cos(pitch), np.sin(pitch)
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            
            Rx = np.array([
                [1, 0, 0],
                [0, c_r, -s_r],
                [0, s_r, c_r]
            ])
            Ry = np.array([
                [c_p, 0, s_p],
                [0, 1, 0],
                [-s_p, 0, c_p]
            ])
            Rz = np.array([
                [c_y, -s_y, 0],
                [s_y, c_y, 0],
                [0, 0, 1]
            ])
            return Rz @ Ry @ Rx

        def objective(angles):
            R = rot_matrix(angles)
            rotated = points @ R.T
            W_max = np.max(np.abs(rotated))
            return W_max

        # Optimize
        bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        
        # Run DE
        de_res = differential_evolution(objective, bounds, maxiter=200, tol=1e-9, seed=42, polish=False)
        
        best_angles = de_res.x
        best_w_max = de_res.fun
        
        # Refine
        res = minimize(objective, best_angles, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.fun < best_w_max:
            best_w_max = res.fun
            best_angles = res.x

        R_opt = rot_matrix(best_angles)
        
        # Calculate scale and final positions
        s = 0.5 / (best_w_max + D_min / 2.0)
        centers = s * (points @ R_opt.T)
        
        # Calculate actual radius
        dists = pdist(centers)
        r_centers = np.min(dists) / 2 if len(dists) > 0 else 0
        r_bounds = 0.5 - np.max(np.abs(centers))
        r = min(r_centers, r_bounds)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers

    return best_centers
