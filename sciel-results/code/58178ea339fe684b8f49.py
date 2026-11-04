import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 26

def construct():
    def compute_largest_empty_circle(centers, radii):
        """
        Find the largest circle that fits among existing circles and within [0,1]^2.
        Uses Voronoi diagram and circle-circle intersections to identify candidate positions.
        """
        if len(centers) == 0:
            return np.array([0.5, 0.5]), 0.5
        
        centers = np.array(centers, dtype=float)
        radii = np.array(radii, dtype=float)
        n = len(centers)
        
        best_r = 0.0
        best_c = np.array([0.5, 0.5])
        
        # Candidate 1: Voronoi vertices
        if n >= 3:
            try:
                vor = Voronoi(centers)
                for v in vor.vertices:
                    if np.any(np.isnan(v)):
                        continue
                    if v[0] < -1e-9 or v[0] > 1+1e-9 or v[1] < -1e-9 or v[1] > 1+1e-9:
                        continue
                    v = np.clip(v, 0, 1)
                    r = min(v[0], 1-v[0], v[1], 1-v[1])
                    for i in range(n):
                        d = np.linalg.norm(v - centers[i])
                        r = min(r, d - radii[i])
                    if r > best_r:
                        best_r = r
                        best_c = v
            except Exception:
                pass
        
        # Candidate 2: Circle-circle intersection points
        for i in range(n):
            for j in range(i+1, n):
                c1, c2 = centers[i], centers[j]
                r1, r2 = radii[i], radii[j]
                d = np.linalg.norm(c1 - c2)
                if d < 1e-9 or d > r1 + r2 or d < abs(r1 - r2):
                    continue
                a = (d**2 + r1**2 - r2**2) / (2*d)
                h2 = r1**2 - a**2
                if h2 < 0:
                    continue
                h = np.sqrt(h2)
                mid = c1 + a * (c2 - c1) / d
                perp = np.array([-(c2[1]-c1[1])/d, (c2[0]-c1[0])/d])
                for sign in [1, -1]:
                    p = mid + sign * h * perp
                    if p[0] < -1e-9 or p[0] > 1+1e-9 or p[1] < -1e-9 or p[1] > 1+1e-9:
                        continue
                    p = np.clip(p, 0, 1)
                    r = min(p[0], 1-p[0], p[1], 1-p[1])
                    for k in range(n):
                        dk = np.linalg.norm(p - centers[k])
                        r = min(r, dk - radii[k])
                    if r > best_r:
                        best_r = r
                        best_c = p
        
        # Candidate 3: Grid search for robustness
        grid_n = 15
        for gx in np.linspace(0, 1, grid_n):
            for gy in np.linspace(0, 1, grid_n):
                p = np.array([gx, gy])
                r = min(p[0], 1-p[0], p[1], 1-p[1])
                for k in range(n):
                    dk = np.linalg.norm(p - centers[k])
                    r = min(r, dk - radii[k])
                if r > best_r:
                    best_r = r
                    best_c = p
        
        return best_c, max(best_r, 1e-8)
    
    def greedy_pack(n):
        """Greedily place n circles, always picking the largest possible next circle."""
        centers = []
        radii = []
        
        # Place first circles in corners for good initial spread
        corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        for c in corners:
            centers.append(c)
            radii.append(0.12)
        
        # Add center
        centers.append([0.5, 0.5])
        radii.append(0.15)
        
        while len(centers) < n:
            c_new, r_new = compute_largest_empty_circle(centers, radii)
            if r_new < 1e-6:
                c_new = np.random.uniform(0.1, 0.9, 2)
                r_new = 0.01
            centers.append(c_new)
            radii.append(r_new)
        
        return np.array(centers), np.array(radii)
    
    def energy(params):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        pen_bound = 0.0
        for i in range(N):
            pen_bound += max(0, r_[i] - x_[i])**2
            pen_bound += max(0, r_[i] - (1 - x_[i]))**2
            pen_bound += max(0, r_[i] - y_[i])**2
            pen_bound += max(0, r_[i] - (1 - y_[i]))**2
            
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x_[i] - x_[j])**2 + (y_[i] - y_[j])**2
                min_dist = r_[i] + r_[j]
                if dist_sq < min_dist**2:
                    overlap = min_dist - np.sqrt(dist_sq)
                    pen_overlap += overlap**2
                    
        lam = 100.0
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)
    
    def optimize(c_init, r_init, maxiter=200):
        params0 = np.zeros(3*N)
        params0[0::3] = c_init[:, 0]
        params0[1::3] = c_init[:, 1]
        params0[2::3] = r_init
        
        bounds = [(0, 1), (0, 1), (0, 0.5)] * N
        
        res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        
        params_opt = res.x
        x_opt = params_opt[0::3].copy()
        y_opt = params_opt[1::3].copy()
        r_opt = params_opt[2::3].copy()
        
        # Post-processing: ensure validity
        for _ in range(30):
            changed = False
            for i in range(N):
                r_max = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                if r_opt[i] > r_max:
                    r_opt[i] = r_max
                    changed = True
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist - 1e-12:
                        if min_dist > 1e-9:
                            scale = (dist - 1e-10) / min_dist
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
            if not changed:
                break
        
        return x_opt, y_opt, r_opt
    
    # Main: greedy init + multi-start optimization
    np.random.seed(123)
    
    # Greedy initialization
    c_greedy, r_greedy = greedy_pack(N)
    best_x, best_y, best_r = c_greedy[:, 0].copy(), c_greedy[:, 1].copy(), r_greedy.copy()
    best_score = np.sum(best_r)
    
    # Optimize from greedy start
    x_o, y_o, r_o = optimize(c_greedy, r_greedy)
    if np.sum(r_o) > best_score:
        best_x, best_y, best_r = x_o, y_o, r_o
        best_score = np.sum(r_o)
    
    # Multi-start: perturb greedy
    for trial in range(15):
        c_pert = c_greedy + np.random.normal(0, 0.03, (N, 2))
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert = np.clip(r_greedy * np.random.uniform(0.7, 1.0, N), 0.01, 0.45)
        
        x_p, y_p, r_p = optimize(c_pert, r_pert, maxiter=150)
        score = np.sum(r_p)
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p
    
    # Multi-start: random init
    for trial in range(10):
        c_rand = np.random.uniform(0.05, 0.95, (N, 2))
        r_rand = np.full(N, 0.08)
        x_p, y_p, r_p = optimize(c_rand, r_rand, maxiter=150)
        score = np.sum(r_p)
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p
    
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
