import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26

def construct():
    
    def largest_empty_circle(centers, radii):
        """
        Finds the center and radius of the largest circle that can be added
        to the given set of circles without overlapping, within [0,1]x[0,1].
        Uses Voronoi vertices and bisector-boundary intersections.
        """
        if len(centers) == 0:
            return (0.5, 0.5, 0.5)
        
        centers = np.array(centers)
        radii = np.array(radii)
        
        best_r = 0
        best_c = None
        
        def evaluate_point(p):
            # p is (x, y)
            if p[0] < -1e-9 or p[0] > 1 + 1e-9 or p[1] < -1e-9 or p[1] > 1 + 1e-9:
                return 0, None
            # Clamp to box for safety in radius calc, but strictly check bounds
            px = np.clip(p[0], 0, 1)
            py = np.clip(p[1], 0, 1)
            r = min(px, 1-px, py, 1-py)
            for i in range(len(centers)):
                d = np.sqrt((px-centers[i][0])**2 + (py-centers[i][1])**2)
                r = min(r, d - radii[i])
            return r, np.array([px, py])

        # 1. Voronoi Vertices
        if len(centers) >= 3:
            try:
                vor = Voronoi(centers)
                for v in vor.vertices:
                    if np.any(np.isnan(v)):
                        continue
                    r, p = evaluate_point(v)
                    if r > best_r:
                        best_r = r
                        best_c = p
            except Exception:
                pass
        
        # 2. Intersections of Pair Bisectors with Box Boundary
        # This covers the case where the LEC center is on the boundary of the box
        # and equidistant to two circle centers.
        n_centers = len(centers)
        for i in range(n_centers):
            for j in range(i+1, n_centers):
                c1 = centers[i]
                c2 = centers[j]
                
                # Midpoint
                mid = (c1 + c2) / 2.0
                # Direction vector of the line c1->c2
                dir_vec = c2 - c1
                # Normal vector to the bisector (perpendicular to dir_vec)
                normal = np.array([-dir_vec[1], dir_vec[0]])
                
                # Parametric form: P(t) = mid + t * normal
                # Intersect with x=0, x=1, y=0, y=1
                
                candidates = []
                
                # x = 0
                if abs(normal[0]) > 1e-9:
                    t = -mid[0] / normal[0]
                    p = mid + t * normal
                    candidates.append(p)
                    
                # x = 1
                if abs(normal[0]) > 1e-9:
                    t = (1 - mid[0]) / normal[0]
                    p = mid + t * normal
                    candidates.append(p)
                    
                # y = 0
                if abs(normal[1]) > 1e-9:
                    t = -mid[1] / normal[1]
                    p = mid + t * normal
                    candidates.append(p)
                    
                # y = 1
                if abs(normal[1]) > 1e-9:
                    t = (1 - mid[1]) / normal[1]
                    p = mid + t * normal
                    candidates.append(p)
                    
                for p in candidates:
                    r, pt = evaluate_point(p)
                    if r > best_r:
                        best_r = r
                        best_c = pt

        # 3. Box Corners
        for x in [0.0, 1.0]:
            for y in [0.0, 1.0]:
                r, p = evaluate_point(np.array([x, y]))
                if r > best_r:
                    best_r = r
                    best_c = p

        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 1e-6
            
        return best_c[0], best_c[1], max(best_r, 1e-6)

    def greedy_initialize(n):
        centers = []
        radii = []
        
        # Start with a central circle
        centers.append([0.5, 0.5])
        radii.append(0.5)
        
        for _ in range(n - 1):
            c_x, c_y, r = largest_empty_circle(centers, radii)
            centers.append([c_x, c_y])
            radii.append(r)
            
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
                    
        lam = 50.0
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)

    def optimize(c_init, r_init):
        params0 = np.zeros(3*N)
        params0[0::3] = c_init[:, 0]
        params0[1::3] = c_init[:, 1]
        params0[2::3] = r_init
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
            
        res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        
        params_opt = res.x
        x_opt = params_opt[0::3]
        y_opt = params_opt[1::3]
        r_opt = params_opt[2::3]
        
        # Post-processing: Ensure strict validity
        for _ in range(50):
            changed = False
            # Boundary fix
            for i in range(N):
                r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                if r_opt[i] > r_max_bound:
                    r_opt[i] = r_max_bound
                    changed = True
                    
            # Overlap fix
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        # Scale down both radii
                        if min_dist > 1e-9:
                            scale = dist / min_dist
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
            if not changed:
                break
                
        return x_opt, y_opt, r_opt

    # 1. Generate initial configuration using Greedy Method
    c_init, r_init = greedy_initialize(N)
    
    # 2. Optimize
    x_opt, y_opt, r_opt = optimize(c_init, r_init)
    best_score = np.sum(r_opt)
    best_x, best_y, best_r = x_opt, y_opt, r_opt
    
    # 3. Multi-start: Try more random perturbations and random initializations
    np.random.seed(42) # Reproducibility
    for i in range(50):
        if i % 2 == 0:
            # Perturb greedy start
            c_pert = c_init + np.random.normal(0, 0.05, (N, 2))
            c_pert = np.clip(c_pert, 0.01, 0.99)
            r_pert = np.clip(r_init * 0.8, 0.01, 0.4)
        else:
            # Random start
            c_pert = np.random.uniform(0.1, 0.9, (N, 2))
            r_pert = np.random.uniform(0.02, 0.1, N)
        
        x_p, y_p, r_p = optimize(c_pert, r_pert)
        score = np.sum(r_p)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p

    # 4. Final validation and formatting
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
