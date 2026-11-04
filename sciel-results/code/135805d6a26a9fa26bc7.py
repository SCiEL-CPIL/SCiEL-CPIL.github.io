import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 26

def construct():
    
    def largest_empty_circle(centers, radii):
        """
        Finds the center and radius of the largest circle that can be added
        to the given set of circles without overlapping, within [0,1]x[0,1].
        """
        if len(centers) == 0:
            return (0.5, 0.5, 0.5)
        
        centers = np.array(centers)
        radii = np.array(radii)
        
        best_r = 0
        best_c = None
        
        # 1. Candidate centers from Voronoi vertices
        if len(centers) >= 3:
            try:
                vor = Voronoi(centers)
                
                # Voronoi vertices
                for v in vor.vertices:
                    if np.any(np.isnan(v)):
                        continue
                    if v[0] < 0 or v[0] > 1 or v[1] < 0 or v[1] > 1:
                        continue
                    
                    r = min(v[0], 1-v[0], v[1], 1-v[1])
                    for i in range(len(centers)):
                        d = np.sqrt((v[0]-centers[i][0])**2 + (v[1]-centers[i][1])**2)
                        r = min(r, d - radii[i])
                    
                    if r > best_r:
                        best_r = r
                        best_c = v
                        
            except Exception:
                pass
        
        # 2. Candidate centers on the boundary of the square
        num_samples = 500
        for side in range(4):
            for t in np.linspace(0, 1, num_samples):
                if side == 0: # y=0
                    x, y = t, 0.0
                elif side == 1: # y=1
                    x, y = t, 1.0
                elif side == 2: # x=0
                    x, y = 0.0, t
                else: # x=1
                    x, y = 1.0, t
                
                r = min(x, 1-x, y, 1-y)
                for i in range(len(centers)):
                    d = np.sqrt((x-centers[i][0])**2 + (y-centers[i][1])**2)
                    r = min(r, d - radii[i])
                
                if r > best_r:
                    best_r = r
                    best_c = np.array([x, y])
        
        # 3. Circle-Circle intersection points
        n_centers = len(centers)
        for i in range(n_centers):
            for j in range(i+1, n_centers):
                c1 = centers[i]
                c2 = centers[j]
                r1 = radii[i]
                r2 = radii[j]
                
                d = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                
                if d < 1e-6 or d > r1 + r2 or d < abs(r1 - r2):
                    continue
                
                a = (d**2 + r1**2 - r2**2) / (2*d)
                h = np.sqrt(max(0, r1**2 - a**2))
                
                mid_x = c1[0] + a * (c2[0] - c1[0]) / d
                mid_y = c1[1] + a * (c2[1] - c1[1]) / d
                
                off_x = h * (c2[1] - c1[1]) / d
                off_y = h * (c1[0] - c2[0]) / d
                
                p1 = np.array([mid_x + off_x, mid_y + off_y])
                p2 = np.array([mid_x - off_x, mid_y - off_y])
                
                for p in [p1, p2]:
                    if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1:
                        continue
                    
                    r = min(p[0], 1-p[0], p[1], 1-p[1])
                    for k in range(n_centers):
                        d_k = np.sqrt((p[0]-centers[k][0])**2 + (p[1]-centers[k][1])**2)
                        r = min(r, d_k - radii[k])
                    
                    if r > best_r:
                        best_r = r
                        best_c = p
        
        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 0.01
            
        return best_c[0], best_c[1], max(best_r, 1e-6)

    def greedy_initialize(n, pattern='corners'):
        centers = []
        radii = []
        
        if pattern == 'corners':
            for i in range(4):
                x = 0.1 if i % 2 == 0 else 0.9
                y = 0.1 if i < 2 else 0.9
                centers.append([x, y])
                radii.append(0.1)
        elif pattern == 'grid':
            # Start with a 5x5 grid-like pattern
            positions = [
                [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
                [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
                [0.1, 0.9], [0.5, 0.9], [0.9, 0.9]
            ]
            for pos in positions:
                centers.append(pos)
                radii.append(0.08)
        elif pattern == 'hex':
            # Hexagonal packing initial
            rows = [0.1, 0.35, 0.6, 0.85]
            cols_offset = [0.0, 0.125, 0.0, 0.125]
            count = 0
            for i, y in enumerate(rows):
                for j in range(7):
                    if count >= 26:
                        break
                    x = 0.1 + j * 0.125 + cols_offset[i]
                    if x <= 0.9:
                        centers.append([x, y])
                        radii.append(0.06)
                        count += 1
                if count >= 26:
                    break
        
        while len(centers) < n:
            c_x, c_y, r = largest_empty_circle(centers, radii)
            centers.append([c_x, c_y])
            radii.append(r)
        
        return np.array(centers[:n]), np.array(radii[:n])

    def relax_positions(centers, iterations=50, min_dist=0.12):
        """
        Pushes overlapping points apart to maximize the minimum distance.
        This helps create a better initial spread for the optimizer.
        """
        centers = np.array(centers, dtype=float)
        n = len(centers)
        
        for _ in range(iterations):
            for i in range(n):
                for j in range(i+1, n):
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    if d < min_dist:
                        if d < 1e-6:
                            d_vec = np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01)])
                            d = np.linalg.norm(d_vec)
                        push = (min_dist - d) / 2.0
                        centers[i] += (d_vec / d) * push
                        centers[j] -= (d_vec / d) * push
            
            # Keep inside bounds
            centers = np.clip(centers, 0.05, 0.95)
            
        return centers

    def energy(params, lam=50.0):
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
                    
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)

    def optimize(c_init, r_init, maxiter=500, lam=50.0):
        params0 = np.zeros(3*N)
        params0[0::3] = c_init[:, 0]
        params0[1::3] = c_init[:, 1]
        params0[2::3] = r_init
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
            
        def energy_local(params):
            return energy(params, lam)
        
        res = minimize(energy_local, params0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        
        params_opt = res.x
        x_opt = params_opt[0::3]
        y_opt = params_opt[1::3]
        r_opt = params_opt[2::3]
        
        # Post-processing: Ensure strict validity
        for _ in range(30):
            changed = False
            for i in range(N):
                r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                if r_opt[i] > r_max_bound:
                    r_opt[i] = r_max_bound
                    changed = True
                    
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        if min_dist > 1e-9:
                            scale = dist / min_dist
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
            if not changed:
                break
                
        return x_opt, y_opt, r_opt

    def local_refinement(x, y, r, num_trials=50):
        """
        Try swapping radii between pairs of circles and re-optimizing locally
        to see if we can improve the sum.
        """
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for trial in range(num_trials):
            i, j = np.random.choice(N, 2, replace=False)
            # Swap radii
            r_swap = r.copy()
            r_swap[i], r_swap[j] = r_swap[j], r_swap[i]
            
            # Try optimizing with swapped radii
            x_s, y_s, r_s = optimize(x, y, r_swap, maxiter=200, lam=30.0)
            score = np.sum(r_s)
            
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x_s, y_s, r_s
        
        return best_x, best_y, best_r

    # 1. Generate initial configurations using multiple patterns
    np.random.seed(42)
    
    patterns = ['corners', 'grid', 'hex']
    best_score = -1
    best_x, best_y, best_r = None, None, None
    
    for pattern in patterns:
        c_init, r_init = greedy_initialize(N, pattern=pattern)
        x_opt, y_opt, r_opt = optimize(c_init, r_init, maxiter=400)
        score = np.sum(r_opt)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_opt, y_opt, r_opt
    
    # 2. Multi-start: Try random init + relaxation
    for i in range(15):
        c_rand = np.random.uniform(0.1, 0.9, (N, 2))
        c_relaxed = relax_positions(c_rand, iterations=40, min_dist=0.11)
        
        # Estimate radii based on relaxed positions
        r_est = np.zeros(N)
        for i in range(N):
            min_dist = min(c_relaxed[i][0], 1-c_relaxed[i][0], c_relaxed[i][1], 1-c_relaxed[i][1])
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(c_relaxed[i] - c_relaxed[j])
                    min_dist = min(min_dist, d/2.0)
            r_est[i] = max(min_dist * 0.85, 0.01)
            
        x_p, y_p, r_p = optimize(c_relaxed, r_est, maxiter=300)
        score = np.sum(r_p)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p
    
    # 3. Multi-start: Perturb best solution
    for i in range(10):
        c_pert = best_x[:, None] * np.ones((1, 2))
        c_pert[:, 0] = best_x
        c_pert[:, 1] = best_y
        c_pert += np.random.normal(0, 0.03, (N, 2))
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert = np.clip(best_r * 0.85, 0.01, 0.4)
        
        x_p, y_p, r_p = optimize(c_pert, r_pert, maxiter=300)
        score = np.sum(r_p)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p
    
    # 4. Local refinement
    best_x, best_y, best_r = local_refinement(best_x, best_y, best_r, num_trials=30)
    
    # 5. Final validation and formatting
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
