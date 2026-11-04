import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

def construct():
    # We will use a hybrid approach:
    # 1. Multiple random/greedy initializations.
    # 2. Local optimization using L-BFGS-B with adaptive penalty.
    # 3. Select the best valid configuration.
    
    def energy(params, lam):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        # 1. Boundary violations
        pen_bound = 0.0
        for i in range(N):
            d_left = x_[i] - r_[i]
            d_right = (1 - x_[i]) - r_[i]
            d_bottom = y_[i] - r_[i]
            d_top = (1 - y_[i]) - r_[i]
            
            if d_left < 0: pen_bound += d_left**2
            if d_right < 0: pen_bound += d_right**2
            if d_bottom < 0: pen_bound += d_bottom**2
            if d_top < 0: pen_bound += d_top**2
            
        # 2. Overlap violations
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x_[i] - x_[j])**2 + (y_[i] - y_[j])**2
                min_dist = r_[i] + r_[j]
                if dist_sq < min_dist**2:
                    dist = np.sqrt(dist_sq + 1e-12)
                    overlap = min_dist - dist
                    if overlap > 0:
                        pen_overlap += overlap**2
                        
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)
        
    def is_valid(x, y, r, tol=1e-5):
        # Check boundaries
        if np.any(x < r - tol) or np.any(1-x < r - tol) or np.any(y < r - tol) or np.any(1-y < r - tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist**2 - tol:
                    return False
        return True

    best_score = -1
    best_result = None
    
    # Run multiple restarts
    n_restarts = 10
    
    for k in range(n_restarts):
        # Initialization Strategy
        if k == 0:
            # Greedy Delaunay init
            def greedy_initialize(n):
                centers = np.array([
                    [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]
                ])
                radii = np.array([0.1, 0.1, 0.1, 0.1])
                circles = []
                for i in range(len(centers)):
                    circles.append((centers[i][0], centers[i][1], radii[i]))
                
                for _ in range(n - len(circles)):
                    if len(circles) < 3:
                        pass
                    pts = np.array([c[:2] for c in circles])
                    rad = np.array([c[2] for c in circles])
                    try:
                        tri = Delaunay(pts)
                    except Exception:
                        circles.append((0.5, 0.5, 0.1))
                        continue

                    best_circle = None
                    best_radius = 0.0
                    for simplex in tri.simplices:
                        p1 = pts[simplex[0]]
                        p2 = pts[simplex[1]]
                        p3 = pts[simplex[2]]
                        ax, ay = p1
                        bx, by = p2
                        cx, cy = p3
                        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                        if abs(d) < 1e-10: continue
                        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
                        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
                        r_circum = np.sqrt((ux - ax)**2 + (uy - ay)**2)
                        min_allowed = r_circum
                        for idx in simplex:
                            min_allowed = min(min_allowed, r_circum - rad[idx])
                        if min_allowed > 0:
                            r_bound = min(ux, 1 - ux, uy, 1 - uy)
                            r_new = min(min_allowed, r_bound)
                            if r_new > best_radius:
                                best_radius = r_new
                                best_circle = (ux, uy, r_new)
                    if best_circle is None:
                        cx, cy = 0.5, 0.5
                        r_bound = min(cx, 1-cx, cy, 1-cy)
                        r_max = r_bound
                        for i in range(len(circles)):
                            dx = cx - circles[i][0]
                            dy = cy - circles[i][1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            r_max = min(r_max, dist - circles[i][2])
                        if r_max > 1e-6:
                            best_circle = (cx, cy, r_max)
                        else:
                            for _ in range(100):
                                cx = np.random.uniform(0.1, 0.9)
                                cy = np.random.uniform(0.1, 0.9)
                                r_bound = min(cx, 1-cx, cy, 1-cy)
                                r_max = r_bound
                                for i in range(len(circles)):
                                    dx = cx - circles[i][0]
                                    dy = cy - circles[i][1]
                                    dist = np.sqrt(dx*dx + dy*dy)
                                    r_max = min(r_max, dist - circles[i][2])
                                if r_max > best_radius:
                                    best_radius = r_max
                                    best_circle = (cx, cy, r_max)
                    if best_circle is not None and best_radius > 1e-6:
                        circles.append(best_circle)
                    else:
                        circles.append((0.5, 0.5, 0.01))
                return np.array(circles)
            
            init_circles = greedy_initialize(N)
            x0 = init_circles[:, 0]
            y0 = init_circles[:, 1]
            r0 = init_circles[:, 2]
        else:
            # Random Initialization with separation
            x0 = np.random.uniform(0.1, 0.9, N)
            y0 = np.random.uniform(0.1, 0.9, N)
            r0 = np.full(N, 0.05)
            
            # Simple relaxation to separate
            for _ in range(50):
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x0[i] - x0[j]
                        dy = y0[i] - y0[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = r0[i] + r0[j]
                        if dist < min_dist:
                            if dist > 1e-6:
                                push = (min_dist - dist) / 2
                                x0[i] += push * dx / dist
                                y0[i] += push * dy / dist
                                x0[j] -= push * dx / dist
                                y0[j] -= push * dy / dist
                            else:
                                x0[i] += np.random.uniform(-0.01, 0.01)
                                y0[i] += np.random.uniform(-0.01, 0.01)
                x0 = np.clip(x0, r0, 1-r0)
                y0 = np.clip(y0, r0, 1-r0)
                for i in range(N):
                    r_max = min(x0[i], 1-x0[i], y0[i], 1-y0[i])
                    for j in range(N):
                        if i == j: continue
                        dx = x0[i] - x0[j]
                        dy = y0[i] - y0[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        r_max = min(r_max, dist - r0[j])
                    r0[i] = max(0.01, min(r0[i], r_max))

        params0 = np.zeros(3*N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        bounds = [(0, 1), (0, 1), (0, 0.5)] * N
        
        # Adaptive Penalty Optimization
        current_lam = 10.0
        params_opt = params0.copy()
        
        for step in range(5):
            res = minimize(
                lambda p: energy(p, current_lam), 
                params_opt, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            params_opt = res.x
            current_lam *= 5.0
            
        x_opt = params_opt[0::3]
        y_opt = params_opt[1::3]
        r_opt = params_opt[2::3]
        
        # Post-processing: Ensure strict validity
        for _ in range(20):
            changed = False
            for i in range(N):
                r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                if r_opt[i] > r_max_bound + 1e-8:
                    r_opt[i] = max(0, r_max_bound)
                    changed = True
                    
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist - 1e-8:
                        if min_dist > 0:
                            scale = dist / min_dist
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
            if not changed:
                break
        
        if is_valid(x_opt, y_opt, r_opt):
            score = np.sum(r_opt)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x_opt, y_opt, r_opt))
        else:
            # Fallback salvage
            for _ in range(50):
                for i in range(N):
                    r_opt[i] *= 0.9
                if is_valid(x_opt, y_opt, r_opt):
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_opt, y_opt, r_opt))
                    break

    if best_result is None:
        result = np.zeros((N, 3))
        cols = 6
        rows = 5
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N: break
                x = (c + 0.5) / cols
                y = (r + 0.5) / rows
                dx = 1.0 / cols
                dy = 1.0 / rows
                rad = min(dx, dy) * 0.49
                result[idx] = [x, y, rad]
                idx += 1
        return result

    return best_result
