"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles."""
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay, KDTree
import time

N = 21
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    best_score = 0.0
    best_result = None
    
    def unpack(params):
        w = params[0]
        h = 2.0 - w
        circles = params[1:].reshape(N, 3)
        return w, h, circles
    
    def get_delaunay_edges(centers):
        """Get Delaunay triangulation edges for circle centers."""
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        edge = tuple(sorted((simplex[i], simplex[j])))
                        edges.add(edge)
            return list(edges)
        except Exception:
            # Fallback: all pairs if Delaunay fails
            return [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def solve_radii_lp(centers, w, h, edges):
        """Solve LP for maximum sum of radii given fixed centers and adjacency."""
        # Variables: r_0, ..., r_{N-1}
        # Objective: maximize sum(r_i) => minimize -sum(r_i)
        c = -np.ones(N)
        
        # Constraints:
        # 1. r_i >= 0
        # 2. r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i (boundary)
        # 3. r_i + r_j <= dist(i,j) for adjacent pairs (non-overlap)
        
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Build A_ub, b_ub for inequality constraints A_ub @ r <= b_ub
        A_ub_list = []
        b_ub_list = []
        
        # Boundary constraints: r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
        for i in range(N):
            # r_i <= x_i
            row = np.zeros(N)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(x[i])
            
            # r_i <= w - x_i
            row = np.zeros(N)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(w - x[i])
            
            # r_i <= y_i
            row = np.zeros(N)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(y[i])
            
            # r_i <= h - y_i
            row = np.zeros(N)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(h - y[i])
        
        # Non-overlap constraints for adjacent circles
        for (i, j) in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(dist)
        
        if not A_ub_list:
            return np.zeros(N)
        
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        # Bounds: r_i >= 0
        bounds = [(0, None)] * N
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                return np.maximum(result.x, 0.0)
            else:
                return np.zeros(N)
        except Exception:
            return np.zeros(N)
    
    def penalty_function(params):
        w = params[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 1e10
        circles = params[1:].reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Objective: maximize sum of radii => minimize negative sum
        obj = -np.sum(r)
        
        # Penalty for boundary violations
        pen = 0.0
        for i in range(N):
            if r[i] < 0:
                pen += 100.0 * r[i]**2
            if x[i] - r[i] < 0:
                pen += 100.0 * (x[i] - r[i])**2
            if x[i] + r[i] > w:
                pen += 100.0 * (x[i] + r[i] - w)**2
            if y[i] - r[i] < 0:
                pen += 100.0 * (y[i] - r[i])**2
            if y[i] + r[i] > h:
                pen += 100.0 * (y[i] + r[i] - h)**2
        
        # Penalty for circle-circle overlap (only check Delaunay neighbors for speed)
        centers = np.column_stack([x, y])
        edges = get_delaunay_edges(centers)
        
        for (i, j) in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                pen += 100.0 * (min_dist - dist)**2
        
        return obj + pen
    
    def make_initial_guess(w_init, h_init, seed=0):
        rng = np.random.RandomState(seed)
        params = np.zeros(1 + 3*N)
        params[0] = w_init
        # Initialize circles in a grid pattern with some randomness
        side = int(np.ceil(np.sqrt(N)))
        for idx in range(N):
            i = idx % side
            j = idx // side
            base_r = 0.5 / side
            x = (i + 0.5) * w_init / side + rng.randn() * 0.01
            y = (j + 0.5) * h_init / side + rng.randn() * 0.01
            r = base_r * min(w_init, h_init) / 1.0 + rng.randn() * 0.005
            x = np.clip(x, 0, w_init)
            y = np.clip(y, 0, h_init)
            r = max(0.001, min(r, min(w_init, h_init)/2 - 0.001))
            params[1 + 3*idx] = x
            params[2 + 3*idx] = y
            params[3 + 3*idx] = r
        return params
    
    def polish_with_lp(params):
        """Given positions and width, solve LP for optimal radii."""
        w = params[0]
        h = 2.0 - w
        circles = params[1:].reshape(N, 3)
        centers = circles[:, :2]
        
        edges = get_delaunay_edges(centers)
        r_opt = solve_radii_lp(centers, w, h, edges)
        
        # Update params with optimal radii
        new_params = params.copy()
        for i in range(N):
            new_params[3 + 3*i] = r_opt[i]
        
        return new_params
    
    # Try multiple aspect ratios and random restarts
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 1.2, 0.8, 1.8, 2.5, 0.6]
    n_restarts = 3
    
    total_attempts = len(aspect_ratios) * n_restarts
    time_per_attempt = (DEADLINE - time.time()) / total_attempts if total_attempts > 0 else 5.0
    time_per_attempt = max(1.0, min(time_per_attempt, 15.0))
    
    for ar_idx, ar in enumerate(aspect_ratios):
        if time.time() > DEADLINE:
            break
        w_init = ar * 2.0 / (ar + 1.0)
        h_init = 2.0 - w_init
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                break
            
            x0 = make_initial_guess(w_init, h_init, seed=ar_idx*100 + restart)
            
            try:
                result = minimize(
                    penalty_function,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-10}
                )
            except Exception:
                continue
            
            if result.fun < 1e6:
                # Polish with LP
                polished = polish_with_lp(result.x)
                
                w, h, circles = unpack(polished)
                r = circles[:, 2]
                x = circles[:, 0]
                y = circles[:, 1]
                
                # Validate
                valid = True
                for i in range(N):
                    if r[i] < -1e-9 or x[i] - r[i] < -1e-6 or x[i] + r[i] > w + 1e-6:
                        valid = False
                        break
                    if y[i] - r[i] < -1e-6 or y[i] + r[i] > h + 1e-6:
                        valid = False
                        break
                
                if valid:
                    centers = np.column_stack([x, y])
                    edges = get_delaunay_edges(centers)
                    for (i, j) in edges:
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if dist < r[i] + r[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        continue
                
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = polished.copy()
    
    # Final polish: high-precision LP on best solution
    if best_result is not None and time.time() < DEADLINE - 10:
        try:
            polished = polish_with_lp(best_result)
            w, h, circles = unpack(polished)
            r = circles[:, 2]
            x = circles[:, 0]
            y = circles[:, 1]
            
            valid = True
            for i in range(N):
                if r[i] < -1e-9 or x[i] - r[i] < -1e-6 or x[i] + r[i] > w + 1e-6:
                    valid = False
                    break
                if y[i] - r[i] < -1e-6 or y[i] + r[i] > h + 1e-6:
                    valid = False
                    break
            
            if valid:
                centers = np.column_stack([x, y])
                edges = get_delaunay_edges(centers)
                for (i, j) in edges:
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if dist < r[i] + r[j] - 1e-6:
                        valid = False
                        break
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = polished.copy()
        except Exception:
            pass
    
    # Build final result
    if best_result is not None:
        w, h, circles = unpack(best_result)
        result_array = np.zeros((N, 3))
        for i in range(N):
            x = np.clip(circles[i, 0], circles[i, 2], w - circles[i, 2])
            y = np.clip(circles[i, 1], circles[i, 2], h - circles[i, 2])
            r = max(0.001, min(circles[i, 2], min(w, h)/2 - 0.001))
            result_array[i] = [x, y, r]
        return result_array
    else:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for idx in range(N):
            i = idx % side
            j = idx // side
            r = 0.5 / side - 0.001
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            cells.append([x, y, r])
        return np.array(cells)
