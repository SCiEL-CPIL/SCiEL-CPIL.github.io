"""Improved packing of 300 equal circles in a unit circle using smooth optimization."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def compute_score(points):
        """Compute the minimum radius for the entire configuration using KDTree."""
        tree = cKDTree(points)
        min_dist = np.inf
        
        # Check pairwise distances using KDTree
        for i in range(N):
            neighbors = tree.query_ball_point(points[i], 0.2)
            for j in neighbors:
                if j > i:
                    d = np.linalg.norm(points[i] - points[j])
                    if d < min_dist:
                        min_dist = d
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)

    def soft_min_objective(x, beta=50.0):
        """Smooth soft-min objective: maximize this (negative of soft-min radius)."""
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances using KDTree
        tree = cKDTree(points)
        pair_dists = []
        for i in range(N):
            neighbors = tree.query_ball_point(points[i], 0.3)
            for j in neighbors:
                if j > i:
                    d = np.linalg.norm(points[i] - points[j])
                    pair_dists.append(d / 2.0)
        
        if not pair_dists:
            pair_dists = [1.0]
        
        pair_dists = np.array(pair_dists)
        
        # Boundary constraints
        radii = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - radii
        
        # Combine all constraints
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Soft min via log-sum-exp
        max_d = np.max(all_dists)
        log_sum = np.log(np.sum(np.exp(-(all_dists - max_d) * beta))) / beta
        soft_min = max_d - log_sum
        
        return -soft_min

    def initial_hex(n):
        """Generate initial hexagonal lattice points."""
        s_est = math.sqrt(2 * math.pi * 0.95**2 / (n * math.sqrt(3)))
        
        best_points = None
        best_score = -1
        
        for trial in range(20):
            spacing = s_est * (1.0 + trial * 0.005)
            points = []
            r_max = int(1.0 / (spacing * math.sqrt(3) / 2)) + 2
            for j in range(-r_max, r_max + 1):
                for i in range(-r_max, r_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * math.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                        points.append((x, y))
            
            if len(points) >= n:
                points_arr = np.array(points)
                dists = np.hypot(points_arr[:, 0], points_arr[:, 1])
                idx = np.argsort(dists)[:n]
                candidate = points_arr[idx]
                
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
            
            if time.time() > deadline - 5:
                break
        
        if best_points is None:
            best_points = np.random.randn(n, 2) * 0.8
            max_r = np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
            if max_r > 0.9:
                best_points *= 0.9 / max_r
        return best_points

    def nelder_mead_optimize(points, max_iter=200, beta=50.0):
        """Use Nelder-Mead on the soft-min objective."""
        x0 = points.flatten()
        
        # Bounds: keep points inside unit circle with small margin
        bounds = [(-0.99, 0.99)] * (2 * N)
        
        result = minimize(
            soft_min_objective,
            x0,
            args=(beta,),
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
        )
        
        opt_points = result.x.reshape(-1, 2)
        
        # Project to unit circle
        radii = np.linalg.norm(opt_points, axis=1)
        mask = radii > 0.999
        if np.any(mask):
            opt_points[mask] *= 0.999 / radii[mask]
        
        return opt_points, compute_score(opt_points)

    def local_optimize(points, max_iter=300):
        """Fast local optimization using KDTree for neighbor queries."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.01
        no_improve_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                
                for _ in range(3):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = lr * math.cos(angle)
                    dy = lr * math.sin(angle)
                    
                    old_x, old_y = points[i]
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    r_new = math.hypot(new_x, new_y)
                    if r_new > 0.999:
                        scale = 0.999 / r_new
                        new_x *= scale
                        new_y *= scale
                    
                    # Use KDTree for fast neighbor distance query
                    temp_points = points.copy()
                    temp_points[i] = [new_x, new_y]
                    tree = cKDTree(temp_points)
                    
                    # Find closest neighbor
                    dists, _ = tree.query([new_x, new_y], k=2)
                    closest_dist = dists[1] if len(dists) > 1 else np.inf
                    
                    bound_dist = 1.0 - math.hypot(new_x, new_y)
                    
                    new_local_r = min(closest_dist / 2.0, bound_dist)
                    
                    if new_local_r > current_score:
                        points[i] = [new_x, new_y]
                        improved = True
                        break
            
            if it % 30 == 0 or not improved:
                current_score = compute_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    lr *= 0.95
                    if no_improve_count > 10 or lr < 1e-5:
                        break
            else:
                no_improve_count = 0
        
        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
            
        return best_points, best_score

    def high_precision_polish(points, max_iter=100):
        """Final high-precision polish with tighter tolerances."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.001
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 1.0:
                break
            
            # Fine-tune using small random perturbations
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline - 1.0:
                    break
                
                for _ in range(5):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = lr * math.cos(angle)
                    dy = lr * math.sin(angle)
                    
                    new_x, new_y = points[i][0] + dx, points[i][1] + dy
                    
                    r_new = math.hypot(new_x, new_y)
                    if r_new > 0.9999:
                        scale = 0.9999 / r_new
                        new_x *= scale
                        new_y *= scale
                    
                    # Check if improvement
                    temp_points = points.copy()
                    temp_points[i] = [new_x, new_y]
                    
                    tree = cKDTree(temp_points)
                    dists, _ = tree.query([new_x, new_y], k=2)
                    closest_dist = dists[1] if len(dists) > 1 else np.inf
                    
                    bound_dist = 1.0 - math.hypot(new_x, new_y)
                    new_local_r = min(closest_dist / 2.0, bound_dist)
                    
                    if new_local_r > current_score + 1e-10:
                        points[i] = [new_x, new_y]
                        break
            
            if it % 10 == 0:
                current_score = compute_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    lr *= 0.9
                    if no_improve > 5 or lr < 1e-7:
                        break
        
        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
            
        return best_points, best_score

    # Generate initial configuration
    points = initial_hex(N)
    
    # Phase 1: Nelder-Mead optimization with soft-min surrogate
    points, score = nelder_mead_optimize(points, max_iter=150, beta=50.0)
    
    # Phase 2: Local random search with KDTree
    points, score2 = local_optimize(points, max_iter=200)
    if score2 > score:
        score = score2
    
    # Phase 3: Fine Nelder-Mead polish
    if time.time() < deadline - 30:
        points, score3 = nelder_mead_optimize(points, max_iter=80, beta=100.0)
        if score3 > score:
            score = score3
    
    # Phase 4: High-precision polish
    if time.time() < deadline - 5:
        points, score4 = high_precision_polish(points, max_iter=80)
        if score4 > score:
            score = score4
    
    return points
# EVOLVE-BLOCK-END
