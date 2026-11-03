import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Pair distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_dist_sq)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def objective_and_grad(x, beta=100.0):
        centers = x.reshape(N, 2)
        
        # 1. Wall distances (scaled by 2 to match pair distance scale)
        # v_wall = 2 * (0.5 - |c|)
        wall_dists = 2.0 * (0.5 - np.abs(centers))
        flat_walls = wall_dists.flatten()
        
        # 2. Pair distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        dists = np.sqrt(dist_sq)
        flat_pairs = dists.flatten()
        
        all_vals = np.concatenate([flat_walls, flat_pairs])
        
        # Soft min calculation
        v_min = np.min(all_vals)
        shifted = all_vals - v_min
        exp_terms = np.exp(-beta * shifted)
        S = np.sum(exp_terms)
        p = exp_terms / S
        
        val = -(v_min - (1.0/beta) * np.log(S))
        grad_all_vals = -p
        
        grad_centers = np.zeros_like(centers)
        
        # 1. Walls
        grad_walls = grad_all_vals[:2*N].reshape(N, 2)
        signs = np.sign(centers)
        # d(2*(0.5 - |c|))/dc = -2 * sign(c)
        grad_centers += grad_walls * (-2.0 * signs)
        
        # 2. Pairs
        grad_pairs_flat = grad_all_vals[2*N:]
        grad_pairs = grad_pairs_flat.reshape(N, N)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_diff = diff / dists[:, :, np.newaxis]
            unit_diff = np.nan_to_num(unit_diff, nan=0.0, posinf=0.0, neginf=0.0)
            
        grad_pairs_contrib = np.sum(grad_pairs * unit_diff, axis=1)
        grad_centers += grad_pairs_contrib
        
        return val, grad_centers.flatten()

    def hexagonal_init(rows, cols, jitter=0.0, seed=0):
        rng = np.random.RandomState(seed)
        # Adjust d to fit N points in the square
        # Approximate area per point
        d = 1.0 / cols * 0.99
        row_spacing = d * math.sqrt(3) / 2
        total_height = (rows - 1) * row_spacing + d
        if total_height > 1.0:
             # Scale down
             scale = 1.0 / total_height
             d *= scale
             row_spacing = d * math.sqrt(3) / 2

        pts = []
        for row in range(rows):
            y = -0.5 + d/2 + row * row_spacing
            if y > 0.5 - d/2: break # Safety
            
            if row % 2 == 0:
                x_start = -0.5 + d/2
                for col in range(cols):
                    x = x_start + col * d
                    if x > 0.5 - d/2: break
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for col in range(cols - 1):
                    x = x_start + col * d
                    if x > 0.5 - d/2: break
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        else:
            return None

    # --- Stage 1: Simulated Annealing ---
    best_centers_sa = None
    best_radius_sa = 0.0
    
    # Generate diverse initializations
    inits = []
    for rows in [14, 15, 16]:
        cols = math.ceil(N / rows)
        pts = hexagonal_init(rows, cols, jitter=0.1, seed=rows*10)
        if pts is not None:
            inits.append(pts)
            
    # Add a random init
    rng = np.random.RandomState(42)
    inits.append(rng.uniform(-0.49, 0.49, (N, 2)))

    for init in inits:
        if time.time() > deadline - 30: # Leave time for L-BFGS
            break
            
        current = init.copy()
        current_radius = compute_radius(current)
        
        if current_radius > best_radius_sa:
            best_radius_sa = current_radius
            best_centers_sa = current.copy()
            
        # Annealing parameters
        T = 0.01
        T_min = 0.0001
        cooling_rate = 0.995
        max_steps = 20000
        
        rng_sa = np.random.RandomState(123)
        
        for step in range(max_steps):
            if time.time() > deadline - 30:
                break
                
            # Perturbation
            idx = rng_sa.randint(N)
            current[idx] += rng_sa.normal(0, T, 2)
            
            # Clip to bounds (softly, let gradient handle mostly, but hard clip prevents invalid)
            np.clip(current, -0.5, 0.5, out=current)
            
            new_radius = compute_radius(current)
            
            # Accept if better or with probability exp((new-old)/T)
            if new_radius > current_radius:
                current_radius = new_radius
                if new_radius > best_radius_sa:
                    best_radius_sa = new_radius
                    best_centers_sa = current.copy()
            else:
                delta = new_radius - current_radius
                if delta < 0:
                    prob = math.exp(delta / T)
                    if rng_sa.random() < prob:
                        current_radius = new_radius
                    else:
                        # Reject: revert
                        # We need to revert the change. Since we modified in place, 
                        # we should have saved the old state or just re-perturb?
                        # Better: save old point before perturbation.
                        # Optimization: Only calculate new radius if we accept? No, we need it for decision.
                        # To save time, we can just revert by subtracting the noise?
                        # current[idx] -= noise? We didn't store noise.
                        # Let's just break this specific SA loop if it gets too slow?
                        # Actually, reverting requires storing the old coordinate.
                        pass # This implementation is slightly flawed for rejection without storing old state. 
                             # Let's fix: store old coord.
                             
        # Note: The above SA loop is a bit inefficient due to in-place modification and rejection logic.
        # Let's refine the SA loop to be robust.
        
    # --- Refined SA Loop for correctness and speed ---
    best_centers_sa = None
    best_radius_sa = 0.0
    
    for init in inits:
        if time.time() > deadline - 40:
            break
            
        current = init.copy()
        current_radius = compute_radius(current)
        
        if current_radius > best_radius_sa:
            best_radius_sa = current_radius
            best_centers_sa = current.copy()
            
        T = 0.02
        T_min = 0.0005
        cooling_rate = 0.99
        steps_per_temp = 50
        max_temps = 1000 # 50k steps max per init
        
        rng_sa = np.random.RandomState(456)
        
        for t_step in range(max_temps):
            if time.time() > deadline - 40:
                break
                
            for _ in range(steps_per_temp):
                idx = rng_sa.randint(N)
                old_x = current[idx].copy()
                noise = rng_sa.normal(0, T, 2)
                current[idx] = old_x + noise
                np.clip(current, -0.5, 0.5, out=current)
                
                new_radius = compute_radius(current)
                
                if new_radius > current_radius:
                    current_radius = new_radius
                    if new_radius > best_radius_sa:
                        best_radius_sa = new_radius
                        best_centers_sa = current.copy()
                else:
                    delta = new_radius - current_radius
                    if delta < 0:
                        prob = math.exp(delta / T)
                        if rng_sa.random() < prob:
                            current_radius = new_radius
                        else:
                            current[idx] = old_x # Revert
                            
            T *= cooling_rate
            if T < T_min:
                break

    # --- Stage 2: L-BFGS Polish ---
    best_centers = best_centers_sa
    best_radius = best_radius_sa
    
    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_radius = compute_radius(best_centers)

    # Polish the best result from SA
    if time.time() < deadline - 10:
        x0 = best_centers.flatten()
        
        def obj(x, beta=100.0):
            val, _ = objective_and_grad(x, beta)
            return val
            
        def obj_grad(x, beta=100.0):
            _, grad = objective_and_grad(x, beta)
            return grad

        try:
            res = minimize(
                obj, x0, method='L-BFGS-B', 
                jac=obj_grad, 
                options={'maxiter': 300, 'gtol': 1e-10, 'disp': False}
            )
            
            centers_opt = res.x.reshape(N, 2)
            r_opt = compute_radius(centers_opt)
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = centers_opt.copy()
                
        except Exception as e:
            pass

    # If we have time, try polishing a few other top candidates from SA? 
    # We only kept the single best. Let's try to keep top 2-3 from SA if we modified the loop.
    # For now, just return the polished best.
    
    return best_centers
