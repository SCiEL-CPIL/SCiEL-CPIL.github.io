"""Circle packing in unit square using vectorized parallel simulated annealing."""
import numpy as np
import time

N = 26

def _is_valid(circles, eps=1e-9):
    """Check if circles are strictly inside [0,1]^2 and disjoint."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Wall check
    if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    if np.any(r < -eps):
        return False
        
    n = len(circles)
    # Pairwise distance check
    # Vectorized: compute all pairwise distances
    # d[i, j] = dist(i, j)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    sum_r = r[:, None] + r[None, :]
    
    # Check if dist[i, j] < r[i] + r[j] for i != j
    # We can check the minimum of (dist - sum_r) for i != j
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    vals = dist[mask] - sum_r[mask]
    
    if np.any(vals < -eps):
        return False
    return True

def _objective(circles):
    """
    Objective function for SA.
    We want to maximize sum(r).
    We penalize overlaps and wall violations.
    Score = sum(r) - penalty.
    """
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Wall penalty
    # Distance to walls: min(x, 1-x, y, 1-y)
    # If r > dist_to_wall, penalty is (r - dist_to_wall)
    dist_to_wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
    wall_viol = np.maximum(0.0, r - dist_to_wall)
    
    # Overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    sum_r = r[:, None] + r[None, :]
    
    # Overlap amount: max(0, r_i + r_j - dist_ij)
    # Avoid diagonal
    mask = ~np.eye(len(circles), dtype=bool)
    overlap_viol = np.maximum(0.0, sum_r - dist)
    overlap_viol = overlap_viol[mask]
    
    total_viol = np.sum(wall_viol) + np.sum(overlap_viol)
    
    # Penalty weight needs to be large enough to enforce constraints
    # But not so large that it traps in local minima with zero radius.
    # A common strategy is adaptive or large constant.
    # Let's use a large constant penalty.
    penalty_weight = 10.0
    penalty = penalty_weight * total_viol
    
    score = np.sum(r) - penalty
    return score

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    best_circles = None
    best_score = -np.inf
    
    # Parallel SA
    n_pop = 4
    pop_size = 50  # 200 circles total, vectorized
    
    # Initialize population
    # Mix of grid and random
    all_circles = np.zeros((n_pop * pop_size, 3))
    
    for k in range(n_pop * pop_size):
        if k % 2 == 0:
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            idx = (k // 2) % (side * side)
            i = idx // side
            j = idx % side
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            # Add some noise
            cx += np.random.uniform(-0.05, 0.05)
            cy += np.random.uniform(-0.05, 0.05)
            r = max(0.01, r + np.random.uniform(-0.01, 0.01))
        else:
            # Random init
            cx = np.random.uniform(0.1, 0.9)
            cy = np.random.uniform(0.1, 0.9)
            r = np.random.uniform(0.02, 0.08)
        
        all_circles[k, 0] = cx
        all_circles[k, 1] = cy
        all_circles[k, 2] = r
    
    # Reshape to (n_pop, pop_size, 3)
    pop = all_circles.reshape(n_pop, pop_size, 3)
    
    # SA Parameters
    T_start = 0.1
    T_end = 0.0001
    cooling_rate = 0.999
    
    # Precompute masks for vectorization? 
    # It's hard to vectorize across the population for pairwise distances easily without huge memory.
    # pop_size=50, N=26. 
    # For each of 200 circles, we need 25x25 distance matrix.
    # 200 * 25 * 25 = 125,000 operations per step. This is fast in NumPy.
    
    t_start = time.time()
    step = 0
    
    # We will run SA steps until deadline
    while time.time() < deadline - 5:
        T = T_start * (T_end / T_start) ** (step / 1000.0)
        if T < T_end:
            break
            
        # 1. Generate neighbors
        # Perturb positions and radii
        # Gaussian noise
        noise = np.random.normal(0, 0.01, pop.shape)
        # Scale noise by temperature? Or fixed step size?
        # Fixed step size is often better for geometric problems.
        # Let's use a step size that decreases slightly or stays constant.
        step_size = 0.02
        new_pop = pop + noise * step_size
        
        # Clamp to valid ranges (loose)
        new_pop[:, :, 0] = np.clip(new_pop[:, :, 0], 0.0, 1.0)
        new_pop[:, :, 1] = np.clip(new_pop[:, :, 1], 0.0, 1.0)
        new_pop[:, :, 2] = np.clip(new_pop[:, :, 2], 0.0, 0.5)
        
        # 2. Evaluate objective
        # We need to evaluate objective for each of the 200 circles.
        # This is a loop over the population, but vectorized within each circle.
        # Or we can vectorize the whole thing.
        
        # Let's try vectorizing the evaluation for the whole population at once.
        # Shape: (n_pop * pop_size, N, 3)
        flat_new = new_pop.reshape(-1, N, 3)
        
        x = flat_new[:, :, 0]
        y = flat_new[:, :, 1]
        r = flat_new[:, :, 2]
        
        # Wall violation
        dist_to_wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        wall_viol = np.maximum(0.0, r - dist_to_wall)
        
        # Pairwise distances
        # x shape: (M, N)
        dx = x[:, :, None] - x[:, None, :]  # (M, N, N)
        dy = y[:, :, None] - y[:, None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        
        sum_r = r[:, :, None] + r[:, None, :]
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        # Apply mask
        # overlap_viol = np.maximum(0.0, sum_r - dist)
        # We only care about i < j or i != j
        # Sum over all i != j
        overlap_viol = np.maximum(0.0, sum_r - dist)
        # Zero out diagonal
        overlap_viol = overlap_viol * mask  # mask is broadcastable?
        # mask shape (N, N), overlap_viol shape (M, N, N). 
        # Need to reshape mask to (1, N, N)
        overlap_viol = overlap_viol * mask[None, :, :]
        
        total_viol = np.sum(wall_viol, axis=1) + np.sum(overlap_viol, axis=(1, 2))
        
        penalty_weight = 10.0
        penalty = penalty_weight * total_viol
        
        new_scores = np.sum(r, axis=1) - penalty
        
        # 3. Acceptance
        # Current scores
        flat_cur = pop.reshape(-1, N, 3)
        xc = flat_cur[:, :, 0]
        yc = flat_cur[:, :, 1]
        rc = flat_cur[:, :, 2]
        
        dwc = np.minimum(np.minimum(xc, 1.0-xc), np.minimum(yc, 1.0-yc))
        wvc = np.maximum(0.0, rc - dwc)
        
        dxc = xc[:, :, None] - xc[:, None, :]
        dyc = yc[:, :, None] - yc[:, None, :]
        distc = np.sqrt(dxc*dxc + dyc*dyc)
        src = rc[:, :, None] + rc[:, None, :]
        
        ovc = np.maximum(0.0, src - distc)
        ovc = ovc * mask[None, :, :]
        
        total_viol_c = np.sum(wvc, axis=1) + np.sum(ovc, axis=(1, 2))
        penalty_c = penalty_weight * total_viol_c
        cur_scores = np.sum(rc, axis=1) - penalty_c
        
        # Delta E = new_score - cur_score
        delta_e = new_scores - cur_scores
        
        # Accept if delta_e > 0
        accept = delta_e > 0
        
        # Accept if delta_e < 0 with probability exp(delta_e / T)
        # Only for those not already accepted
        not_accepted = ~accept
        # To avoid log(0) or exp(large negative), we can just compute where needed
        # But vectorized is fine.
        # exp(delta_e / T)
        # If T is very small, delta_e / T is very negative -> 0.
        # If delta_e is positive, it's already accepted.
        # So we only care about negative delta_e.
        
        # Calculate prob for negative deltas
        # To avoid overflow/underflow issues in exp, we can clip or just let it go to 0.
        # np.exp handles -inf -> 0.
        
        # We need to be careful with T=0. But T > T_end > 0.
        prob = np.exp(delta_e[not_accepted] / T)
        rand = np.random.rand(np.sum(not_accepted))
        
        # Update accept mask
        # This part is tricky to vectorize cleanly with dynamic indexing.
        # Let's do it in a loop over the population for simplicity, it's only 200 items.
        # Or use np.where
        
        # Actually, let's just loop over the 200 circles for acceptance.
        # It's fast enough.
        new_flat = flat_new.copy()
        for i in range(len(flat_new)):
            if not accept[i]:
                if rand[i] < prob[i]:
                    accept[i] = True
        
        # Apply acceptance
        # If accepted, take new, else keep old
        # We can do this with np.where
        # But we need to map back to pop shape.
        
        # Let's just reconstruct
        final_flat = np.where(accept[:, None, None], new_flat, flat_cur)
        new_pop = final_flat.reshape(n_pop, pop_size, 3)
        
        pop = new_pop
        
        # Track best
        # We need to find the single best circle in the population
        # We already have new_scores (which are scores for new_flat)
        # But we need to compare with global best.
        # Actually, cur_scores and new_scores are for the flattened arrays.
        # We need to pick the best circle from the current population state.
        
        # Let's re-evaluate the current population to find the best one?
        # No, we have the scores for the proposed moves.
        # The current population state is a mix of old and new.
        # We should just evaluate the current population's best.
        
        # Optimization: 
        # Instead of re-evaluating everything, we can just keep the best circle found so far.
        # But we need to know which circle in the current `pop` is the best.
        # The scores we have (`new_scores` and `cur_scores`) are for the *proposed* and *original* states.
        # After acceptance, the state is mixed.
        # To be safe, let's evaluate the best candidate from the union of old and new?
        # Or just evaluate the current `pop` every K steps?
        
        # Let's evaluate the current `pop` every 10 steps to find the global best.
        if step % 10 == 0:
            # Flatten pop
            cur_flat = pop.reshape(-1, N, 3)
            # Reuse the evaluation logic
            x_c = cur_flat[:, :, 0]
            y_c = cur_flat[:, :, 1]
            r_c = cur_flat[:, :, 2]
            
            dw_c = np.minimum(np.minimum(x_c, 1.0-x_c), np.minimum(y_c, 1.0-y_c))
            wv_c = np.maximum(0.0, r_c - dw_c)
            
            dx_c = x_c[:, :, None] - x_c[:, None, :]
            dy_c = y_c[:, :, None] - y_c[:, None, :]
            dist_c = np.sqrt(dx_c*dx_c + dy_c*dy_c)
            sr_c = r_c[:, :, None] + r_c[:, None, :]
            
            ov_c = np.maximum(0.0, sr_c - dist_c)
            ov_c = ov_c * mask[None, :, :]
            
            tv_c = np.sum(wv_c, axis=1) + np.sum(ov_c, axis=(1, 2))
            pen_c = penalty_weight * tv_c
            scores_c = np.sum(r_c, axis=1) - pen_c
            
            best_idx = np.argmax(scores_c)
            best_circ = cur_flat[best_idx]
            
            # Check validity of the best candidate
            if _is_valid(best_circ, eps=1e-7):
                val = np.sum(best_circ[:, 2])
                if val > best_score:
                    best_score = val
                    best_circles = best_circ.copy()
        
        step += 1
        
        # Cooling
        # T = T_start * (T_end / T_start) ** (step / 1000.0) is handled in next loop iteration
        
    # Final check
    if best_circles is None:
        # Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_circles = np.array(cells)
        
    return best_circles
