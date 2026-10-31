import numpy as np
import time
import random
import math

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = -np.inf
    best_x = None
    
    # Objective: Maximize sum(r)
    # Constraints:
    # 1. 2*(W+H) <= 4  =>  H <= 2 - W
    # 2. Circles inside [0, W] x [0, H]
    # 3. No overlap
    
    def evaluate(x):
        """
        x: array of length 3*N + 1
           [W, x_1..x_N, y_1..y_N, r_1..r_N]
        Returns: (score, valid)
        Score is sum(r) if valid, else -infinity (or penalized for SA).
        """
        W = x[0]
        H = 2.0 - W
        
        if W <= 0.1 or H <= 0.1:
            return -np.inf, False
            
        xs = x[1:N+1]
        ys = x[N+1:2*N+1]
        rs = x[2*N+1:]
        
        # Check bounds
        # x in [r, W-r]
        if np.any(xs < rs - 1e-9) or np.any(xs > W - rs + 1e-9):
            return -np.inf, False
        if np.any(ys < rs - 1e-9) or np.any(ys > H - rs + 1e-9):
            return -np.inf, False
        
        # Check overlaps
        # Vectorized distance check is fast enough for N=21
        # dist_sq[i,j] < (r_i + r_j)^2
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist*min_dist - 1e-9:
                    return -np.inf, False
        
        return np.sum(rs), True

    def get_penalized_score(x):
        """
        For Simulated Annealing, we need a continuous score landscape.
        We use a penalty method: Score = sum(r) - penalty(overlap) - penalty(bounds)
        This allows the algorithm to "tunnel" through slightly invalid states.
        """
        W = x[0]
        H = 2.0 - W
        
        if W <= 0.1 or H <= 0.1:
            return -1e10
            
        xs = x[1:N+1]
        ys = x[N+1:2*N+1]
        rs = x[2*N+1:]
        
        score = np.sum(rs)
        
        # Bounds penalty
        # Soft constraint: x >= r, x <= W-r
        bound_pen = 0.0
        bound_pen += np.sum(np.maximum(0, rs - xs)**2)
        bound_pen += np.sum(np.maximum(0, xs - (W - rs))**2)
        bound_pen += np.sum(np.maximum(0, rs - ys)**2)
        bound_pen += np.sum(np.maximum(0, ys - (H - rs))**2)
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist*min_dist:
                    overlap = min_dist - np.sqrt(dist_sq)
                    overlap_pen += overlap**2
        
        # Weights for penalties
        w_bound = 100.0
        w_overlap = 100.0
        
        return score - w_bound * bound_pen - w_overlap * overlap_pen

    def random_initial(W_init):
        H_init = 2.0 - W_init
        x = np.zeros(3*N + 1)
        x[0] = W_init
        # Random positions
        for i in range(N):
            # Try to place without immediate overlap (greedy random init)
            placed = False
            for _ in range(50):
                r = random.uniform(0.05, 0.15)
                px = random.uniform(r, W_init - r)
                py = random.uniform(r, H_init - r)
                
                # Check against already placed
                ok = True
                for j in range(i):
                    dx = px - x[1+j]
                    dy = py - x[N+1+j]
                    dist_sq = dx*dx + dy*dy
                    min_d = r + x[2*N+1+j]
                    if dist_sq < min_d*min_d:
                        ok = False
                        break
                
                if ok:
                    x[1+i] = px
                    x[N+1+i] = py
                    x[2*N+1+i] = r
                    placed = True
                    break
            
            if not placed:
                # Fallback: small circle at random pos
                r = 0.05
                px = random.uniform(r, W_init - r)
                py = random.uniform(r, H_init - r)
                x[1+i] = px
                x[N+1+i] = py
                x[2*N+1+i] = r
                
        return x

    # Simulated Annealing
    T = 0.5
    T_min = 1e-6
    alpha = 0.9995
    
    # Start with a few different aspect ratios
    best_score_global = -np.inf
    best_x_global = None
    
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        x = random_initial(ar)
        current_score, _ = evaluate(x)
        
        # If initial is invalid, we still proceed with SA using penalized score
        # But we need a starting temperature based on the landscape
        # Let's run SA for a fixed number of steps or time budget
        
        x_sa = x.copy()
        score_sa, _ = evaluate(x_sa)
        
        # If the initial config is already valid, start T high enough to perturb
        # If invalid, T needs to be high to overcome penalties
        
        for step in range(50000):
            if time.time() > deadline - 5:
                break
            
            # Generate neighbor
            x_new = x_sa.copy()
            
            # Perturb a random subset of variables
            # 1. Aspect ratio
            if random.random() < 0.05:
                x_new[0] += random.gauss(0, 0.05)
                # Clamp W to keep H positive
                x_new[0] = np.clip(x_new[0], 0.2, 1.8)
                
            # 2. Positions and Radii
            # Perturb all or random few? For N=21, perturbing all is okay but might be too noisy.
            # Let's perturb a random circle's position and radius, and maybe W.
            
            idx = random.randint(0, N-1)
            # Position
            x_new[1+idx] += random.gauss(0, 0.05)
            x_new[N+1+idx] += random.gauss(0, 0.05)
            # Radius
            x_new[2*N+1+idx] += random.gauss(0, 0.01)
            
            # Ensure radii are positive
            x_new[2*N+1:] = np.maximum(x_new[2*N+1:], 0.01)
            
            new_score, new_valid = evaluate(x_new)
            
            # Use penalized score for acceptance if new is invalid but current is valid?
            # Standard SA: accept if better. If worse, accept with prob exp((new-old)/T).
            # But if new is invalid (-inf), we should use the penalized score to guide back.
            
            if new_valid:
                score_new = new_score
            else:
                score_new = get_penalized_score(x_new)
                
            if not (score_sa == -np.inf): # If current is valid
                score_cur = score_sa
            else:
                score_cur = get_penalized_score(x_sa)
                
            delta = score_new - score_cur
            
            if delta > 0:
                x_sa = x_new
                score_sa = new_score
                if new_valid and new_score > best_score_global:
                    best_score_global = new_score
                    best_x_global = x_new.copy()
            else:
                # Accept with probability
                if T > T_min:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        x_sa = x_new
                        score_sa = new_score
                        if new_valid and new_score > best_score_global:
                            best_score_global = new_score
                            best_x_global = x_new.copy()
            
            T *= alpha
            
            # If we get stuck in a valid local minimum, occasionally kick with higher T?
            # Or just let it cool.
            
    # Final refinement: Local search on the best found
    if best_x_global is not None:
        # Simple hill climbing to fix any tiny invalidities or squeeze more
        x_opt = best_x_global.copy()
        for _ in range(10000):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                # Try moving circle i slightly
                old_x = x_opt[1+i]
                old_y = x_opt[N+1+i]
                
                # Try small steps
                for dx in [-0.01, 0.01]:
                    for dy in [-0.01, 0.01]:
                        x_try = x_opt.copy()
                        x_try[1+i] = old_x + dx
                        x_try[N+1+i] = old_y + dy
                        
                        # Check if valid
                        score_try, valid_try = evaluate(x_try)
                        if valid_try and score_try > evaluate(x_opt)[0] + 1e-6:
                            x_opt = x_try
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try growing radii slightly
                for i in range(N):
                    old_r = x_opt[2*N+1+i]
                    x_try = x_opt.copy()
                    x_try[2*N+1+i] = old_r * 1.005
                    
                    score_try, valid_try = evaluate(x_try)
                    if valid_try:
                        x_opt = x_try
                        improved = True
                        break
                if not improved:
                    break
            
    # Return best
    if best_x_global is not None:
        # Extract circles
        W = best_x_global[0]
        H = 2.0 - W
        xs = best_x_global[1:N+1]
        ys = best_x_global[N+1:2*N+1]
        rs = best_x_global[2*N+1:]
        
        result = np.column_stack((xs, ys, rs))
        
        # Final sanity check
        if evaluate(result)[1]:
            return result
    
    # Fallback: Grid
    # 5x5 grid in 1x1 square (perimeter 4)
    W = 1.0
    H = 1.0
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append((x, y, r))
    return np.array(cells[:N])
