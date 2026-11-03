import numpy as np
import time

N = 200

def construct():
    # Parameters
    max_time = 110.0  # Leave 10s buffer
    start_time = time.time()
    
    # Binary search bounds for radius
    r_low = 0.030
    r_high = 0.0375
    best_P = None
    best_r = r_low
    
    # Helper: Relax a configuration for a given radius r
    def relax(P, r, max_iters=2000, alpha=0.01):
        """
        Minimize overlap energy for fixed r.
        Returns P and the minimum distance (pair and boundary).
        """
        P = P.copy()
        # Adaptive alpha
        current_alpha = alpha
        prev_energy = float('inf')
        
        for it in range(max_iters):
            # Compute pairwise distances
            # P is (N, 2)
            # diff is (N, N, 2)
            diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            
            # Mask diagonal
            np.fill_diagonal(dist, np.inf)
            
            # Forces
            forces = np.zeros_like(P)
            
            # Pairwise repulsion
            # Only consider pairs with dist < 2r
            mask = dist < 2 * r
            # Avoid division by zero (though dist is inf on diag, and we mask)
            # dist_safe = np.where(mask, dist, 1.0)
            # force_mag = (2*r - dist) / dist_safe
            # force_mag = np.where(mask, force_mag, 0)
            
            # Optimization: Compute only for masked elements?
            # For N=200, N^2=40000. It's small enough to compute all.
            dist_safe = np.where(dist > 1e-8, dist, 1.0)
            force_mag = (2 * r - dist) / dist_safe
            force_mag = np.where(mask, force_mag, 0.0)
            
            # Force vector for i from j is force_mag[i,j] * (P[i] - P[j]) / dist[i,j]
            # Note: diff[i,j] is P[i] - P[j]
            # So force on i is sum_j force_mag[i,j] * diff[i,j] / dist[i,j]
            # Wait, diff[i,j] is vector.
            # force_vec = (force_mag / dist_safe)[:, :, np.newaxis] * diff
            # Sum over j (axis=1)
            
            coeff = force_mag / dist_safe
            coeff = np.where(mask, coeff, 0.0)
            forces += np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary forces
            # Left: x < -0.5 + r
            # Right: x > 0.5 - r
            # Bottom: y < -0.5 + r
            # Top: y > 0.5 - r
            
            # Left
            mask_l = P[:, 0] < -0.5 + r
            if np.any(mask_l):
                forces[mask_l, 0] += (r - (P[mask_l, 0] - (-0.5))) # r - (x + 0.5) = r - x - 0.5
                # Correction: dist to left wall is x - (-0.5) = x + 0.5.
                # We want dist >= r. So if x + 0.5 < r, push right.
                # Force = (r - (x + 0.5)) in +x direction.
            
            # Right
            mask_r = P[:, 0] > 0.5 - r
            if np.any(mask_r):
                forces[mask_r, 0] -= (P[mask_r, 0] - (0.5 - r))
                # dist to right wall is 0.5 - x.
                # If 0.5 - x < r, push left.
                # Force = (r - (0.5 - x)) in -x direction.
                # Magnitude = r - 0.5 + x.
                
            # Bottom
            mask_b = P[:, 1] < -0.5 + r
            if np.any(mask_b):
                forces[mask_b, 1] += (r - (P[mask_b, 1] - (-0.5)))
            
            # Top
            mask_t = P[:, 1] > 0.5 - r
            if np.any(mask_t):
                forces[mask_t, 1] -= (P[mask_t, 1] - (0.5 - r))
            
            # Update positions
            P += alpha * forces
            
            # Clamp to square (hard constraint for stability, though forces should handle it)
            P[:, 0] = np.clip(P[:, 0], -0.5 + r, 0.5 - r)
            P[:, 1] = np.clip(P[:, 1], -0.5 + r, 0.5 - r)
            
            # Check convergence / energy
            # Calculate min dist
            min_pair = np.min(dist)
            min_bound = np.min([
                np.min(P[:, 0] + 0.5),
                np.min(0.5 - P[:, 0]),
                np.min(P[:, 1] + 0.5),
                np.min(0.5 - P[:, 1])
            ])
            current_min = min(min_pair, min_bound)
            
            # If we reached a valid state for this r, we can stop early?
            # But we want to maximize the margin.
            # If current_min >= 2*r (pair) and min_bound >= r, it's valid.
            # Actually, if we are relaxing for r, we want min_pair >= 2r and min_bound >= r.
            # If achieved, we can break.
            if min_pair >= 2 * r and min_bound >= r:
                break
                
            # Adaptive alpha: if energy increases, reduce alpha
            # We don't have a smooth energy easily, but we can check if min_dist improved.
            # Simple heuristic: if min_dist decreased significantly, reduce alpha.
            if it > 0 and current_min < prev_energy - 1e-5:
                alpha *= 0.9
            prev_energy = current_min
            
            # Time check
            if time.time() - start_time > max_time - 5:
                break
                
        return P, min_pair, min_bound

    def try_radius(r):
        """
        Try to find a valid configuration for radius r.
        Returns (success, P, score)
        """
        # Generate multiple initial configurations
        inits = []
        
        # 1. Random
        inits.append(np.random.uniform(-0.5, 0.5, (N, 2)))
        
        # 2. Jittered Hex Grid
        # 14 rows approx
        for m in [12, 14, 16]:
            base = N // m
            rem = N % m
            row_counts = [base + 1] * rem + [base] * (m - rem)
            n_max = max(row_counts)
            dx = 1.0 / n_max
            dy = dx * np.sqrt(3) / 2.0
            # Scale to fit
            # Adjust dx, dy to be slightly larger than r?
            # Just generate standard hex and jitter
            P_hex = []
            y_start = -((m - 1) * dy) / 2.0
            for j, n in enumerate(row_counts):
                y = y_start + j * dy
                x_start = -((n - 1) * dx) / 2.0
                for i in range(n):
                    x = x_start + i * dx
                    P_hex.append([x, y])
            P_hex = np.array(P_hex)
            # Jitter
            jitter = np.random.normal(0, dx * 0.1, P_hex.shape)
            inits.append(P_hex + jitter)
            
        # 3. Jittered Square Grid
        m_sq = int(np.sqrt(N))
        # Adjust to get exactly N
        # Try a few grid sizes
        for m_sq in [14, 15, 16]:
            P_sq = np.linspace(-0.5, 0.5, m_sq)
            grid_x, grid_y = np.meshgrid(P_sq, P_sq)
            P_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            if len(P_grid) > N:
                P_grid = P_grid[:N]
            elif len(P_grid) < N:
                # Pad with random
                pad = np.random.uniform(-0.5, 0.5, (N - len(P_grid), 2))
                P_grid = np.vstack([P_grid, pad])
            jitter = np.random.normal(0, 0.01, P_grid.shape)
            inits.append(P_grid + jitter)
            
        best_P = None
        best_score = -1
        
        for P_init in inits:
            if time.time() - start_time > max_time - 10:
                break
            P_relaxed, min_pair, min_bound = relax(P_init, r, max_iters=500, alpha=0.02)
            
            # Check validity
            # Score is min(min_pair/2, min_bound)
            # But we need min_pair >= 2r and min_bound >= r
            # The actual score is the max possible r for this P.
            score = min(min_pair / 2.0, min_bound)
            
            if score > best_score:
                best_score = score
                best_P = P_relaxed
                
            # If we found a valid packing for r, we can return early?
            # Yes, if score >= r, it's valid.
            if score >= r:
                return True, best_P, score
                
        return False, best_P, best_score

    # Binary Search
    # We want to find the max r.
    # Let's do a few iterations.
    # Instead of strict binary search which might miss good local minima,
    # we can do a linear sweep with refinement.
    
    # Initial sweep
    r_steps = np.linspace(0.034, 0.0365, 10)
    
    for r_test in r_steps:
        if time.time() - start_time > max_time - 20:
            break
        success, P, score = try_radius(r_test)
        if success:
            best_r = max(best_r, r_test)
            if score > best_r:
                best_r = score
                best_P = P
        else:
            # If failed, we might have a good P with score < r_test
            if score > best_r:
                best_r = score
                best_P = P
                
    # Refine around best_r
    if best_P is not None:
        # Try to push a bit higher
        for _ in range(5):
            r_try = best_r + 0.0001
            success, P, score = try_radius(r_try)
            if success:
                best_r = r_try
                best_P = P
            else:
                if score > best_r:
                    best_r = score
                    best_P = P
                else:
                    break
                    
    if best_P is None:
        # Fallback to simple grid if everything failed
        m = 14
        P = np.zeros((N, 2))
        # Just return a valid grid
        for i in range(N):
            row = i // 14
            col = i % 14
            P[i, 0] = -0.5 + 0.5 * (2 * col + 1) / 14
            P[i, 1] = -0.5 + 0.5 * (2 * row + 1) / 14
        return P
        
    return best_P
