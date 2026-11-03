import numpy as np
from scipy.optimize import minimize
import time

N = 21
START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s buffer

def construct():
    def time_remaining():
        return TIME_LIMIT - (time.time() - START_TIME)

    def check_validity(x, y, r, W, H):
        """
        Check if configuration is valid.
        Returns (is_valid, violation_value).
        violation_value is the max amount by which constraints are violated.
        """
        if W <= 0 or H <= 0:
            return False, 1e10
        if abs(2*(W+H) - 4) > 1e-4:
            return False, 1e10
        
        # Boundary checks
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        boundary_viol = max(0, -min_x, max_x - W, -min_y, max_y - H)
        
        # Overlap checks
        # Use numpy broadcasting for pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        rad_sum_sq = (r[:, None] + r[None, :])**2
        
        # We want dist_sq >= rad_sum_sq
        # violation is max(0, rad_sum_sq - dist_sq)
        # Exclude diagonal
        np.fill_diagonal(dist_sq, 1e10)
        np.fill_diagonal(rad_sum_sq, 0)
        
        overlap_viol = np.max(rad_sum_sq - dist_sq)
        if overlap_viol < 0:
            overlap_viol = 0
            
        total_viol = boundary_viol + overlap_viol
        return total_viol < 1e-6, total_viol

    def relax_and_expand(x, y, r, W, H, max_iters=500):
        """
        Iteratively resolve overlaps and expand radii.
        """
        # 1. Resolve overlaps by pushing circles apart
        for _ in range(max_iters):
            if time_remaining() < 5:
                break
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx**2 + dy**2
                    rad_sum = r[i] + r[j]
                    if dist_sq < rad_sum**2:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-10 else 1e-10
                        # Push apart
                        overlap = rad_sum - dist
                        if overlap > 0:
                            # Normalize direction
                            nx = dx / dist
                            ny = dy / dist
                            # Move each by half the overlap (or weighted by inverse radius if we cared, but equal is fine for start)
                            shift = 0.5 * overlap
                            x[i] += nx * shift
                            x[j] -= nx * shift
                            y[i] += ny * shift
                            y[j] -= ny * shift
                            moved = True
            if not moved:
                break
        
        # 2. Expand radii greedily
        step = 0.005
        for _ in range(100):
            if time_remaining() < 5:
                break
            improved = False
            for i in range(N):
                new_r = r[i] + step
                if x[i] - new_r < 0 or x[i] + new_r > W:
                    continue
                if y[i] - new_r < 0 or y[i] + new_r > H:
                    continue
                valid = True
                for j in range(N):
                    if i == j: 
                        continue
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    rad_sum = new_r + r[j]
                    if dist_sq < rad_sum**2:
                        valid = False
                        break
                if valid:
                    r[i] = new_r
                    improved = True
            if not improved:
                step *= 0.9
                if step < 1e-6:
                    break
        return x, y, r

    def local_search(x, y, r, W, H, iterations=500):
        """
        Simple local search: perturb positions, re-relax, re-expand.
        """
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        temp = 0.05
        for _ in range(iterations):
            if time_remaining() < 5:
                break
            x_new = x.copy()
            y_new = y.copy()
            # Perturb
            noise = np.random.normal(0, temp, N)
            x_new += noise
            y_new += np.random.normal(0, temp, N)
            
            # Keep radii for now, try to relax and expand
            r_new = r.copy()
            x_new, y_new, r_new = relax_and_expand(x_new, y_new, r_new, W, H, max_iters=50)
            
            # Check validity
            is_valid, viol = check_validity(x_new, y_new, r_new, W, H)
            if is_valid:
                score = np.sum(r_new)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_new.copy()
                    x, y, r = best_x, best_y, best_r
                    temp = max(0.01, temp * 0.95)
                else:
                    # Metropolis acceptance? Or just hill climb?
                    # Hill climb is safer for packing.
                    pass
            else:
                # If invalid, we can still accept if violation is small and score is high?
                # For simplicity, stick to valid solutions only.
                pass
        
        return best_x, best_y, best_r, best_score

    def grid_init(W, H):
        """
        Initialize circles in a grid pattern scaled to W, H.
        """
        # Approximate square root grid
        side = int(np.ceil(np.sqrt(N)))
        # Create grid points
        x_coords = []
        y_coords = []
        for i in range(N):
            row = i // side
            col = i % side
            # Center in cell
            x = (col + 0.5) * (W / side)
            y = (row + 0.5) * (H / side)
            x_coords.append(x)
            y_coords.append(y)
        
        x = np.array(x_coords)
        y = np.array(y_coords)
        # Initial radius: small fraction of cell size
        cell_w = W / side
        cell_h = H / side
        r = np.full(N, min(cell_w, cell_h) * 0.4)
        return x, y, r

    best_global_score = 0
    best_global_x = None
    best_global_y = None
    best_global_r = None
    
    # Try various aspect ratios
    # Perimeter 2(W+H)=4 => W+H=2
    # W in [0.5, 1.5]
    W_values = np.linspace(0.6, 1.4, 9)
    
    np.random.seed(42)
    
    for W in W_values:
        H = 2.0 - W
        if H <= 0.1: continue
        
        # Initialize
        x, y, r = grid_init(W, H)
        
        # Relax and expand
        x, y, r = relax_and_expand(x, y, r, W, H, max_iters=200)
        
        # Check validity
        is_valid, viol = check_validity(x, y, r, W, H)
        if not is_valid:
            # Try to fix with more relaxation
            x, y, r = relax_and_expand(x, y, r, W, H, max_iters=500)
            is_valid, viol = check_validity(x, y, r, W, H)
            if not is_valid:
                continue
        
        score = np.sum(r)
        
        # Local search
        x, y, r, score = local_search(x, y, r, W, H, iterations=100)
        
        if score > best_global_score:
            best_global_score = score
            best_global_x = x.copy()
            best_global_y = y.copy()
            best_global_r = r.copy()
            # print(f"New best: W={W:.2f}, H={H:.2f}, Score={score:.4f}")
            
        if time_remaining() < 10:
            break

    # If we didn't find anything valid (shouldn't happen), fallback
    if best_global_x is None:
        W, H = 1.0, 1.0
        x, y, r = grid_init(W, H)
        x, y, r = relax_and_expand(x, y, r, W, H)
        # Force valid by shrinking if needed
        is_valid, viol = check_validity(x, y, r, W, H)
        if not is_valid:
             # Shrink radii until valid
             for _ in range(100):
                 r *= 0.95
                 is_valid, viol = check_validity(x, y, r, W, H)
                 if is_valid: break
        best_global_x = x
        best_global_y = y
        best_global_r = r

    result = np.zeros((N, 3))
    result[:, 0] = best_global_x
    result[:, 1] = best_global_y
    result[:, 2] = best_global_r
    
    return result
