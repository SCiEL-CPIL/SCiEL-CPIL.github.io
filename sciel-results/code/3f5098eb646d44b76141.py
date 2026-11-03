import numpy as np
from scipy.optimize import minimize
import time

N = 21
START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s buffer

def _pairwise_constraints(x, y, r):
    """Compute pairwise distance violations (positive = overlap)."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    rad_sum = r[:, None] + r[None, :]
    return np.maximum(0, rad_sum - dist)

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
        overlap_viol = np.max(_pairwise_constraints(x, y, r))
            
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
        
        # 2. Expand radii greedily with adaptive step
        step = 0.01
        for _ in range(200):
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
                step *= 0.85
                if step < 1e-7:
                    break
        return x, y, r

    def gradient_optimize(x, y, r, W, H, max_iters=1000):
        """
        Use gradient-based optimization to maximize sum of radii while maintaining constraints.
        """
        def objective(z):
            """Minimize negative sum of radii (i.e., maximize sum of radii) plus penalty for violations."""
            xz = z[:N]
            yz = z[N:2*N]
            rz = z[2*N:]
            
            # Boundary violations
            min_x = np.min(xz - rz)
            max_x = np.max(xz + rz)
            min_y = np.min(yz - rz)
            max_y = np.max(yz + rz)
            boundary_viol = max(0, -min_x, max_x - W, -min_y, max_y - H)
            
            # Overlap violations
            overlap_viol = np.max(_pairwise_constraints(xz, yz, rz))
            
            total_viol = boundary_viol + overlap_viol
            
            # Penalize violations heavily
            penalty = 100 * total_viol + 10 * total_viol**2
            return -np.sum(rz) + penalty
        
        def gradient(z):
            """Numerical gradient (simple finite differences)."""
            eps = 1e-7
            grad = np.zeros_like(z)
            for i in range(len(z)):
                z_plus = z.copy()
                z_plus[i] += eps
                grad[i] = (objective(z_plus) - objective(z)) / eps
            return grad
        
        z = np.concatenate([x, y, r])
        
        for _ in range(max_iters):
            if time_remaining() < 5:
                break
            
            # Simple gradient descent with adaptive step
            grad = gradient(z)
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-8:
                break
            
            # Adaptive step size
            step_size = 0.01 / (1 + norm_grad)
            z_new = z - step_size * grad
            
            # Clip to reasonable bounds
            z_new[:N] = np.clip(z_new[:N], 0, W)
            z_new[N:2*N] = np.clip(z_new[N:2*N], 0, H)
            z_new[2*N:] = np.clip(z_new[2*N:], 0, 0.5)
            
            # Check if improvement
            if objective(z_new) < objective(z):
                z = z_new
            else:
                # Try smaller step
                step_size *= 0.5
                z_new = z - step_size * grad
                z_new[:N] = np.clip(z_new[:N], 0, W)
                z_new[N:2*N] = np.clip(z_new[N:2*N], 0, H)
                z_new[2*N:] = np.clip(z_new[2*N:], 0, 0.5)
                if objective(z_new) < objective(z):
                    z = z_new
                else:
                    break
        
        x_opt = z[:N]
        y_opt = z[N:2*N]
        r_opt = z[2*N:]
        
        # Final relaxation and expansion
        x_opt, y_opt, r_opt = relax_and_expand(x_opt, y_opt, r_opt, W, H, max_iters=100)
        
        return x_opt, y_opt, r_opt

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
        r = np.full(N, min(cell_w, cell_h) * 0.35)
        return x, y, r

    def random_init(W, H, seed=0):
        """
        Random initialization with non-overlapping guarantee via rejection sampling.
        """
        np.random.seed(seed)
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Start with a reasonable base radius
        base_r = min(W, H) / 10
        
        for i in range(N):
            attempts = 0
            while attempts < 1000:
                # Try random position
                r_i = base_r * (0.8 + 0.4 * np.random.random())
                x_i = np.random.uniform(r_i, W - r_i)
                y_i = np.random.uniform(r_i, H - r_i)
                
                # Check overlap with existing circles
                valid = True
                for j in range(i):
                    dist = np.sqrt((x_i - x[j])**2 + (y_i - y[j])**2)
                    if dist < r_i + r[j]:
                        valid = False
                        break
                
                if valid:
                    x[i] = x_i
                    y[i] = y_i
                    r[i] = r_i
                    break
                attempts += 1
            
            # If no valid position found, use a fallback
            if attempts >= 1000:
                # Place in a corner-ish position with small radius
                r_i = base_r * 0.5
                x[i] = r_i + 0.01
                y[i] = r_i + 0.01
                r[i] = r_i
        
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
        
        # Try multiple initializations
        best_local_score = 0
        best_local_x, best_local_y, best_local_r = None, None, None
        
        # Strategy 1: Grid init
        x, y, r = grid_init(W, H)
        x, y, r = relax_and_expand(x, y, r, W, H, max_iters=200)
        is_valid, viol = check_validity(x, y, r, W, H)
        if is_valid:
            score = np.sum(r)
            if score > best_local_score:
                best_local_score = score
                best_local_x, best_local_y, best_local_r = x.copy(), y.copy(), r.copy()
        
        # Strategy 2: Random init
        x, y, r = random_init(W, H, seed=int(W*100))
        x, y, r = relax_and_expand(x, y, r, W, H, max_iters=200)
        is_valid, viol = check_validity(x, y, r, W, H)
        if is_valid:
            score = np.sum(r)
            if score > best_local_score:
                best_local_score = score
                best_local_x, best_local_y, best_local_r = x.copy(), y.copy(), r.copy()
        
        # If we have a valid solution, try gradient optimization
        if best_local_x is not None and time_remaining() > 20:
            x_opt, y_opt, r_opt = gradient_optimize(best_local_x, best_local_y, best_local_r, W, H, max_iters=500)
            is_valid, viol = check_validity(x_opt, y_opt, r_opt, W, H)
            if is_valid:
                score = np.sum(r_opt)
                if score > best_local_score:
                    best_local_score = score
                    best_local_x, best_local_y, best_local_r = x_opt.copy(), y_opt.copy(), r_opt.copy()
        
        if best_local_score > best_global_score:
            best_global_score = best_local_score
            best_global_x = best_local_x.copy()
            best_global_y = best_local_y.copy()
            best_global_r = best_local_r.copy()
        
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
