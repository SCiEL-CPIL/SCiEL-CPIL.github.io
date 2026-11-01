import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def is_valid(xy, r, tol=1e-8):
        n = len(r)
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def get_score(xy, r):
        return np.sum(r)

    def force_directed_relax(xy, r, max_iters=500, step_size=0.01):
        """
        Attempts to resolve overlaps by moving circles.
        Returns (xy_new, overlap_sum).
        """
        xy = xy.copy()
        n = len(r)
        # Precompute indices
        iu = np.triu_indices(n, k=1)
        
        for it in range(max_iters):
            # Calculate forces
            forces = np.zeros((n, 2))
            overlap_sum = 0.0
            
            # Vectorized overlap calculation
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            
            # Only consider upper triangle for forces, then symmetrize
            # dist[iu] and min_dist[iu]
            d_iu = dist[iu]
            m_iu = min_dist[iu]
            overlaps = np.maximum(0, m_iu - d_iu)
            overlap_sum = np.sum(overlaps)
            
            if overlap_sum < 1e-10:
                break
            
            # Apply forces
            # For each pair (i, j) with overlap, push i and j apart
            # dx[i,j] = x[i] - x[j]
            # Force on i should be in direction (x[i]-x[j]) / dist
            # Force on j should be in direction (x[j]-x[i]) / dist
            
            # We need to accumulate forces
            # Using sparse logic or loops is safer for force direction
            # But vectorized is faster.
            # Let's use a loop for clarity and safety with N=32 (small)
            # Actually, N=32 is small, a double loop is fast enough in Python if optimized? 
            # 32*31/2 = 496 pairs. 500 iters * 496 pairs = 248k ops. Very fast.
            
            for k in range(len(iu[0])):
                i = iu[0][k]
                j = iu[1][k]
                if overlaps[k] > 0:
                    d = d_iu[k]
                    if d < 1e-10:
                        # Random push
                        angle = np.random.uniform(0, 2*np.pi)
                        fx = np.cos(angle)
                        fy = np.sin(angle)
                    else:
                        fx = dx[i, j] / d
                        fy = dy[i, j] / d
                    
                    # Proportional to overlap
                    force_mag = overlaps[k]
                    
                    forces[i, 0] += fx * force_mag
                    forces[i, 1] += fy * force_mag
                    forces[j, 0] -= fx * force_mag
                    forces[j, 1] -= fy * force_mag
            
            # Update positions
            xy += step_size * forces
            
            # Clamp to boundaries
            # If a circle is pushed out, clamp it. This might cause new overlaps, but it's a heuristic.
            xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
            
            # Adaptive step size?
            # If overlap increased, reduce step size.
            # Simple approach: fixed step size is often sufficient for this small N.
            
        return xy, overlap_sum

    def try_find_valid_config(R, rng, num_restarts=50):
        """
        Tries to find a valid configuration for N circles of radius R.
        Returns (xy, r) if found, else None.
        """
        r = np.full(N, R)
        best_xy = None
        best_overlap = np.inf
        
        for restart in range(num_restarts):
            # Initialization strategies
            if restart % 3 == 0:
                # Grid init
                side = 6
                xy = np.zeros((N, 2))
                for i in range(N):
                    row = i // side
                    col = i % side
                    xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            elif restart % 3 == 1:
                # Random init
                xy = rng.uniform(R, 1-R, size=(N, 2))
            else:
                # Hex-like init
                rows = 4
                cols = 8
                xy = np.zeros((N, 2))
                for i in range(N):
                    row = i // cols
                    col = i % cols
                    offset = 0.5/cols if row % 2 == 1 else 0
                    xy[i] = [(col + 0.5)/cols + offset, (row + 0.5)/rows]
                    # Clamp
                    xy[i, 0] = np.clip(xy[i, 0], R, 1-R)
                    xy[i, 1] = np.clip(xy[i, 1], R, 1-R)
            
            # Relax
            xy_relaxed, overlap = force_directed_relax(xy, r, max_iters=200, step_size=0.05)
            
            if overlap < best_overlap:
                best_overlap = overlap
                best_xy = xy_relaxed.copy()
            
            if overlap < 1e-8:
                # Verify validity
                if is_valid(best_xy, r):
                    return best_xy, r
        
        return None

    # Binary search for maximum R
    R_low = 0.05
    R_high = 0.12
    best_R = R_low
    best_config = None
    
    rng = np.random.RandomState(42)
    
    # Initial check to set R_high lower if necessary
    # We assume 0.12 is too high, 0.05 is too low.
    
    while time.time() < deadline - 5:
        R_mid = (R_low + R_high) / 2
        
        config = try_find_valid_config(R_mid, rng, num_restarts=20)
        
        if config is not None:
            xy, r = config
            if is_valid(xy, r):
                best_R = R_mid
                best_config = (xy, r)
                R_low = R_mid
            else:
                R_high = R_mid
        else:
            R_high = R_mid
        
        # If range is small enough, break
        if R_high - R_low < 1e-5:
            break
            
    if best_config is not None:
        xy, r = best_config
        # Try to improve by allowing radii to vary slightly?
        # The task is to maximize sum of radii.
        # If we found a valid config for R, we can try to increase individual radii.
        # Let's do a local search on radii now.
        
        def penalty_function(params):
            n = N
            xy = params[:2*n].reshape(n, 2)
            r = params[2*n:]
            
            d_left = xy[:, 0] - r
            d_right = 1 - xy[:, 0] - r
            d_bottom = xy[:, 1] - r
            d_top = 1 - xy[:, 1] - r
            boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                            np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)
            
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            overlap = np.maximum(0, min_dist[iu] - dist[iu])
            overlap_viol = np.sum(overlap**2)
            
            return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

        params = np.concatenate([xy.flatten(), r])
        
        # Refine with penalty method from the valid starting point
        # This combines the structural change (finding a valid core) with the refinement
        try:
            result = minimize(penalty_function, params, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12})
            xy_new = result.x[:2*N].reshape(N, 2)
            r_new = result.x[2*N:]
            
            if is_valid(xy_new, r_new):
                score_new = get_score(xy_new, r_new)
                score_cur = get_score(xy, r)
                if score_new > score_cur:
                    xy, r = xy_new, r_new
        except Exception:
            pass
        
        return np.column_stack([xy, r])
    
    # Fallback
    side = 6
    cells = []
    for i in range(N):
        row = i // side
        col = i % side
        cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3])
    return np.array(cells)
