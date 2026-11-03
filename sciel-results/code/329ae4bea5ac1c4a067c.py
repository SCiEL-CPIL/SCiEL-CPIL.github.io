"""Pack 50 equal circles in a unit square using hard-disk expansion."""
import numpy as np
import time

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    # Wall radius = min of (0.5 - |coord|)
    # Ensure centers are strictly inside for valid radius calculation
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    wall_r = np.min(0.5 - np.abs(centers))
    
    # Pairwise distances
    # Using broadcasting for vectorized distance calculation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(wall_r, min_pair / 2.0)

def _project_to_square(centers):
    """Project centers to stay within [-0.5, 0.5]^2."""
    centers = np.clip(centers, -0.5, 0.5)
    return centers

def _expand_radius(centers, r, max_iters=1000, time_limit=5.0):
    """
    Attempt to increase radius r by resolving overlaps.
    Uses a damped Newton method on the active set of constraints.
    Returns (new_centers, new_r, success)
    """
    start_time = time.time()
    centers = centers.copy()
    
    # We will try to increase r in small steps
    # The "hard disk" idea: assume r is fixed, move centers to eliminate overlaps.
    # If we can eliminate overlaps at r, then r is feasible.
    # We increase r until we can no longer eliminate overlaps (or time runs out).
    
    # Actually, a simpler approach for the inner loop:
    # Given current centers and target r, find a local minimum of overlap.
    # If overlap == 0, r is feasible.
    
    # Let's implement a specific solver for a fixed target radius r_target.
    # Minimize sum of overlaps.
    # Overlap_ij = max(0, 2*r - dist_ij)
    # Wall overlap_i = max(0, 2*r - (0.5 - |c_ix|), 2*r - (0.5 + c_ix), ...)
    
    # To make it differentiable, use soft overlap:
    # o_ij = max(0, 2*r - dist_ij)^2
    # But we want to push them apart.
    
    # Let's use a "repulsion" force model.
    # Force on i from j if dist < 2r:
    # F = k * (2r - dist) * (c_i - c_j) / dist
    
    # This is essentially a molecular dynamics simulation or gradient descent on potential energy.
    # Let's use Gradient Descent with adaptive step size for a fixed r.
    
    def _overlap_gradient(centers, r):
        """Compute gradient of sum of squared overlaps."""
        grad = np.zeros_like(centers)
        
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1e-10) # avoid div by zero
        
        # Overlap amount: positive if dist < 2r
        overlap = 2*r - dist
        # Only consider overlaps
        mask = overlap > 0
        # Gradient of (2r - dist)^2 w.r.t c_i
        # d/dc_i (2r - ||c_i - c_j||)^2 = 2*(2r - d) * (- (c_i - c_j)/d)
        # Sum over j for each i.
        
        # Vectorized calculation
        # factor[i,j] = 2 * (2r - dist[i,j]) * mask[i,j] / dist[i,j]
        # grad[i] = - sum_j factor[i,j] * (c_i - c_j)
        # Note: (c_i - c_j) is the vector from j to i.
        # Force should push i away from j.
        # If c_i > c_j, c_i - c_j is positive. We want to increase c_i.
        # So grad should point in direction of (c_i - c_j).
        # Let's check:
        # E = (2r - d)^2
        # dE/dc_i = 2(2r-d) * d(2r-d)/dc_i = 2(2r-d) * (-(c_i-c_j)/d)
        # So gradient is in direction -(c_i-c_j).
        # We want to MINIMIZE energy, so we move in direction -grad = +(c_i-c_j).
        # This pushes i away from j. Correct.
        
        factors = np.zeros_like(dist)
        factors[mask] = 2.0 * overlap[mask] / dist[mask]
        
        # grad[i] = - sum_j factors[i,j] * (c_i - c_j)
        # Wait, the gradient of the energy is what we computed.
        # We want to move in direction -grad.
        # -grad[i] = sum_j factors[i,j] * (c_i - c_j)
        
        # Sum over j (axis 1)
        # term[i,j] = factors[i,j] * (c_i - c_j)
        # This is tricky to vectorize directly as element-wise product with diff.
        
        # Let's compute force directly.
        # F_i = sum_j k * (2r - d_ij) * (c_i - c_j) / d_ij
        # This is the negative gradient of the potential.
        
        # We can compute this as:
        # For each pair i,j where overlap>0:
        # delta = (2r - d_ij) / d_ij * (c_i - c_j)
        # grad[i] -= delta
        # grad[j] += delta
        
        # Vectorized:
        # diff is (N, N, 2)
        # factors is (N, N)
        # We want to sum factors[i,j] * diff[i,j] over j for each i.
        # But diff[i,j] = c_i - c_j.
        # sum_j factors[i,j] * (c_i - c_j) = c_i * sum_j factors[i,j] - sum_j factors[i,j]*c_j
        
        # Let's just use a loop for clarity and speed since N=50 is small, or use sparse logic?
        # N=50, N^2=2500. A loop in Python is slow. Vectorize.
        
        # Sum of factors per row
        sum_factors = np.sum(factors, axis=1)
        
        # sum_j factors[i,j] * c_j
        # This is matrix multiplication: factors @ centers
        weighted_centers = factors @ centers
        
        # grad[i] (energy gradient) = - ( c_i * sum_factors[i] - weighted_centers[i] )
        # Wait, let's re-verify signs.
        # E = sum (2r - d)^2
        # dE/dc_i = sum_j 2(2r-d) * (-(c_i-c_j)/d) = - sum_j factors[i,j] * (c_i-c_j)
        # = - ( c_i * sum_j factors[i,j] - sum_j factors[i,j] * c_j )
        # = - ( c_i * sum_factors[i] - weighted_centers[i] )
        
        # We want to move in direction -dE/dc_i to minimize E.
        # Direction = c_i * sum_factors[i] - weighted_centers[i]
        
        pair_grad = centers * sum_factors[:, np.newaxis] - weighted_centers
        
        # Wall constraints
        # E_wall = sum_i (max(0, 2r - (0.5 - c_ix))^2 + ...)
        # For left wall: 2r - (0.5 - c_ix) = 2r - 0.5 + c_ix
        # Overlap if c_ix < 0.5 - 2r
        # Let L = 0.5 - 2r
        # If c_ix < L, overlap = L - c_ix
        # dE/dc_ix = 2(L - c_ix) * (-1) = 2(c_ix - L)
        # Move direction -dE = 2(L - c_ix). Pushes right. Correct.
        
        L = 0.5 - 2*r
        R = -0.5 + 2*r # Wait, right wall is 0.5. Distance is 0.5 - c_ix.
                       # Overlap if 0.5 - c_ix < 2r => c_ix > 0.5 - 2r = L.
                       # So right wall constraint is c_ix <= L.
                       # Overlap if c_ix > L.
                       # E = (c_ix - L)^2. dE/dc_ix = 2(c_ix - L).
                       # Move direction -dE = 2(L - c_ix). Pushes left. Correct.
                       
        # So both walls use the same logic with L = 0.5 - 2r.
        # If c_ix < -L (i.e. c_ix < -(0.5-2r) = 2r - 0.5), then left wall overlap.
        # Left wall: distance 0.5 - |c_ix|. If c_ix < 0, dist = 0.5 + c_ix.
        # Overlap if 0.5 + c_ix < 2r => c_ix < 2r - 0.5 = -L.
        # Overlap amount = -L - c_ix.
        # dE/dc_ix = 2(-L - c_ix) * (-1) = 2(L + c_ix).
        # Move direction -dE = 2(-L - c_ix). Pushes right (increases c_ix). Correct.
        
        # Let's compute wall gradient (energy gradient)
        wall_grad = np.zeros_like(centers)
        
        # Left/Bottom walls (c < -L)
        mask_lb = centers < -L
        wall_grad[mask_lb] += 2.0 * (L + centers[mask_lb])
        
        # Right/Top walls (c > L)
        mask_rt = centers > L
        wall_grad[mask_rt] += 2.0 * (centers[mask_rt] - L)
        
        # Total energy gradient
        total_grad = pair_grad + wall_grad
        
        # We want to move in direction -total_grad
        # So the "force" or "direction" is -total_grad
        return -total_grad

    # Gradient Descent loop
    # Start with a small step size
    step = 0.01
    centers_current = centers.copy()
    
    # Initial overlap check
    def _check_overlap(c, r):
        # Returns max overlap
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        max_pair_overlap = np.max(2*r - dist)
        
        L = 0.5 - 2*r
        # Wall overlaps
        # Left/Bottom: -L - c
        max_lb_overlap = np.max(-L - c)
        # Right/Top: c - L
        max_rt_overlap = np.max(c - L)
        
        return max(max_pair_overlap, max_lb_overlap, max_rt_overlap)
    
    current_overlap = _check_overlap(centers_current, r)
    if current_overlap <= 1e-8:
        return centers_current, r, True
        
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        grad = _overlap_gradient(centers_current, r)
        norm_grad = np.linalg.norm(grad)
        
        if norm_grad < 1e-10:
            break
            
        # Move
        centers_new = centers_current + step * grad
        
        # Project to square (hard constraint for stability, though gradient handles it)
        centers_new = np.clip(centers_new, -0.5, 0.5)
        
        new_overlap = _check_overlap(centers_new, r)
        
        if new_overlap < current_overlap:
            centers_current = centers_new
            current_overlap = new_overlap
            # Adaptive step: increase if improving
            step *= 1.1
        else:
            # Decrease step
            step *= 0.5
            
        if current_overlap <= 1e-8:
            return centers_current, r, True
            
    return centers_current, r, False

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Initial configurations
    # 1. Grid
    ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
    ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    
    inits = [grid]
    
    # 2. Hexagonal
    # Approximate hex grid
    pts = []
    dy = 0.14 * np.sqrt(3)/2
    for j in range(10):
        for i in range(10):
            x = i * 0.14 - 0.5
            y = j * dy - 0.5
            if j % 2 != 0: x += 0.07
            pts.append((x,y))
    hex_grid = np.array(pts)[:N]
    # Center it
    hex_grid -= np.mean(hex_grid, axis=0)
    inits.append(hex_grid)
    
    # 3. Random
    np.random.seed(42)
    rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(rand)
    
    # For each init, try to expand radius
    for init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        centers = init.copy()
        # Start with a small radius
        r = 0.01
        # Max possible radius is approx 0.5/7 ~ 0.07
        # We will increase r in steps
        
        # Determine step size for r
        # Smaller steps are safer but slower.
        # Let's try to find a feasible r first.
        # Then increase.
        
        # Actually, let's just run the expansion loop.
        # The _expand_radius function takes a target r.
        # We need to find the max r.
        
        # Strategy:
        # 1. Find a feasible r0 (e.g., by shrinking until no overlap).
        # 2. Increase r in small steps, calling _expand_radius for each step.
        # 3. If _expand_radius fails, try smaller step or break.
        
        # Step 1: Find feasible start
        r_test = 0.05
        centers_test = centers.copy()
        # If r_test is too big, we might have overlaps.
        # Let's just start at r=0.01 and go up.
        
        current_r = 0.01
        current_centers = centers.copy()
        
        # Check if current_r is feasible
        # If not, shrink r until it is.
        while current_r > 0.001:
            _, _, success = _expand_radius(current_centers, current_r, max_iters=50, time_limit=0.5)
            if success:
                current_centers = _expand_radius(current_centers, current_r, max_iters=100, time_limit=1.0)[0]
                break
            else:
                current_r *= 0.9
                
        # Now increase r
        r_step = 0.001
        max_r = 0.075
        
        while current_r < max_r and time.time() - start_time < DEADLINE - 5:
            # Try to expand to current_r + r_step
            target_r = current_r + r_step
            
            new_centers, achieved_r, success = _expand_radius(current_centers, target_r, max_iters=500, time_limit=2.0)
            
            if success:
                current_centers = new_centers
                current_r = achieved_r
                # Update best
                r_val = _radius(current_centers)
                if r_val > best_r:
                    best_r = r_val
                    best_centers = current_centers.copy()
            else:
                # Failed to expand. Try smaller step?
                # Or just break?
                # If we break, we might miss a local minimum where a slightly different path works.
                # But for hard disks, if we can't push through, we're likely stuck.
                # Let's try a smaller step once.
                r_step /= 2.0
                if r_step < 1e-5:
                    break
                # Try again with smaller step
                target_r = current_r + r_step
                new_centers, achieved_r, success = _expand_radius(current_centers, target_r, max_iters=200, time_limit=1.0)
                if success:
                    current_centers = new_centers
                    current_r = achieved_r
                    r_val = _radius(current_centers)
                    if r_val > best_r:
                        best_r = r_val
                        best_centers = current_centers.copy()
                else:
                    break
                    
    # If time remains, do random restarts
    while time.time() - start_time < DEADLINE - 5:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        centers = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        current_r = 0.01
        # Quick feasibility check
        _, _, success = _expand_radius(centers, current_r, max_iters=50, time_limit=0.5)
        if not success:
            continue
            
        centers = _expand_radius(centers, current_r, max_iters=100, time_limit=1.0)[0]
        
        r_step = 0.001
        max_r = 0.075
        
        while current_r < max_r and time.time() - start_time < DEADLINE - 5:
            target_r = current_r + r_step
            new_centers, achieved_r, success = _expand_radius(centers, target_r, max_iters=300, time_limit=1.5)
            
            if success:
                centers = new_centers
                current_r = achieved_r
                r_val = _radius(centers)
                if r_val > best_r:
                    best_r = r_val
                    best_centers = centers.copy()
            else:
                break
                
    if best_centers is None:
        # Fallback to grid
        return grid
        
    return best_centers

if __name__ == "__main__":
    # Test
    t0 = time.time()
    c = construct()
    t1 = time.time()
    r = _radius(c)
    print(f"Radius: {r}, Time: {t1-t0}")
