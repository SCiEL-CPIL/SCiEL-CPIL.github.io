import time
import numpy as np
from scipy.spatial import Delaunay, distance
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Helper: compute the radius of a packing (min of half-pair-dist and boundary-dist)
    def compute_radius(pos):
        # Boundary
        xs = pos[:, 0]
        ys = pos[:, 1]
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)
        r_b = np.min(dist_b)
        
        # Pairwise
        # Use cdist for speed
        dists = distance.cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        r_p = np.min(dists) / 2.0
        
        return min(r_b, r_p)

    def hexagonal_init(n, s=0.074):
        points = []
        # Generate a grid of hex points
        # Rows
        for i in range(-12, 12):
            for j in range(-12, 12):
                x = j * s + (i % 2) * s / 2
                y = i * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points)
        # Filter points roughly in the square
        mask = (points[:, 0] > -0.5) & (points[:, 0] < 0.5) & (points[:, 1] > -0.5) & (points[:, 1] < 0.5)
        points = points[mask]
        
        if len(points) > n:
            # Shuffle and take N
            idx = np.random.choice(len(points), n, replace=False)
            points = points[idx]
        elif len(points) < n:
            needed = n - len(points)
            extra = np.random.uniform(-0.4, 0.4, size=(needed, 2))
            points = np.vstack([points, extra])
            
        return points

    def square_grid_init(n):
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]

    # Objective for L-BFGS-B: maximize minimum distance
    # We minimize the negative of the minimum distance, but that's non-smooth.
    # Instead, we minimize a smooth surrogate: -log-sum-exp of distances? 
    # Or simply: minimize sum of (1/d^p) for p large?
    # Let's try: minimize -min(dist) is hard.
    # Better: Minimize the sum of squared violations of a target distance?
    # Let's stick to the "inflation" idea but with a better local solver.
    # Objective: Minimize sum of max(0, target_r - d_ij/2)^2 + max(0, target_r - d_i_b)^2
    
    def overlap_energy(pos, r):
        dists = distance.cdist(pos, pos)
        np.fill_diagonal(dists, 2*r + 1)
        
        pair_overlap = np.maximum(0, 2*r - dists)
        pair_energy = np.sum(pair_overlap**2)
        
        xs = pos[:, 0]
        ys = pos[:, 1]
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)
        
        boundary_overlap = np.maximum(0, r - dist_b)
        boundary_energy = np.sum(boundary_overlap**2)
        
        return pair_energy + boundary_energy

    def polish(pos, r_target, max_iter=200, tol=1e-8):
        """
        Local optimization to find a configuration with radius >= r_target.
        Returns (success, new_pos, actual_radius).
        """
        # Use L-BFGS-B to minimize overlap energy for r_target
        # We can also try to increase r_target slightly if we have time.
        
        # Start with current pos
        current_pos = pos.copy()
        
        # Define objective and gradient (approximate or use finite diff? L-BFGS-B can do finite diff but it's slow)
        # Let's use a smooth objective: minimize sum of exp(-d/epsilon) ? No, that's for max-min.
        # Let's stick to the overlap energy which is C1 except at d=2r and d=r.
        # For r slightly below the limit, it should be smooth enough.
        
        bounds = [(-0.5, 0.5), (-0.5, 0.5)] * N
        
        try:
            result = minimize(
                lambda x: overlap_energy(x, r_target),
                current_pos,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6, 'disp': False}
            )
            current_pos = result.x
        except Exception:
            return False, pos, compute_radius(pos)
            
        # Check feasibility
        energy = overlap_energy(current_pos, r_target)
        actual_r = compute_radius(current_pos)
        
        if actual_r >= r_target - 1e-7:
            return True, current_pos, actual_r
        else:
            return False, current_pos, actual_r

    def delaunay_flip_improve(pos, r_target, max_flips=100):
        """
        Experimental: Use Delaunay triangulation to identify bad triangles and flip edges.
        This is complex to implement robustly for N=200 with boundary constraints.
        We will skip this for now and rely on better local search + multiple restarts.
        """
        pass

    best_radius = 0.0
    best_positions = None
    
    # Initializations
    # 1. Hexagonal
    # 2. Square Grid
    # 3. Random
    
    inits = [
        hexagonal_init(N),
        square_grid_init(N),
        np.random.uniform(-0.4, 0.4, size=(N, 2))
    ]
    
    # Add a few more random inits if time permits
    for _ in range(3):
        inits.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))

    # Strategy:
    # 1. For each init, run a "inflation" loop.
    # 2. Start with r = 0.02.
    # 3. Polish to feasibility.
    # 4. Increase r by small step.
    # 5. If infeasible, binary search for the max r.
    # 6. Keep the best result.
    
    # To save time, we will limit the number of inits we process fully.
    # Each polish takes ~0.1-0.5s. Binary search takes ~10 polishes.
    # So ~0.5-5s per init. We have ~130s. We can do ~20-50 inits.
    # We have 6 inits. We can afford more steps.
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        current_pos = init_pos.copy()
        
        # Start inflation
        r = 0.02
        step = 0.001
        max_r = 0.045
        
        # First, ensure we are feasible at r=0.02
        success, current_pos, actual_r = polish(current_pos, r, max_iter=50)
        
        # If even 0.02 is not feasible, try a smaller r or skip
        if not success:
            # Try to just maximize from this init without inflation
            # Just run polish with a high r and see what we get?
            # Or just use the initial radius?
            actual_r = compute_radius(current_pos)
            if actual_r > best_radius:
                best_radius = actual_r
                best_positions = current_pos.copy()
            continue

        while r < max_r:
            if time.time() > deadline - 5:
                break
                
            r_next = r + step
            success, new_pos, new_r = polish(current_pos, r_next, max_iter=100)
            
            if success:
                current_pos = new_pos
                r = r_next
            else:
                # Binary search between r and r_next
                r_low = r
                r_high = r_next
                
                # We know r_low is feasible (current_pos)
                # We know r_high is not (new_pos failed, but maybe a different config would work? 
                # Assuming the local basin is exhausted, we treat r_high as infeasible for this path)
                
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    r_mid = (r_low + r_high) / 2
                    # Use current_pos as start, or maybe the failed new_pos?
                    # Let's use current_pos (known feasible at r_low) to ensure stability
                    success_mid, pos_mid, r_mid_actual = polish(current_pos, r_mid, max_iter=50)
                    
                    if success_mid:
                        r_low = r_mid
                        current_pos = pos_mid
                    else:
                        r_high = r_mid
                        
                # r_low is the best feasible radius found
                if r_low > best_radius:
                    best_radius = r_low
                    best_positions = current_pos.copy()
                    
                break # Move to next init

        # If we reached max_r and were still feasible, check if we can go higher
        if r >= max_r:
            # Try to push a bit more
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                r_try = r + 0.0005
                success, new_pos, new_r = polish(current_pos, r_try, max_iter=50)
                if success:
                    current_pos = new_pos
                    r = r_try
                else:
                    break
            
            if r > best_radius:
                best_radius = r
                best_positions = current_pos.copy()

    # Final Polish: 
    # If we have time left, try to polish the best_positions found so far with a tighter tolerance
    # or try to nudge the radius up slightly.
    
    if best_positions is not None and time.time() < deadline - 5:
        # Try to improve best_radius by polishing at best_radius + small epsilon
        # Or just re-polish at best_radius to ensure it's optimal for that radius
        # Let's try to push it up by 0.0001
        r_try = best_radius + 0.0001
        success, new_pos, new_r = polish(best_positions, r_try, max_iter=200, tol=1e-10)
        if success:
            best_radius = new_r
            best_positions = new_pos
            
        # One more push
        if time.time() < deadline - 2:
            r_try = best_radius + 0.0001
            success, new_pos, new_r = polish(best_positions, r_try, max_iter=200, tol=1e-10)
            if success:
                best_radius = new_r
                best_positions = new_pos

    # Fallback
    if best_positions is None:
        best_positions = square_grid_init(N)
        
    # Validation
    best_positions = np.clip(best_positions, -0.49999, 0.49999)
    
    return best_positions

if __name__ == "__main__":
    # Quick test
    pos = construct()
    r = compute_radius(pos)
    print(f"Radius: {r}")
