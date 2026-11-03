import numpy as np
import time
import math

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def compute_radius(centers):
        """
        Computes the max radius for the given centers.
        Returns 0 if invalid.
        """
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check bounds
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Pair distance
        # Use vectorized distance calculation
        # diff[i, j] = c_i - c_j
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        # Set diagonal to inf
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_dist_sq)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(rows, cols, jitter=0.0, seed=0):
        """
        Generates a hexagonal lattice packing.
        """
        rng = np.random.RandomState(seed)
        # Optimal hexagonal spacing for a square of side 1
        # If we have 'cols' circles in a row, width = cols * d + (cols-1)*0 (if touching)
        # Actually, for hex packing, horizontal distance is d, vertical is d*sqrt(3)/2.
        # Let's estimate d based on cols.
        # Width available is 1.
        # If we center the grid, the extent is (cols-1)*d + d = cols*d? 
        # No, the centers span from -0.5+d/2 to 0.5-d/2.
        # So (cols-1)*d <= 1 - d => cols*d <= 1 => d <= 1/cols.
        # Let's use d = 1/cols * 0.98 to be safe, or solve for d such that it fits.
        
        # Let's try to fit as many as possible.
        # Approximate d
        d = 1.0 / cols * 0.99
        
        # Check if it fits vertically
        row_spacing = d * math.sqrt(3) / 2
        total_height = (rows - 1) * row_spacing + d
        if total_height > 1.0:
            # Shrink d to fit
            # (rows-1) * d * sqrt(3)/2 + d <= 1
            # d * ((rows-1)*sqrt(3)/2 + 1) <= 1
            d = 1.0 / ((rows - 1) * math.sqrt(3) / 2 + 1) * 0.99
            
        pts = []
        for row in range(rows):
            y = -0.5 + d/2 + row * (d * math.sqrt(3) / 2)
            if row % 2 == 0:
                # Full row
                x_start = -0.5 + d/2
                for col in range(cols):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
            else:
                # Offset row
                x_start = -0.5 + d
                for col in range(cols - 1):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        else:
            return None

    def square_grid_init(m, jitter=0.0, seed=0):
        rng = np.random.RandomState(seed)
        d = 1.0 / m
        pts = []
        for i in range(m):
            for j in range(m):
                x = -0.5 + (i + 0.5) * d
                y = -0.5 + (j + 0.5) * d
                if jitter > 0:
                    x += rng.uniform(-jitter*d, jitter*d)
                    y += rng.uniform(-jitter*d, jitter*d)
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        else:
            return None

    def nelder_mead_simplex(centers, max_iter=500, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        """
        Custom Nelder-Mead implementation for 2N variables.
        Objective: Maximize radius.
        We minimize -radius.
        """
        x0 = centers.flatten()
        n_vars = len(x0)
        
        # Initialize simplex
        # Standard initialization: x0, and x0 + delta * e_i
        # delta is important. Let's start with a small delta relative to the box size.
        # The box is 1x1.
        # Initial radius is roughly 0.03.
        # Let's use a delta that allows moving 10% of the box size? No, that's too big.
        # Let's use delta = 0.01
        delta = 0.01
        
        simplex = np.zeros((n_vars + 1, n_vars))
        simplex[0] = x0
        for i in range(n_vars):
            simplex[i+1] = x0
            simplex[i+1][i] += delta
            
        # Evaluate objective
        def objective(x):
            c = x.reshape(N, 2)
            # Hard clip to ensure validity during exploration? 
            # No, better to return a large penalty if outside.
            # But the problem says "inside the square".
            # If we go outside, compute_radius returns 0.
            # We want to maximize r, so minimize -r.
            r = compute_radius(c)
            # Add a small penalty for being close to boundary to keep us inside?
            # compute_radius already handles this by reducing r.
            return -r
            
        f_simplex = np.array([objective(s) for s in simplex])
        
        # Sort simplex
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]
        
        best_f = f_simplex[0]
        
        # Time management inside NM
        nm_start = time.time()
        nm_deadline = nm_start + 10.0 # Allow up to 10s per NM run
        
        for it in range(max_iter):
            if time.time() > nm_deadline:
                break
                
            # Check convergence
            if np.max(f_simplex) - np.min(f_simplex) < tol:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = objective(xr)
            
            if fr < f_simplex[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                # Just replace
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                # Contraction
                if fr < f_simplex[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                fc = objective(xc)
                
                if fc < f_simplex[-1]:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n_vars + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_simplex[i] = objective(simplex[i])
            
            # Re-sort
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Update best
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                
            # If we haven't improved in a while, break?
            # The convergence check handles this.
            
        # Return best point
        return simplex[0].reshape(N, 2)

    # Main Logic
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal grids with different row counts and jitters
    # N=200. sqrt(200) ~ 14.
    # Hex packing is denser.
    # Rows * Cols ~ 200.
    for rows in range(12, 18):
        cols = math.ceil(N / rows)
        # Add non-jittered
        pts = hexagonal_init(rows, cols, jitter=0.0, seed=rows)
        if pts is not None:
            inits.append(pts)
        # Add jittered
        for jitter in [0.05, 0.1, 0.2]:
            pts = hexagonal_init(rows, cols, jitter=jitter, seed=rows*100 + int(jitter*100))
            if pts is not None:
                inits.append(pts)
                
    # 2. Square grids
    m = int(np.ceil(np.sqrt(N)))
    for jitter in [0.0, 0.05, 0.1]:
        pts = square_grid_init(m, jitter=jitter, seed=42)
        if pts is not None:
            inits.append(pts)
            
    # 3. Random
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-0.49, 0.49, (N, 2))
        inits.append(pts)

    # Evaluate all inits
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        r = compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
            
    # Run Nelder-Mead on the best few initializations
    # We need to be careful with time.
    # Let's sort inits by radius and try the top 5.
    init_scores = [(compute_radius(init), i) for i, init in enumerate(inits)]
    init_scores.sort(reverse=True)
    
    # Try top 5 distinct-ish inits
    tried_indices = set()
    for score, idx in init_scores:
        if len(tried_indices) >= 5:
            break
        if idx in tried_indices:
            continue
        tried_indices.add(idx)
        
        if time.time() > deadline - 15:
            break
            
        # Run NM on this init
        # We need to allocate time carefully.
        # If we have 140s left, and we want to try 5, we have ~25s each?
        # But NM is expensive. Let's limit NM to 10s per run.
        
        start_nm = time.time()
        optimized_centers = nelder_mead_simplex(inits[idx], max_iter=200, tol=1e-9)
        end_nm = time.time()
        
        r_opt = compute_radius(optimized_centers)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized_centers.copy()
            
        # If we improved significantly, maybe re-run NM on the result?
        # Or just move to next init.
        
    # Final polish on the absolute best
    if best_centers is not None and time.time() < deadline - 5:
        # Run one more NM on the best result found so far
        # Use a tighter tolerance and maybe smaller initial step?
        # The custom NM uses fixed delta. 
        # Let's just run it again with a different seed or perturbation?
        # Actually, let's just run it on the current best.
        final_centers = nelder_mead_simplex(best_centers, max_iter=200, tol=1e-10)
        r_final = compute_radius(final_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final_centers.copy()

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
