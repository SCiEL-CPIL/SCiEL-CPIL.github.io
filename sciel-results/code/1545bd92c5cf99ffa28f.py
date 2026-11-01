import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # --- Helper Functions ---
    
    def compute_radius(centers):
        """
        Compute the largest common radius for the given centers.
        Centers are in [-0.5, 0.5]^3.
        """
        if not np.all(np.isfinite(centers)):
            return 0.0
        if centers.shape != (N, 3):
            return 0.0
        
        # Distance to walls
        dists_to_walls = 0.5 - np.abs(centers)
        min_wall_dist = np.min(dists_to_walls)
        
        # Pairwise distances
        # Using cdist is generally efficient and stable
        dists = cdist(centers, centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        return min(min_wall_dist, min_pair_dist / 2.0)

    def soft_min_objective(centers_flat, beta, iu):
        """
        Objective function for L-BFGS-B.
        Minimizes the soft-min approximation of the radius constraints.
        r = min(wall_dists, pair_dists/2).
        We want to maximize r, so we minimize -r.
        Using soft-min: S_beta(x) = (1/beta) * log(sum(exp(-beta * x))).
        As beta -> inf, S_beta(x) -> min(x).
        So we minimize S_beta(constraints).
        """
        centers = centers_flat.reshape(-1, 3)
        
        # 1. Wall distances: 0.5 - |c_i|
        dists_to_walls = 0.5 - np.abs(centers)
        wall_vals = dists_to_walls.flatten()
        
        # 2. Pairwise distances / 2
        # We use the precomputed indices for the upper triangle to avoid duplicates
        # Note: cdist returns a full matrix. We extract unique pairs.
        dists = cdist(centers, centers, metric='euclidean')
        pair_vals = dists[iu] / 2.0
        
        all_vals = np.concatenate([wall_vals, pair_vals])
        
        # Calculate Soft Min: (1/beta) * ln(sum(exp(-beta * x)))
        # To avoid overflow/underflow, we use the log-sum-exp trick.
        # Let M = max(-beta * x). Then sum(exp(-beta * x)) = exp(M) * sum(exp(-beta * x - M)).
        # ln(sum(exp(-beta * x))) = M + ln(sum(exp(-beta * x - M))).
        # Result = (M + ln(sum(...))) / beta.
        
        neg_beta_x = -beta * all_vals
        max_val = np.max(neg_beta_x)
        
        # If max_val is very large positive, exp(max_val - max_val) = 1, safe.
        # If max_val is very large negative, exp terms are small, safe.
        # We just need to be careful if all_vals are huge. But they are bounded by 0.5.
        
        exp_terms = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            return 1e10
            
        obj = (max_val + np.log(sum_exp)) / beta
        
        return obj

    def build_bcc_init(n=100, noise_scale=0.005, seed=42):
        """
        Build a BCC (Body-Centered Cubic) inspired initialization.
        This creates a dense, regular structure that is a good starting point.
        """
        # Approximate grid size for 100 points
        # A 5x5x4 grid of BCC cells gives 5*5*4*2 = 200 points.
        # We take a subset or just generate enough and pick 100.
        nx, ny, nz = 5, 5, 4
        margin = 0.02
        xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
        ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
        zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
        
        all_centers = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Corner points
                    all_centers.append([xs[i], ys[j], zs[k]])
                    # Body center
                    if i < nx-1 and j < ny-1 and k < nz-1:
                        all_centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2])
        
        all_centers = np.array(all_centers)
        rng = np.random.default_rng(seed)
        
        if len(all_centers) >= N:
            # Randomly select N points to break symmetry slightly but keep structure
            idx = rng.choice(len(all_centers), N, replace=False)
            centers = all_centers[idx]
        else:
            centers = all_centers
            # Pad if needed (should not happen with 200 pts)
            while len(centers) < N:
                centers = np.vstack([centers, centers[0]])
        
        centers += rng.normal(0, noise_scale, centers.shape)
        centers = np.clip(centers, -0.499, 0.499)
        
        return centers

    def build_halton_init(n=100, seed=42):
        """
        Build a Halton sequence based initialization for good space-filling properties.
        """
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result
        
        rng = np.random.default_rng(seed)
        centers = np.zeros((n, 3))
        for i in range(1, n+1):
            centers[i-1, 0] = -0.5 + halton(i, 2)
            centers[i-1, 1] = -0.5 + halton(i, 3)
            centers[i-1, 2] = -0.5 + halton(i, 5)
        
        centers += rng.normal(0, 0.005, centers.shape)
        centers = np.clip(centers, -0.499, 0.499)
        
        return centers

    def optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=6):
        """
        Run L-BFGS-B optimization with increasing beta from initial centers.
        Returns (best_centers, best_radius).
        """
        best_centers = centers.copy()
        best_radius = compute_radius(best_centers)
        
        iu = np.triu_indices(N, k=1)
        betas = np.geomspace(beta_start, beta_end, n_beta_steps)
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            
            x0 = best_centers.flatten()
            
            try:
                res = minimize(
                    soft_min_objective,
                    x0,
                    args=(beta, iu),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                new_centers = res.x.reshape(-1, 3)
                # Project back into cube strictly
                new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                
                r = compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers.copy()
            except Exception:
                pass
        
        return best_centers, best_radius

    # --- Main Logic ---

    # 1. Initialize with BCC
    init_centers = build_bcc_init()
    best_centers = init_centers.copy()
    best_r = compute_radius(best_centers)
    
    # 2. Run optimization from BCC
    c_opt, r_opt = optimize_from_initial(init_centers, deadline)
    if r_opt > best_r:
        best_r = r_opt
        best_centers = c_opt.copy()
    
    # 3. If time permits, try Halton initialization
    if time.time() < deadline - 40.0:
        init_halton = build_halton_init()
        c_opt_h, r_opt_h = optimize_from_initial(init_halton, deadline)
        if r_opt_h > best_r:
            best_r = r_opt_h
            best_centers = c_opt_h.copy()

    # 4. Final high-precision polish
    if time.time() < deadline - 15.0:
        iu = np.triu_indices(N, k=1)
        beta = 10000.0
        try:
            res = minimize(
                soft_min_objective,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            r = compute_radius(new_centers)
            if r > best_r:
                best_r = r
                best_centers = new_centers
        except Exception:
            pass

    # 5. Final validation and safety
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    if not np.all(np.isfinite(best_centers)):
        # Fallback to grid if something went wrong
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].astype(np.float64)

    return best_centers
