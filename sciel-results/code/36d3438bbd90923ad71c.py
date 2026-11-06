import numpy as np
import time
import math
import random
from scipy.optimize import minimize

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 210.0  # Leave 20s margin for safety against 230s limit

    # 1. Setup Fast Band Solver
    # Use a reduced set of plane waves for the internal search loop to be fast.
    # The evaluator uses 289 (17x17). We use 15x15 (225) or 13x13 (169) for speed.
    # 13x13 is often sufficient for ranking shapes, but let's stick to 15x15 for better accuracy 
    # while still being significantly faster than 17x17 if optimized, or just use 15x15.
    half_pw = 7  # 15x15 grid
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # The evaluator uses 30 points. We use a representative subset for speed.
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    k_points = []
    n_k = 6
    for i in range(n_k + 1):
        for j in range(i, n_k + 1): # Ensure kx <= ky
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    # Add high symmetry points specifically
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    k_points = np.array(k_points)
    
    # Precompute kG and kG2 for each k-point to avoid recalculating in matvec
    # kG[k_idx] is a (nG, 2) array
    kG_list = []
    kG2_list = []
    for kx, ky in k_points:
        kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
        kG2 = kG[:, 0]**2 + kG[:, 1]**2
        kG_list.append(kG)
        kG2_list.append(kG2)
        
    # Precompute FFT of the inverse epsilon? No, epsilon changes.
    # But we can precompute the FFT grid indices.
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        # Note: For TM modes in 2D, the eigenvalue problem involves 1/epsilon.
        # We solve A v = lambda v, where A is built from 1/eps_F.
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for idx, kG2 in enumerate(kG2_list):
            def matvec(v):
                w = kG2 * v
                # Reshape to 2D for FFT
                u = np.fft.fft2(w.reshape(15, 15))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-4)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def generate_cell(params):
        """
        Generate a symmetric 32x32 cell from parameters.
        params: [rx, ry, n, cx, cy]
        We define 4 shapes in the 4 quadrants.
        Due to symmetry about center (15.5, 15.5), we can define one shape in the top-left quadrant 
        and mirror it.
        Actually, the symmetry is C4v about the center of the cell.
        The cell is 32x32. Center is between 15 and 16.
        Fundamental domain: 0 <= i < 16, 0 <= j < 16.
        We place a superellipse in the top-left quadrant.
        Center of the superellipse in the fundamental domain: (cx, cy) where cx, cy in [0, 16].
        Equation: |x - cx|^n + |y - cy|^n <= R^n
        We use separate Rx, Ry for aspect ratio.
        |x - cx|^n / Rx^n + |y - cy|^n / Ry^n <= 1
        """
        rx, ry, n, cx, cy = params
        cell = np.zeros((size, size), dtype=int)
        
        # Create coordinate grid for the full cell
        # But we only need to fill the fundamental domain and mirror.
        # Fundamental domain: i in [0, 15], j in [0, 15]
        # Wait, the symmetry is about the center of the 32x32 grid.
        # Indices 0..31. Center is 15.5.
        # Reflection: i' = 31 - i, j' = 31 - j.
        # Rotation 90: i' = 31 - j, j' = i.
        
        # Let's just iterate over the top-left 16x16 block.
        # If a point (i, j) in [0,15]x[0,15] is inside the shape, we set it to 1.
        # Then we mirror it to the other 3 quadrants.
        
        # Shape center in top-left quadrant.
        # To maximize gap, usually the dielectric is placed such that it couples specific BZ points.
        # Common good structures for 4-5 gap are "pill" shapes or "crosses".
        # Let's place the shape centered at (cx, cy) within the 16x16 block.
        
        ys, xs = np.indices((16, 16))
        
        # Distance calculation
        # To handle the superellipse robustly, we use the formula:
        # (|x-cx|/rx)^n + (|y-cy|/ry)^n <= 1
        
        dx = np.abs(xs - cx)
        dy = np.abs(ys - cy)
        
        # Avoid division by zero if rx or ry is 0
        if rx < 0.1 or ry < 0.1:
             mask = np.zeros((16, 16), dtype=bool)
        else:
             mask = (dx / rx) ** n + (dy / ry) ** n <= 1.0
        
        # Fill the top-left 16x16
        cell[0:16, 0:16] = mask.astype(int)
        
        # Mirror to other quadrants
        # Top-Right: flip horizontally
        cell[0:16, 16:32] = cell[0:16, 15:-1:-1]
        # Bottom-Left: flip vertically
        cell[16:32, 0:16] = cell[15:-1:-1, 0:16]
        # Bottom-Right: flip both
        cell[16:32, 16:32] = cell[15:-1:-1, 15:-1:-1]
        
        return cell

    # 2. Optimization Strategy
    # We will perform a coarse grid search over the parameters, then refine the best candidate.
    
    # Parameters: rx, ry, n, cx, cy
    # Ranges:
    # rx, ry: 0.5 to 15.0 (size of shape)
    # n: 0.5 to 4.0 (exponent, 2 is ellipse, 4 is squircle, <1 is star-like)
    # cx, cy: 0.0 to 15.0 (position in top-left quadrant)
    
    best_score = -np.inf
    best_params = None
    best_cell = None
    
    # Coarse Grid
    rx_vals = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]
    ry_vals = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]
    n_vals = [1.5, 2.0, 2.5, 3.0, 4.0]
    cx_vals = [4.0, 8.0, 12.0]
    cy_vals = [4.0, 8.0, 12.0]
    
    # To save time, we might not test all combinations. 
    # Total combos: 6*6*5*3*3 = 1620.
    # Each eval ~ 0.5s? 1620 * 0.5 = 810s. Too slow.
    # We need to reduce the grid or make eval faster.
    # Let's reduce grid.
    rx_vals = [4.0, 6.0, 8.0, 10.0, 12.0]
    ry_vals = [4.0, 6.0, 8.0, 10.0, 12.0]
    n_vals = [2.0, 3.0, 4.0]
    cx_vals = [5.0, 8.0, 11.0]
    cy_vals = [5.0, 8.0, 11.0]
    # Combos: 5*5*3*3*3 = 675. Still might be tight if eval is 1s.
    # Let's check eval time.
    
    # Let's do a smarter search: 
    # 1. Fix cx, cy to center of quadrant (8,8) initially? 
    #    Actually, off-center might be better for breaking degeneracy or shifting bands.
    #    But symmetry is enforced.
    # 2. Use Nelder-Mead on a subset of promising regions.
    
    # Let's try a random sampling approach with local refinement.
    # Sample 50 random configurations, evaluate, then refine top 5.
    
    num_samples = 40
    samples = []
    
    # Include some deterministic good guesses
    # "Pill" shape: n=4, rx=ry=6, cx=8, cy=8
    samples.append([6.0, 6.0, 4.0, 8.0, 8.0])
    # "Square" shape: n=10, rx=ry=5, cx=8, cy=8
    samples.append([5.0, 5.0, 10.0, 8.0, 8.0])
    # "Circle" shape: n=2, rx=ry=7, cx=8, cy=8
    samples.append([7.0, 7.0, 2.0, 8.0, 8.0])
    # Off-center pill
    samples.append([6.0, 6.0, 4.0, 5.0, 5.0])
    samples.append([6.0, 6.0, 4.0, 11.0, 11.0])
    
    # Random samples
    while len(samples) < num_samples:
        rx = random.uniform(3.0, 14.0)
        ry = random.uniform(3.0, 14.0)
        n = random.uniform(1.0, 6.0)
        cx = random.uniform(2.0, 14.0)
        cy = random.uniform(2.0, 14.0)
        samples.append([rx, ry, n, cx, cy])
        
    # Evaluate samples
    scored_samples = []
    for params in samples:
        if time.time() > deadline - 30: break
        cell = generate_cell(params)
        score = compute_score(cell)
        scored_samples.append((score, params, cell))
        
    # Sort by score
    scored_samples.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top 5 candidates using Nelder-Mead
    for i in range(5):
        if time.time() > deadline - 10: break
        if i >= len(scored_samples): break
        
        best_s, best_p, best_c = scored_samples[i]
        
        def objective(params):
            cell = generate_cell(params)
            return -compute_score(cell)
            
        # Bounds:
        # rx, ry: [1, 15]
        # n: [1, 10]
        # cx, cy: [0, 15]
        
        try:
            res = minimize(objective, best_p, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 0.1, 'fatol': 0.001})
            if res.fun < -best_s: # res.fun is -score
                new_score = -res.fun
                new_params = res.x
                new_cell = generate_cell(new_params)
                # Verify
                final_score = compute_score(new_cell)
                if final_score > best_score:
                    best_score = final_score
                    best_params = new_params
                    best_cell = new_cell
        except Exception:
            pass

    # If no cell was found or score is very low, return a default good structure
    if best_cell is None or best_score < 0.1:
        # Default: 4 squares of 5x5 at (8,8), (23,8), (8,23), (23,23)
        best_cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for cx, cy in centers:
            best_cell[cx-2:cx+3, cy-2:cy+3] = 1
        best_score = compute_score(best_cell)

    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
