import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 190.0  # Safety margin before 200s hard limit

    # PWM Setup
    N_G = 17
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # Precompute diff indices for H matrix construction
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    valid_diffs = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
    diff_indices = np.where(valid_diffs, (dx + N_G//2) * N_G + (dy + N_G//2), 0)
    
    # K points in irreducible BZ (30 points matching evaluator)
    k_points = []
    for i in range(10):
        t = i / 9.0 * 0.5
        k_points.append([t, 0.0])
        k_points.append([0.5, t])
        k_points.append([t, t])
    k_points.append([0.25, 0.1])
    k_points.append([0.35, 0.2])
    k_points.append([0.45, 0.3])
    k_points = np.array(k_points)

    def get_fourier_eps_inv(cell_arr):
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        # Map G vectors to FFT grid indices
        # G_vectors are in [-8, 8]. FFT grid is 0..31.
        # Index for G is (G + 16) % 32
        Gx_grid = (G_vectors[:, 0] + size//2) % size
        Gy_grid = (G_vectors[:, 1] + size//2) % size
        return coeffs_2d[Gx_grid, Gy_grid]

    def evaluate(cell_arr):
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # H[i, j] = eps_inv_coeffs[G_i - G_j] * (k + G_j)^2
            H = eps_inv_coeffs[diff_indices] * k_plus_G_sq[None, :]
            H[~valid_diffs] = 0.0
            
            # Solve eigenvalue problem
            # H is not symmetric, use eigvals
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # --- Structural Change: RBF-based Parameterized Design ---
    # Instead of random pixel flipping, we optimize a set of Gaussian weights
    # placed at high-symmetry points to generate the binary mask.
    # This allows for smoother exploration of the design space.
    
    # Define centers for RBFs (C4v symmetric)
    # Center (16,16) and a ring of centers
    centers = [(15.5, 15.5)] # Center of cell
    # Add centers at various radii to allow complex shapes
    for r in [4.0, 8.0, 12.0]:
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            cx = 15.5 + r * np.cos(theta)
            cy = 15.5 + r * np.sin(theta)
            centers.append((cx, cy))
            
    centers = np.array(centers)
    n_centers = len(centers)
    
    # Initialize weights
    weights = np.zeros(n_centers)
    weights[0] = 1.0 # Center is on
    
    def generate_cell_from_weights(w):
        cell = np.zeros((size, size), dtype=int)
        # Compute distance from each pixel to each center
        # Pixel centers: (i+0.5)/size - 0.5 -> scale to pixels? 
        # Let's work in pixel coordinates for simplicity
        ys, xs = np.mgrid[0:size, 0:size]
        # Normalize to [-0.5, 0.5] or just use pixel indices
        # Using pixel indices: center is at 15.5
        for idx, (cx, cy) in enumerate(centers):
            if w[idx] == 0:
                continue
            dist_sq = (xs - cx)**2 + (ys - cy)**2
            # Gaussian width
            sigma = 2.5
            field = w[idx] * np.exp(-dist_sq / (2 * sigma**2))
            cell += field
            
        # Threshold
        cell = (cell > 0.5).astype(int)
        
        # Enforce C4v symmetry explicitly (just in case numerical noise breaks it)
        # The centers are symmetric, but let's enforce the pixel symmetry
        # Actually, if centers are symmetric and weights are shared per orbit, 
        # the field is symmetric. But my weights array has unique weights for each center.
        # To ensure C4v, I should group centers into orbits and share weights.
        
        # Let's re-do the weight representation to be orbit-based.
        return cell

    # Better approach: Define orbits of centers.
    # Orbit 0: (15.5, 15.5) - 1 point
    # Orbit 1: (19.5, 15.5), (15.5, 19.5), ... 8 points
    # Orbit 2: (23.5, 15.5), ... 8 points
    # Orbit 3: (27.5, 15.5), ... 8 points
    
    orbits = []
    # Center
    orbits.append([(15.5, 15.5)])
    # Ring 1
    for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
        orbits.append([(15.5 + 4.0*np.cos(theta), 15.5 + 4.0*np.sin(theta))])
    # Wait, the orbit definition above is wrong. An orbit contains ALL points related by symmetry.
    # Let's define orbits properly.
    
    def get_orbit_points(cx, cy):
        N = size
        # Pixel coordinates are 0..31. Centers are float.
        # Symmetry operations on pixel indices (i, j):
        # (i, j), (j, N-1-i), (N-1-i, N-1-j), (N-1-j, i)
        # (i, N-1-j), (N-1-i, j), (j, i), (N-1-j, N-1-i)
        # For continuous centers, we apply the same logic to coordinates.
        # Cell center is (15.5, 15.5).
        # Let's work in coordinates centered at 0.
        # x = cx - 15.5, y = cy - 15.5
        # Ops: (x,y), (y,-x), (-x,-y), (-y,x), (-x,y), (-y,-x), (y,x), (x,-y)
        
        x = cx - 15.5
        y = cy - 15.5
        
        pts = [
            (x, y),
            (y, -x),
            (-x, -y),
            (-y, x),
            (-x, y),
            (-y, -x),
            (y, x),
            (x, -y)
        ]
        
        unique_pts = []
        for p in pts:
            # Check if already in unique_pts
            found = False
            for u in unique_pts:
                if abs(p[0]-u[0]) < 1e-5 and abs(p[1]-u[1]) < 1e-5:
                    found = True
                    break
            if not found:
                unique_pts.append(p)
        
        # Convert back to cell coords
        return [(px + 15.5, py + 15.5) for px, py in unique_pts]

    # Define unique orbits
    orbit_representatives = [
        (15.5, 15.5),
        (19.5, 15.5),
        (23.5, 15.5),
        (27.5, 15.5),
        (19.5, 19.5),
        (23.5, 23.5)
    ]
    
    orbits = []
    for rep in orbit_representatives:
        pts = get_orbit_points(rep[0], rep[1])
        orbits.append(pts)
        
    n_orbits = len(orbits)
    
    def generate_cell_from_orbit_weights(w):
        cell = np.zeros((size, size), dtype=float)
        ys, xs = np.mgrid[0:size, 0:size]
        
        for o_idx, orbit in enumerate(orbits):
            w_val = w[o_idx]
            if w_val == 0:
                continue
            for (cx, cy) in orbit:
                dist_sq = (xs - cx)**2 + (ys - cy)**2
                sigma = 3.0
                field = w_val * np.exp(-dist_sq / (2 * sigma**2))
                cell += field
                
        # Threshold
        binary_cell = (cell > 0.5).astype(int)
        return binary_cell

    # Initialize weights
    best_weights = np.zeros(n_orbits)
    best_weights[0] = 1.0 # Center on
    best_weights[1] = 0.5 # Inner ring
    
    best_cell = generate_cell_from_orbit_weights(best_weights)
    best_score = evaluate(best_cell)
    
    # Local search on weights
    # Weights can be continuous, but we need binary output.
    # We will perturb weights and check if the resulting binary cell is valid and better.
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        # Pick an orbit to modify
        o_idx = np.random.randint(n_orbits)
        
        # Perturb weight
        delta = np.random.uniform(-0.2, 0.2)
        new_weights = best_weights.copy()
        new_weights[o_idx] += delta
        
        # Clamp weights to [0, 1]
        new_weights = np.clip(new_weights, 0.0, 1.0)
        
        new_cell = generate_cell_from_orbit_weights(new_weights)
        
        # Check if cell is valid (should be, but maybe all 0 or all 1?)
        if np.all(new_cell == 0) or np.all(new_cell == 1):
            continue
            
        score = evaluate(new_cell)
        
        if score > best_score:
            best_score = score
            best_cell = new_cell
            best_weights = new_weights
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Random restart
            best_weights = np.random.uniform(0, 1, n_orbits)
            best_weights[0] = 1.0 # Keep center on
            best_cell = generate_cell_from_orbit_weights(best_weights)
            best_score = evaluate(best_cell)
            no_improve = 0

    # Final local search on pixels to refine
    cell = best_cell.copy()
    no_improve = 0
    max_no_improve = 200
    
    def get_pixel_orbit(i, j):
        N = size
        points = [
            (i, j),
            (j, N-1-i),
            (N-1-i, N-1-j),
            (N-1-j, i),
            (i, N-1-j),
            (N-1-i, j),
            (j, i),
            (N-1-j, N-1-i)
        ]
        return list(set(points))

    while time.time() - start_time < time_limit:
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        orbit = get_pixel_orbit(i, j)
        
        # Flip
        for (r, c) in orbit:
            cell[r, c] = 1 - cell[r, c]
            
        score = evaluate(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            no_improve = 0
        else:
            for (r, c) in orbit:
                cell[r, c] = 1 - cell[r, c]
            no_improve += 1
            
        if no_improve > max_no_improve:
            cell = best_cell.copy()
            no_improve = 0

    return best_cell.tolist()
