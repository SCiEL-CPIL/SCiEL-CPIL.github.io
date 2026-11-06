import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    t_start = time.time()
    DEADLINE = 225.0  # Safety margin before 230s kill
    
    def time_left():
        return DEADLINE - (time.time() - t_start)
    
    # --- Symmetry Handling ---
    # The cell is 32x32. Indices 0..31.
    # D4 symmetry: rotations and reflections.
    # We can define a representative set of indices.
    # For a 32x32 grid, the center is at (15.5, 15.5).
    # We map any (i, j) to a canonical (i_can, j_can) in the top-left 1/8th triangle.
    # Actually, simpler: just generate the 8 transformations and take the minimum index or similar?
    # No, better to precompute the mapping for all pixels to a "master" pixel.
    
    # Let's identify the independent pixels.
    # A pixel (i, j) is equivalent to:
    # (i, j), (i, 31-j), (31-i, j), (31-i, 31-j) [Reflections]
    # (j, i), (j, 31-i), (31-j, i), (31-j, 31-i) [Rotations/Swaps]
    
    # We will create a dictionary mapping each pixel to a canonical index.
    # Canonical index: the one with the smallest (i, j) lexicographically among the orbit.
    pixel_map = {}
    for i in range(size):
        for j in range(size):
            orbit = [
                (i, j),
                (i, size-1-j),
                (size-1-i, j),
                (size-1-i, size-1-j),
                (j, i),
                (j, size-1-i),
                (size-1-j, i),
                (size-1-j, size-1-i)
            ]
            # Find the minimum (i, j) pair
            min_idx = min(orbit)
            if min_idx not in pixel_map:
                pixel_map[min_idx] = []
            pixel_map[min_idx].append((i, j))
            
    # List of independent pixels (keys of pixel_map)
    independent_pixels = list(pixel_map.keys())
    n_indep = len(independent_pixels)
    
    # Precompute the full cell from independent values
    def build_cell(indep_vals):
        """
        indep_vals: dict mapping canonical (i,j) to 0/1
        Returns 32x32 numpy array
        """
        cell = np.zeros((size, size), dtype=int)
        for key, val in indep_vals.items():
            for (i, j) in pixel_map[key]:
                cell[i, j] = val
        return cell

    # --- Physics Engine: Plane Wave Method ---
    # We use a subset of G-vectors for speed, but enough for accuracy.
    # Evaluator uses 289 (17x17). We will use 15x15=225 or 13x13=169.
    # Let's try 13x13 (169) for a good balance.
    N_G = 13 
    half = N_G // 2
    
    # G-vectors
    G_list = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_list.append((gx, gy))
    nG = len(G_list)
    
    # Precompute G-k dot products? No, k varies.
    # Precompute G array
    G_arr = np.array(G_list, dtype=float) # (nG, 2)
    
    # K-points in irreducible BZ
    # BZ is [0, 0.5] x [0, 0.5] if we consider the unit cell normalized?
    # Standard: k is in units of 2pi/a. The BZ for square lattice is [-0.5, 0.5]x[-0.5, 0.5].
    # Irreducible BZ: 0 <= ky <= kx <= 0.5.
    # The evaluator uses 30 wave vectors. We will sample the same region.
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # X
    k_points.append((0.5, 0.0))
    # M
    k_points.append((0.5, 0.5))
    # Along Gamma-X
    for k in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((k, 0.0))
    # Along X-M
    for k in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((0.5, k))
    # Along Gamma-M (diagonal)
    for k in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((k, k))
    # Interior points
    k_points.extend([
        (0.3, 0.1), (0.4, 0.1), (0.4, 0.2), (0.3, 0.2),
        (0.2, 0.1), (0.4, 0.3), (0.3, 0.3)
    ])
    
    # Remove duplicates
    k_points = list(set(k_points))
    
    def calculate_gap(cell):
        """
        Calculate band 4-5 gap ratio.
        cell: 32x32 int array (0 or 1)
        """
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps
        
        # Fourier coefficients of inv_eps
        # We need inv_eps_F[(gx, gy)] for gx, gy in range(-half, half+1)
        # FFT gives coefficients for indices 0..N-1.
        # Coeff for (gx, gy) is FFT[(gx % N, gy % N)] / N^2
        inv_eps_fft = np.fft.fft2(inv_eps)
        
        # Precompute matrix of inv_eps_F for all G-G' pairs?
        # G-G' ranges from -12 to 12.
        # We can just compute it on the fly or precompute a lookup.
        # Let's precompute a 2D array of size (2*half+1) x (2*half+1)
        # indexed by [gx+half, gy+half]
        inv_eps_F_mat = np.zeros((N_G, N_G), dtype=complex)
        for ig, (gx, gy) in enumerate(G_list):
            for igp, (gxp, gyp) in enumerate(G_list):
                dgx = gx - gxp
                dgy = gy - gyp
                # Map to FFT index
                idx_x = dgx % size
                idx_y = dgy % size
                inv_eps_F_mat[ig, igp] = inv_eps_fft[idx_x, idx_y] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # Build eigenvalue matrix M
            # M[ig, igp] = (G_kx * Gp_kx + G_ky * Gp_ky) * inv_eps_F_mat[ig, igp]
            # Where G_kx = G_list[ig][0] - kx, etc.
            
            Gk = G_arr - np.array([kx, ky]) # (nG, 2)
            # Dot product (G-k) . (G'-k)
            # This is a matrix product: Gk @ Gk.T
            dot_prod = Gk @ Gk.T # (nG, nG)
            
            M = dot_prod * inv_eps_F_mat
            
            # Solve eigenvalues
            # M is Hermitian
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                # Ensure positive
                eigvals = np.maximum(eigvals, 0)
                omega_4 = np.sqrt(eigvals[3])
                omega_5 = np.sqrt(eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        if min_omega_5 + max_omega_4 == 0:
            return 0.0
            
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio

    # --- Search Strategy ---
    
    # Initial Candidates
    # 1. Solid disk
    # 2. Ring
    # 3. Cross
    
    def init_disk(r):
        vals = {}
        for key in independent_pixels:
            i, j = key
            # Center is 15.5
            dx = i - 15.5
            dy = j - 15.5
            if dx*dx + dy*dy <= r*r:
                vals[key] = 1
            else:
                vals[key] = 0
        return vals

    def init_ring(r_in, r_out):
        vals = {}
        for key in independent_pixels:
            i, j = key
            dx = i - 15.5
            dy = j - 15.5
            d2 = dx*dx + dy*dy
            if r_in*r_in <= d2 <= r_out*r_out:
                vals[key] = 1
            else:
                vals[key] = 0
        return vals

    def init_cross(width):
        vals = {}
        for key in independent_pixels:
            i, j = key
            # Distance to nearest axis (x=15.5 or y=15.5)
            # In pixel coords, axes are between 15 and 16.
            # Let's say if i is in [15, 16] or j is in [15, 16]
            # But we need symmetry.
            # Simpler: if min(|i-15.5|, |j-15.5|) <= width/2
            dx = abs(i - 15.5)
            dy = abs(j - 15.5)
            if min(dx, dy) <= width/2.0:
                vals[key] = 1
            else:
                vals[key] = 0
        return vals

    candidates = [
        init_disk(8),
        init_disk(10),
        init_ring(6, 10),
        init_cross(4),
        init_cross(6),
        # Random
        {key: 1 if np.random.rand() > 0.5 else 0 for key in independent_pixels}
    ]
    
    best_vals = None
    best_score = -float('inf')
    best_cell = None
    
    # Evaluate initial candidates
    for cand in candidates:
        if time_left() < 10:
            break
        cell = build_cell(cand)
        score = calculate_gap(cell)
        if score > best_score:
            best_score = score
            best_vals = cand.copy()
            best_cell = cell.copy()
            
    # Local Optimization: Simulated Annealing / Hill Climbing on Independent Pixels
    # We flip one pixel at a time.
    
    if best_vals is None:
        best_vals = candidates[0].copy()
        best_cell = build_cell(best_vals)
        best_score = calculate_gap(best_cell)
        
    # Sort independent pixels by some heuristic? Random is fine.
    # Use a temperature schedule or simple hill climbing with restarts.
    
    # We will do a few rounds of hill climbing.
    # To speed up, we can pre-compute the effect of flipping a pixel?
    # That's complex. Just re-evaluate.
    # With 169 G-vectors and ~20 k-points, one evaluation is fast (~0.1-0.5s).
    # We have ~200s. So we can do ~500-1000 evaluations.
    
    # Let's do a simple hill climb.
    # Pick a random independent pixel, flip it, evaluate.
    # If better, accept. If worse, accept with probability exp(-delta/T).
    
    T = 0.05
    T_min = 0.001
    cooling = 0.99
    
    current_vals = best_vals.copy()
    current_score = best_score
    
    # Precompute list of independent pixel keys
    keys_list = independent_pixels
    
    while time_left() > 5:
        # Pick a random pixel
        idx = np.random.randint(0, len(keys_list))
        key = keys_list[idx]
        
        # Flip it
        new_vals = current_vals.copy()
        new_vals[key] = 1 - new_vals[key]
        
        # Evaluate
        new_cell = build_cell(new_vals)
        new_score = calculate_gap(new_cell)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < math.exp(delta / T):
            current_vals = new_vals
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_vals = current_vals.copy()
                best_cell = new_cell.copy()
                
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Maybe restart?
            # If no improvement in last 100 steps, reset T?
            
    # Final check
    if best_cell is None:
        best_cell = build_cell(best_vals)
        
    return best_cell.tolist()

