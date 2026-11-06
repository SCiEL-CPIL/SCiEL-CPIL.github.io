import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5.
    Strategy: Single central superellipse with optimized shape exponent and semi-axes,
    followed by pixel-level refinement.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    t_start = time.time()
    DEADLINE = 220.0  # Safety margin before 230s kill
    
    def time_ok():
        return time.time() - t_start < DEADLINE

    # --- Plane Wave Setup ---
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # K-points in irreducible BZ (30 points)
    pi = np.pi
    k_points = []
    # Gamma to X: (t*pi, 0)
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    # X to M: (pi, t*pi)
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    # M to Gamma: (pi*(1-t), pi*(1-t))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2
        
    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)

    # --- Helper Functions ---
    
    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs

    def compute_band_ratio(coeffs):
        """Compute gap-midgap ratio."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            # Build H matrix efficiently
            # H_ij = (k+G_i)^2 * epsilon_{G_i - G_j}
            # Note: The standard eigenvalue problem for photonic crystals is:
            # sum_j (k+G_i)^2 / epsilon_{G_i-G_j} c_j = omega^2 c_i
            # The code uses H = (k+G)^2 * epsilon_coeffs. This implies the matrix is 
            # D * E where D is diagonal (k+G)^2 and E is the Fourier matrix of epsilon.
            # Wait, the standard form is usually (k+G)^2 * (1/epsilon) * c = omega^2 c.
            # Let's check the seed code.
            # Seed: H_vals = k2[valid_rows] * coeffs[diff_idx[...]]
            # coeffs are from 1/eps.
            # So H_ij = (k+G_i)^2 * (1/epsilon)_{G_i-G_j}.
            # This is the correct operator for TM modes: (k+G)^2 / epsilon * E = omega^2 E.
            
            H_vals = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                if eigvals[3] > max_band4:
                    max_band4 = eigvals[3]
                if eigvals[4] < min_band5:
                    min_band5 = eigvals[4]
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def enforce_symmetry(cell):
        """Enforce 4-fold rotational + reflection symmetry (D4)."""
        cell = np.array(cell, dtype=int)
        # 8 symmetries of the square
        symmetries = [
            cell,
            np.rot90(cell, 1),
            np.rot90(cell, 2),
            np.rot90(cell, 3),
            cell[:, ::-1],
            cell[::-1, :],
            cell.T,
            np.rot90(cell.T, 1)
        ]
        # Average and threshold
        # Since pixels are 0/1, we can just check if any symmetry is 1? 
        # No, to maintain strict symmetry, the pixel must be 1 in ALL symmetries 
        # or 0 in ALL symmetries if we want to project.
        # Actually, the standard projection is: if the pixel is 1 in the "master" quadrant,
        # it must be 1 in all corresponding positions.
        # However, simple averaging and thresholding > 0.5 works for binary inputs 
        # if we assume the input is already "mostly" symmetric or we force it.
        # Let's use the rule: Pixel is 1 if ANY of its symmetric counterparts is 1?
        # No, that breaks the "single shape" idea if the shape is not perfectly symmetric in the discrete grid.
        # Better: Define the cell by the top-left 16x16 quadrant and reflect/rotate it.
        # This guarantees symmetry.
        
        # Extract top-left 16x16
        # Note: Center of 32x32 is between (15,15), (15,16), (16,15), (16,16).
        # Due to symmetry, we can define the first 16x16 and map it.
        # Indices: 0..15
        # (i, j) -> (15-i, 15-j) is 180 rot.
        # (i, j) -> (j, i) is transpose.
        # (i, j) -> (15-j, 15-i) is rot90.
        
        # Let's just build it from the first quadrant to ensure perfect symmetry.
        # This function takes a full cell, extracts the unique part, and rebuilds.
        # For refinement, we will modify the unique part directly.
        
        # Rebuild from top-left 16x16
        quad = cell[:16, :16].copy()
        new_cell = np.zeros((size, size), dtype=int)
        
        for i in range(16):
            for j in range(16):
                val = quad[i, j]
                # 8 positions
                new_cell[i, j] = val
                new_cell[15-i, 15-j] = val # 180
                new_cell[j, i] = val       # Transpose
                new_cell[15-j, 15-i] = val # Rot 90 (i->j, j->15-i? No. (x,y)->(y,-x). Center 0. 
                                           # Pixel (i,j) in 0..15 maps to (j, 15-i) in 0..15?
                                           # Let's verify rot90 on 32x32.
                                           # np.rot90 maps (r,c) to (c, N-1-r).
                                           # If we define based on top-left 16x16, we need to be careful.
        
        # Easier: Use the averaging method but force strict symmetry by taking the max or min?
        # If we start from a symmetric generation, averaging is fine.
        # If we do pixel flips, we must enforce symmetry explicitly.
        
        # Let's stick to the generation strategy: Generate a symmetric cell from parameters.
        # For pixel refinement, we will flip pixels in the "independent" set and then
        # re-apply the symmetry mapping to the whole cell.
        
        # Independent pixels: Top-left 16x16 quadrant.
        # Mapping:
        # (i, j) -> (i, j)
        # (i, j) -> (15-i, 15-j)  (180 deg)
        # (i, j) -> (j, i)        (Reflection across y=x)
        # (i, j) -> (15-j, 15-i)  (Reflection across y=-x+15? No, Rot 90)
        # Let's check Rot 90 of (i,j) in 32x32.
        # (r, c) -> (c, 31-r).
        # If r<16, c<16.
        # New r = c (0..15). New c = 31-r (16..31).
        # This maps top-left to top-right.
        # So the 8 symmetries map the top-left quadrant to all 8 octants.
        
        # Let's define the 8 mappings from (i,j) in [0,15]x[0,15] to [0,31]x[0,31]
        # 1. Identity: (i, j)
        # 2. Rot 90: (j, 31-i)
        # 3. Rot 180: (31-i, 31-j)
        # 4. Rot 270: (31-j, i)
        # 5. Reflect X (H-flip): (i, 31-j)
        # 6. Reflect Y (V-flip): (31-i, j)
        # 7. Transpose: (j, i)
        # 8. Anti-transpose: (31-j, 31-i) is same as Rot 180 of Transpose?
        # Actually, D4 group has 8 elements.
        # Let's just use the 8 standard operations on the full grid to check symmetry,
        # but for generation/refinement, we will define the cell by its top-left 16x16 
        # and fill the rest.
        
        # Mapping from (i,j) in 0..15 to the 8 positions:
        # p1 = (i, j)
        # p2 = (i, 31-j)  # H-flip
        # p3 = (31-i, j)  # V-flip
        # p4 = (31-i, 31-j) # 180
        # p5 = (j, i)     # Transpose
        # p6 = (j, 31-i)  # Rot 90 (of p1? No. Rot 90 of (i,j) is (j, 31-i). Yes.)
        # p7 = (31-j, i)  # Rot 270 (of p1? Rot 270 of (i,j) is (31-j, i). Yes.)
        # p8 = (31-j, 31-i) # Anti-transpose (or Rot 180 of Transpose)
        
        # We will use this mapping in the refinement step.
        return cell

    def generate_superellipse(a, b, n_exp):
        """
        Generate a cell with a single central superellipse.
        Equation: |x/a|^n + |y/b|^n <= 1
        Centered at (15.5, 15.5) in pixel coords? 
        Pixel centers are (i+0.5). Center of cell is 16.0 (between 15.5 and 16.5? No, 32 pixels, center is 16.0).
        Coords: (i - 15.5) / 32 ?
        Let's use normalized coords in [-0.5, 0.5].
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        # a, b are in normalized units (fraction of cell size)
        # Equation: |X/a|^n + |Y/b|^n <= 1
        # Handle a,b > 0
        if a < 1e-5 or b < 1e-5:
            return cell
            
        val = np.abs(X / a) ** n_exp + np.abs(Y / b) ** n_exp
        cell[val <= 1.0] = 1
        
        return cell

    def enforce_strict_symmetry_from_quad(quad):
        """
        Takes a 16x16 array and returns a 32x32 array with D4 symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = quad[i, j]
                cell[i, j] = val
                cell[i, 31-j] = val
                cell[31-i, j] = val
                cell[31-i, 31-j] = val
                cell[j, i] = val
                cell[j, 31-i] = val
                cell[31-j, i] = val
                cell[31-j, 31-i] = val
        return cell

    # --- Optimization ---
    
    best_ratio = -np.inf
    best_cell = None
    best_params = None
    
    # Phase 1: Coarse Grid Search
    # Parameters: a (semi-axis x), b (semi-axis y), n (exponent)
    # Due to symmetry, we can assume a >= b or just search a, b in a range.
    # a, b in [0.05, 0.45] (normalized)
    # n in [1.0, 10.0]
    
    a_vals = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    b_vals = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    n_vals = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0]
    
    for a in a_vals:
        for b in b_vals:
            for n in n_vals:
                if not time_ok():
                    break
                cell = generate_superellipse(a, b, n)
                if cell.sum() == 0 or cell.sum() == size*size:
                    continue
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell.copy()
                    best_params = (a, b, n)
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
            
    if best_cell is None:
        # Fallback
        best_cell = generate_superellipse(0.25, 0.25, 2.0)
        best_params = (0.25, 0.25, 2.0)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))

    # Phase 2: Local Optimization (Nelder-Mead or Random Perturbation)
    # We will use a random perturbation hill-climbing with adaptive step size.
    current_a, current_b, current_n = best_params
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    search_start = time.time()
    search_time_limit = 100.0
    
    no_improve_count = 0
    max_no_improve = 50
    step_size = 0.02
    
    while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
        # Perturb parameters
        new_a = current_a * (1.0 + np.random.normal(0, step_size))
        new_b = current_b * (1.0 + np.random.normal(0, step_size))
        new_n = current_n + np.random.normal(0, step_size * current_n)
        
        # Clamp
        new_a = np.clip(new_a, 0.05, 0.45)
        new_b = np.clip(new_b, 0.05, 0.45)
        new_n = np.clip(new_n, 1.0, 20.0)
        
        new_cell = generate_superellipse(new_a, new_b, new_n)
        coeffs = get_fourier_coeffs(new_cell)
        new_ratio = compute_band_ratio(coeffs)
        
        if new_ratio > current_ratio + 1e-6:
            current_cell = new_cell
            current_ratio = new_ratio
            current_a, current_b, current_n = new_a, new_b, new_n
            no_improve_count = 0
            # Optionally reduce step size slightly if many improvements
            step_size *= 0.95
        else:
            no_improve_count += 1
            # Increase step size if stuck? Or keep it?
            if no_improve_count > 20:
                step_size *= 1.05

    # Phase 3: Pixel Refinement
    # Refine the boundary of the superellipse by flipping pixels in the top-left 16x16 quadrant.
    # This allows for "pixelated" shapes that might capture the gap better than the smooth analytic boundary.
    
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 40:
        # Extract the current top-left 16x16
        quad = current_cell[:16, :16].copy()
        
        # List of pixels to try flipping. 
        # Focus on pixels near the boundary of the shape.
        # A pixel is "boundary" if it differs from any of its 4-neighbors.
        boundary_pixels = []
        for i in range(16):
            for j in range(16):
                # Check neighbors in the full 32x32 context? 
                # Simpler: Just try flipping all pixels in the quadrant, but prioritize those 
                # that are "on the edge" of the current dielectric region.
                # Edge if any neighbor is different.
                is_edge = False
                # Neighbors in quad
                if i > 0 and quad[i-1, j] != quad[i, j]: is_edge = True
                if i < 15 and quad[i+1, j] != quad[i, j]: is_edge = True
                if j > 0 and quad[i, j-1] != quad[i, j]: is_edge = True
                if j < 15 and quad[i, j+1] != quad[i, j]: is_edge = True
                # Also check if it's adjacent to the symmetry axis (i=15 or j=15)
                # If i=15, neighbor (16, j) in full cell is (15, j) reflected? 
                # Actually, just checking 4-neighbors in the 16x16 quad is a good proxy.
                
                if is_edge:
                    boundary_pixels.append((i, j))
        
        # Shuffle to randomize order
        np.random.shuffle(boundary_pixels)
        
        refine_start = time.time()
        refine_limit = 30.0
        no_improve_refine = 0
        max_no_improve_refine = 100
        
        # We might need to try flipping multiple times or use a simple local search.
        # Let's iterate through the list, flipping and checking.
        # If improvement, keep it. If not, revert.
        # Then repeat the list until no improvement or time out.
        
        while time.time() - refine_start < refine_limit and no_improve_refine < max_no_improve_refine and time_ok():
            improved_in_pass = False
            for (i, j) in boundary_pixels:
                if not time_ok(): break
                
                old_val = quad[i, j]
                new_val = 1 - old_val
                quad[i, j] = new_val
                
                # Reconstruct full cell
                new_cell = enforce_strict_symmetry_from_quad(quad)
                
                coeffs = get_fourier_coeffs(new_cell)
                new_ratio = compute_band_ratio(coeffs)
                
                if new_ratio > current_ratio + 1e-8:
                    current_cell = new_cell
                    current_ratio = new_ratio
                    improved_in_pass = True
                    no_improve_refine = 0
                else:
                    # Revert
                    quad[i, j] = old_val
                    no_improve_refine += 1
                    if no_improve_refine > max_no_improve_refine:
                        break
            if not improved_in_pass:
                break

    # Final Verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    if final_ratio < best_ratio - 0.01:
        current_cell = best_cell
    else:
        best_ratio = final_ratio

    return current_cell.tolist()
# EVOLVE-BLOCK-END
