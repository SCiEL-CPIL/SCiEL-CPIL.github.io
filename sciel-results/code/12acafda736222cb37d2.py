import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = 210.0  # Safety margin
    
    def time_ok():
        return time.time() - start_time < deadline

    # --- Plane Wave Setup ---
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw
    
    # K-points in irreducible BZ
    pi = np.pi
    k_points = []
    # Gamma to X
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    # X to M
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    # M to Gamma
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2
        
    # Precompute difference indices for Fourier coupling
    # We need coefficients of 1/eps. 
    # Let's precompute the matrix structure.
    # H_{ij} = (k+G_i)^2 * (1/eps)_{G_i - G_j}
    
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    valid_mask = np.zeros((n_pw_total, n_pw_total), dtype=bool)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
                valid_mask[i, j] = True
            else:
                diff_idx[i, j] = -1
                
    valid_rows, valid_cols = np.where(valid_mask)
    valid_diff_indices = diff_idx[valid_rows, valid_cols]
    
    # --- Parameterization ---
    # We model the unit cell using a Fourier series.
    # To maintain D4 symmetry (square lattice), we include modes (m, n) such that:
    # The set of modes is invariant under (m,n) -> (-m,n), (m,-n), (n,m) etc.
    # For simplicity and robustness, we will use a set of low-order modes.
    # The dielectric constant epsilon(x,y) is real and positive.
    # We parameterize 1/epsilon directly to match the PW method requirement?
    # Actually, the PW method for TM uses 1/epsilon in the matrix.
    # Let's parameterize the pixel values (0 or 1) via a sigmoid of a Fourier sum.
    # This ensures smoothness and symmetry.
    
    # Symmetric modes for D4:
    # (0,0)
    # (4,0), (0,4) -> invariance under 90 deg rotation requires careful combination?
    # Actually, if we just enforce the pixel symmetry at the end, we can use any modes.
    # But using symmetric modes is more efficient.
    
    # Let's use a set of modes that naturally respect the 32x32 grid.
    # Modes (m, n) where m, n are multiples of 1, but we group them.
    # To ensure the output is exactly symmetric (cell == cell[::-1]), we can just
    # enforce it after generation.
    
    # Parameters: Amplitudes of Fourier modes for the "potential" V(x,y).
    # Pixel = 1 if V(x,y) > 0 else 0.
    
    # We select a set of (m, n) pairs.
    # To allow complex shapes (like 4 rods), we need higher harmonics.
    # 32x32 grid -> fundamental frequency is 2*pi/32.
    # Modes m, n in [-10, 10] are sufficient.
    
    # Let's pick specific modes that are D4 symmetric.
    # A function f(x,y) is D4 symmetric if f(x,y) = f(y,x) = f(-x,y) etc.
    # Cosine terms cos(m x + n y) + cos(m y + n x) + ...
    
    # For simplicity, let's use a smaller set of parameters and optimize them.
    # We will use a "blob" model: 4 circles at (±d, ±d) with radius r.
    # This is a 3-parameter model: r, d, and maybe a shape distortion.
    # But the prompt suggests "fractal boundary" or "high-order harmonics".
    # Let's use a Fourier expansion of the boundary radius r(theta).
    
    # Model: 4 rods centered at (±d, ±d).
    # Each rod is a disk with radius r(theta) = R * (1 + sum A_k cos(k*theta)).
    # To maintain D4 symmetry of the whole cell, the rods must be identical.
    
    # Parameters:
    # p[0]: offset d (0.1 to 0.4)
    # p[1]: base radius R (0.05 to 0.3)
    # p[2]: A_4 (amplitude of 4th harmonic - squareness)
    # p[3]: A_8 (amplitude of 8th harmonic - more complex shape)
    # p[4]: A_12 (amplitude of 12th harmonic)
    
    def generate_cell_from_params(params):
        d = np.clip(params[0], 0.1, 0.4)
        R = np.clip(params[1], 0.05, 0.35)
        A4 = np.clip(params[2], -0.5, 0.5)
        A8 = np.clip(params[3], -0.5, 0.5)
        A12 = np.clip(params[4], -0.5, 0.5)
        
        cell = np.zeros((size, size), dtype=int)
        centers = [
            (d, d),
            (-d, d),
            (d, -d),
            (-d, -d)
        ]
        
        # Precompute theta and r for all pixels for speed?
        # Or just loop. 32x32 is small.
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                
                inside = False
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    r_pix = np.sqrt(dx*dx + dy*dy)
                    if r_pix < 1e-6:
                        inside = True
                        break
                    theta = np.arctan2(dy, dx)
                    # Modulate radius
                    r_mod = R * (1.0 + A4 * np.cos(4*theta) + A8 * np.cos(8*theta) + A12 * np.cos(12*theta))
                    if r_pix < r_mod:
                        inside = True
                        break
                
                if inside:
                    cell[i, j] = 1
        
        # Enforce strict symmetry (just in case of floating point errors)
        # The construction is symmetric, but let's be safe.
        cell = (cell + cell[::-1, ::-1] + cell[:, ::-1] + cell[::-1, :]) // 4
        cell = (cell > 0.5).astype(int)
        return cell

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

    def compute_band_ratio_full(coeffs):
        """Compute gap-midgap ratio using all 30 k-points and 289 PWs."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            # H is Hermitian
            # H_{ij} = k2_i * c_{G_i - G_j}
            # We only need to fill the upper triangle and symmetrize?
            # Or just fill all valid ones.
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[valid_diff_indices]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def compute_band_ratio_fast(coeffs):
        """Fast approximate evaluation using subset of k-points."""
        k_subset = [0, 4, 9, 14, 19, 24, 29]
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_subset:
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[valid_diff_indices]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def objective(params):
        if not time_ok():
            return 0.0
        cell = generate_cell_from_params(params)
        coeffs = get_fourier_coeffs(cell)
        # Use fast eval for optimization
        return -compute_band_ratio_fast(coeffs)

    # --- Optimization ---
    
    # Initial guesses
    # 1. 4 small rods
    p1 = [0.25, 0.10, 0.0, 0.0, 0.0]
    # 2. 4 larger rods
    p2 = [0.20, 0.20, 0.0, 0.0, 0.0]
    # 3. Square-ish rods (A4 positive makes it square-ish? cos(4theta) peaks at 0, pi/2)
    # At theta=0 (x-axis), cos(0)=1. At theta=pi/4, cos(pi)=-1.
    # So positive A4 makes it elongated along axes (diamond shape relative to grid? No, axes are x,y).
    # Rods are at (d,d). Theta is local.
    p3 = [0.25, 0.15, 0.2, 0.0, 0.0]
    p4 = [0.25, 0.15, -0.2, 0.0, 0.0]
    
    best_params = None
    best_score = -np.inf
    best_cell = None
    
    for p0 in [p1, p2, p3, p4]:
        if not time_ok():
            break
            
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is not None:
        best_cell = generate_cell_from_params(best_params)
    else:
        # Fallback
        best_cell = generate_cell_from_params(p1)

    # --- Refinement ---
    # The fast evaluator might be slightly off. Let's do a local search on pixels
    # or a finer grid of parameters around the best found.
    
    # Let's try a few more specific parameter variations around the best found
    if best_params is not None:
        base_d, base_R, base_A4, base_A8, base_A12 = best_params
        
        candidates = [
            base_params,
            [base_d + 0.02, base_R, base_A4, base_A8, base_A12],
            [base_d - 0.02, base_R, base_A4, base_A8, base_A12],
            [base_d, base_R + 0.02, base_A4, base_A8, base_A12],
            [base_d, base_R - 0.02, base_A4, base_A8, base_A12],
            [base_d, base_R, base_A4 + 0.1, base_A8, base_A12],
            [base_d, base_R, base_A4 - 0.1, base_A8, base_A12],
            [base_d, base_R, base_A4, base_A8 + 0.1, base_A12],
            [base_d, base_R, base_A4, base_A8 - 0.1, base_A12],
            [base_d, base_R, base_A4, base_A8, base_A12 + 0.1],
            [base_d, base_R, base_A4, base_A8, base_A12 - 0.1],
        ]
        
        for p in candidates:
            if not time_ok():
                break
            cell = generate_cell_from_params(p)
            coeffs = get_fourier_coeffs(cell)
            score = compute_band_ratio_full(coeffs)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_params = p

    # --- Local Pixel Search ---
    # Sometimes pixel-level adjustments help.
    current_cell = best_cell.copy()
    current_score = best_score
    
    # Identify independent pixels (1/4 of the grid)
    indep_pixels = [(i, j) for i in range(size // 2) for j in range(size // 2)]
    
    no_improve = 0
    max_no_improve = 20
    threshold = 1e-5
    
    # Shuffle to randomize order
    np.random.shuffle(indep_pixels)
    
    # Limit iterations to fit in time
    max_iterations = 50
    iter_count = 0
    
    while iter_count < max_iterations and no_improve < max_no_improve and time_ok():
        improved = False
        for (i, j) in indep_pixels:
            if not time_ok() or time.time() - start_time > deadline - 10:
                break
            
            old_val = current_cell[i, j]
            current_cell[i, j] = 1 - old_val
            current_cell = (current_cell + current_cell[::-1, ::-1] + current_cell[:, ::-1] + current_cell[::-1, :]) // 4
            current_cell = (current_cell > 0.5).astype(int)
            
            coeffs = get_fourier_coeffs(current_cell)
            new_score = compute_band_ratio_full(coeffs)
            
            if new_score > current_score + threshold:
                current_score = new_score
                improved = True
                no_improve = 0
            else:
                # Revert
                current_cell[i, j] = old_val
                current_cell = (current_cell + current_cell[::-1, ::-1] + current_cell[:, ::-1] + current_cell[::-1, :]) // 4
                current_cell = (current_cell > 0.5).astype(int)
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
        
        if not improved:
            no_improve += 1
            
        iter_count += 1
        np.random.shuffle(indep_pixels)

    return current_cell.tolist()

