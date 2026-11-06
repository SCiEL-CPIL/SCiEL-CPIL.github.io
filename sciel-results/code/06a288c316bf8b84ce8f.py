"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5.
Strategy: Parametric Fourier-mode synthesis with continuous optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    t_start = time.time()
    DEADLINE = 220.0  # Safety margin before 230s kill

    def time_ok():
        return time.time() - t_start < DEADLINE

    # Plane wave setup: 17x17 = 289 plane waves
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
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2
    
    # Precompute difference indices for Fourier coupling
    # We need eps_inv[G] where G = G_i - G_j
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

    # Precompute the H matrix structure for speed
    # H_ij = (k+G_i)^2 * eps_inv[G_i - G_j]
    # We will compute eps_inv_coeffs for the specific G vectors needed.
    
    # The G vectors we care about for the synthesis are low frequency.
    # But the solver needs all 289 coeffs.
    # We will compute coeffs via FFT from the generated pixel map.
    
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
            # Build H matrix
            # H[i, j] = k2[i] * coeffs[diff_idx[i, j]]
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
        """Enforce 4-fold rotational + reflection symmetry."""
        cell = np.array(cell, dtype=int)
        # 8 symmetries of square
        symmetries = [
            cell,
            np.rot90(cell, 1),
            np.rot90(cell, 2),
            np.rot90(cell, 3),
            cell[:, ::-1].copy(),
            cell[::-1, :].copy(),
            cell.T.copy(),
            np.rot90(cell.T, 1).copy()
        ]
        # Sum and threshold
        avg = np.sum(symmetries, axis=0)
        result = (avg >= 4).astype(int)
        return result

    def generate_cell_from_params(params):
        """
        Generate a 32x32 cell from Fourier mode amplitudes and phases.
        Params: [A_20, ph_20, A_02, ph_02, A_22, ph_22, A_11, ph_11, threshold]
        We construct a field f(x,y) = sum A_mn cos(2*pi*m*x + 2*pi*n*y + ph_mn)
        Then cell = f > threshold.
        Due to symmetry, we only need real, even functions.
        Modes: (2,0), (0,2), (2,2), (1,1) are sufficient for 4-fold symmetric patterns.
        Note: (2,0) and (0,2) are degenerate in square symmetry, so we can combine them.
        """
        A20, ph20, A02, ph02, A22, ph22, A11, ph11, thresh = params
        
        # Grid coordinates
        x = (np.arange(size) + 0.5) / size - 0.5
        y = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Construct field
        # (2,0) mode: cos(4*pi*x + ph20)
        f = A20 * np.cos(4 * pi * X + ph20)
        # (0,2) mode: cos(4*pi*y + ph02)
        f += A02 * np.cos(4 * pi * Y + ph02)
        # (2,2) mode: cos(4*pi*(x+y) + ph22) -- Note: (2,2) is 4-fold symmetric if ph=0 or pi
        # Actually (2,2) corresponds to G=(2,2).
        f += A22 * np.cos(4 * pi * (X + Y) + ph22)
        # (1,1) mode: cos(2*pi*(x+y) + ph11)
        f += A11 * np.cos(2 * pi * (X + Y) + ph11)
        
        # Threshold to binary
        cell = (f > thresh).astype(int)
        
        # Enforce symmetry
        cell = enforce_symmetry(cell)
        return cell

    def objective(params):
        """Negative gap ratio for minimization."""
        if not time_ok():
            return 1e6
        try:
            cell = generate_cell_from_params(params)
            coeffs = get_fourier_coeffs(cell)
            ratio = compute_band_ratio(coeffs)
            return -ratio
        except Exception:
            return 1e6

    # Initial guesses for parameters
    # We want to maximize (2,0) and (0,2) components.
    # Let's try a few starting points.
    
    best_ratio = -np.inf
    best_cell = None
    
    # Define a few initial parameter sets
    # [A20, ph20, A02, ph02, A22, ph22, A11, ph11, thresh]
    initial_guesses = [
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1],
        [0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    ]
    
    for guess in initial_guesses:
        if not time_ok():
            break
            
        # Use Nelder-Mead
        options = {
            'maxiter': 100,
            'xatol': 1e-4,
            'fatol': 1e-4,
            'adaptive': True
        }
        
        try:
            res = minimize(objective, guess, method='Nelder-Mead', options=options)
            if res.success or res.fun < 1e5:
                best_params = res.x
                best_ratio_local = -res.fun
                if best_ratio_local > best_ratio:
                    best_ratio = best_ratio_local
                    best_cell = generate_cell_from_params(best_params)
        except Exception:
            pass

    # If no good result found, fallback to a standard 4-rod pattern
    if best_cell is None or best_ratio < 0:
        # Generate a simple 4-rod pattern as fallback
        def generate_four_rods(radius, offset):
            cell = np.zeros((size, size), dtype=int)
            centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    for cx, cy in centers:
                        dx = x - cx
                        dy = y - cy
                        if dx * dx + dy * dy <= radius * radius:
                            cell[i, j] = 1
                            break
            return enforce_symmetry(cell)
        
        best_cell = generate_four_rods(0.12, 0.25)
        coeffs = get_fourier_coeffs(best_cell)
        best_ratio = compute_band_ratio(coeffs)
        
    # Final refinement: Local pixel flip search on the best cell
    if best_cell is not None and time_ok():
        current_cell = best_cell.copy()
        current_ratio = best_ratio
        
        # Independent pixels: upper-left 16x16
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        
        no_improve_count = 0
        max_no_improve = 30
        search_start = time.time()
        search_time_limit = 180.0  # Leave 40s buffer
        
        while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
            improved = False
            np.random.shuffle(indep_pixels)
            
            for (i, j) in indep_pixels:
                if not time_ok() or time.time() - search_start > search_time_limit:
                    break
                
                old_val = current_cell[i, j]
                new_cell = current_cell.copy()
                new_cell[i, j] = 1 - old_val
                new_cell = enforce_symmetry(new_cell)
                
                coeffs = get_fourier_coeffs(new_cell)
                new_ratio = compute_band_ratio(coeffs)
                
                if new_ratio > current_ratio + 1e-8:
                    current_cell = new_cell
                    current_ratio = new_ratio
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if not time_ok() or time.time() - search_start > search_time_limit:
                    break
            
            if not improved:
                no_improve_count += 1
        
        # Update best
        if current_ratio > best_ratio:
            best_cell = current_cell
            best_ratio = current_ratio

    return best_cell.tolist()
# EVOLVE-BLOCK-END
