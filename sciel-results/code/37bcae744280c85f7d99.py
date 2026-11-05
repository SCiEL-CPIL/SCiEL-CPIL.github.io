"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.linalg

# EVOLVE-BLOCK-START

def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver with higher resolution (17x17 PW, denser k-points).
    2. Generate candidate cells using four-quadrant rod configurations.
    3. Perform a local search (hill climbing) on the best coarse candidate to fine-tune pixels.
    4. Maintain D4 symmetry and respect the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 215.0 # Safety margin for 230s limit

    def check_time():
        return time.time() > deadline - 5.0

    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=17, n_kpts=10):
        """
        Compute TM band structure using plane-wave expansion.
        Returns the gap-midgap ratio between bands 4 and 5.
        """
        cell_arr = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell_arr
        inv_eps = 1.0 / eps
        
        N = size
        # Compute 2D FFT of inv_eps to get Fourier coefficients
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        # Define G vectors
        half = n_pw // 2
        if n_pw % 2 == 1:
            m_vals = np.arange(-half, half + 1)
        else:
            m_vals = np.arange(-half, half)
            
        n_total = len(m_vals) ** 2
        g_indices = [(m, n) for m in m_vals for n in m_vals]
        g_m = np.array([g[0] for g in g_indices])
        g_n = np.array([g[1] for g in g_indices])
        
        # Build S matrix: S[i,j] = inv_eps(G_i - G_j)
        dm = g_m[:, None] - g_m[None, :]
        dn = g_n[:, None] - g_n[None, :]
        idx_m = dm % N
        idx_n = dn % N
        S = fft_coeffs[idx_m, idx_n]
        
        # Define k-points in the irreducible Brillouin zone
        # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5
        k_points = []
        # Gamma to X: (t, 0) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_kpts):
            k_points.append((t, 0.0))
        # X to M: (0.5, t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_kpts):
            k_points.append((0.5, t))
        # M to Gamma: (t, t) for t in [0.5, 0]
        for t in np.linspace(0.5, 0, n_kpts):
            k_points.append((t, t))
            
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        max_omega4 = -np.inf
        min_omega5 = np.inf
        
        # Pre-compute G vectors squared for speed? No, depends on k.
        for kx, ky in k_points_unique:
            if check_time():
                break
            
            # |G_j + k|^2
            gk_m = g_m + kx
            gk_n = g_n + ky
            gk_sq = gk_m**2 + gk_n**2
            
            # A = S * gk_sq[None, :]
            A = S * gk_sq[None, :]
            B = S
            
            try:
                # Solve generalized eigenvalue problem A c = lambda B c
                # B is symmetric positive definite (Fourier coeffs of 1/eps)
                # We can transform to standard eigenvalue problem if B is invertible
                # But scipy.linalg.eigh for generalized problem is efficient
                # Ensure A and B are symmetric
                A = (A + A.T) / 2.0
                B = (B + B.T) / 2.0
                
                vals, _ = scipy.linalg.eigh(A, B)
                vals_real = np.real(vals)
                vals_real = vals_real[np.isfinite(vals_real)]
                vals_sorted = np.sort(vals_real)
                
                if len(vals_sorted) >= 5:
                    omega4 = vals_sorted[3]
                    omega5 = vals_sorted[4]
                    if omega5 > 0 and omega4 > 0:
                        max_omega4 = max(max_omega4, omega4)
                        min_omega5 = min(min_omega5, omega5)
            except Exception:
                pass
        
        if max_omega4 > -np.inf and min_omega5 > 0:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2.0
            if midgap > 0:
                return 2.0 * gap / midgap
            else:
                return -np.inf
        else:
            return -np.inf

    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """
        Create a cell with four rods centered in each quadrant.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        for (cx, cy) in centers:
            dx = x_grid - cx
            dy = y_grid - cy
            if rod_radius > 0:
                val = np.abs(dx/rod_radius)**exponent + np.abs(dy/rod_radius)**exponent
                cell[val <= 1.0 + 1e-9] = 1
        
        # Enforce D4 symmetry
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        
        return cell.tolist()

    # Generate candidates
    candidates = []
    radii = [0.08, 0.10, 0.12, 0.14, 0.16]
    exponents = [2.0, 4.0, 6.0]
    
    for r in radii:
        for n in exponents:
            candidates.append(make_four_quadrant_cell(r, n))
            
    # Add a single central rod as fallback
    def make_central_cell(r, n=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        val = np.abs(x_grid/r)**n + np.abs(y_grid/r)**n
        cell[val <= 1.0 + 1e-9] = 1
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell.tolist()
        
    candidates.append(make_central_cell(0.20, 2.0))

    # Score candidates
    best_score = -np.inf
    best_candidate = candidates[0]
    
    for cand in candidates:
        if check_time():
            break
        score = compute_tm_bands(cand)
        if score > best_score:
            best_score = score
            best_candidate = cand

    # Local search
    if not check_time() and best_score > -np.inf:
        best_arr = np.array(best_candidate)
        improved = True
        stall_count = 0
        max_stall = 50
        
        while improved and not check_time() and stall_count < max_stall:
            improved = False
            # Randomly pick a pixel
            i = np.random.randint(0, size // 2)
            j = np.random.randint(0, size)
            
            # Get the symmetric orbit
            orbit = [(i, j), (j, i), (size-1-i, size-1-j), (size-1-j, size-1-i)]
            
            # Check if all pixels in orbit have the same value
            vals = set(best_arr[p, q] for p, q in orbit)
            if len(vals) != 1:
                stall_count += 1
                continue
            
            # Try flipping
            new_val = 1 - best_arr[i, j]
            for p, q in orbit:
                best_arr[p, q] = new_val
            
            new_cell = best_arr.tolist()
            new_score = compute_tm_bands(new_cell)
            
            if new_score > best_score:
                best_score = new_score
                best_candidate = new_cell
                improved = True
                stall_count = 0
            else:
                # Revert
                for p, q in orbit:
                    best_arr[p, q] = 1 - new_val
                stall_count += 1

    return best_candidate

# EVOLVE-BLOCK-END
