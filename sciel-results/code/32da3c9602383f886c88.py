import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Use a fast internal PWE solver with a smaller basis (13x13=169 PWs) for quick evaluation.
    2. Generate diverse candidate cells: four-quadrant rods, central rod, and ring patterns.
    3. Enforce exact D4 symmetry by constructing from a single quadrant and mirroring.
    4. Perform a local search around the best candidate found.
    5. Maintain strict time budget with early termination.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 10s margin for overhead
    start_time = time.time()
    deadline = start_time + 220.0
    
    def time_left():
        return deadline - time.time()
    
    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=8):
        """
        Compute TM band structure using plane-wave expansion.
        
        Parameters:
        -----------
        cell : 2D array of 0/1 (0=air, 1=dielectric)
        n_pw : number of plane waves in each direction (total = n_pw^2)
        n_kpts : number of k-points along the irreducible BZ path
        
        Returns:
        --------
        omega : array of shape (n_kpts, n_bands) with band frequencies
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell  # epsilon map
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/epsilon using FFT
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(m, n):
            """Get Fourier coefficient for G = 2*pi*(m, n)"""
            idx_m = m % size
            idx_n = n % size
            return fft_coeffs[idx_m, idx_n]
        
        # Define the set of G vectors to include in the basis.
        # Use a circular cutoff for better accuracy - increased cutoff
        M = n_pw // 2
        g_indices = []
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                if m*m + n*n <= (M + 1)*(M + 1):  # slightly larger circular cutoff
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        # Build the S matrix: S_ij = inv_eps(G_i - G_j)
        # Pre-compute all S matrix entries
        S = np.zeros((n_total, n_total), dtype=complex)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                S[i, j] = get_coeff(gm, gn)
        
        # Define k-points in the irreducible Brillouin zone
        # High symmetry points: Gamma(0,0), X(0.5,0), M(0.5,0.5) in units of 2*pi
        # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5
        # Path: Gamma -> X -> M -> Gamma
        k_points = []
        # Gamma to X: (t, 0) for t in [0, 0.5]
        n_seg = max(3, n_kpts // 3)
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((t, 0))
        # X to M: (0.5, t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((0.5, t))
        # M to Gamma: (t, t) for t in [0.5, 0]
        for t in np.linspace(0.5, 0, n_seg):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        
        # Pre-compute G vectors
        G_vecs = np.array([(2.0 * np.pi * m, 2.0 * np.pi * n) for m, n in g_indices])
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            # |G_j + k|^2 for all j
            Gk_x = G_vecs[:, 0] + 2.0 * np.pi * kx
            Gk_y = G_vecs[:, 1] + 2.0 * np.pi * ky
            k_mag_sq = Gk_x**2 + Gk_y**2
            
            # Build the matrix M = S * diag(k_mag_sq)
            # M[i, j] = S[i, j] * k_mag_sq[j]
            M = S * k_mag_sq[np.newaxis, :]
            
            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.sort(eigvals)
                # Take real positive parts
                eigvals = np.real(eigvals)
                eigvals = np.maximum(eigvals, 0)
                omega[ki, :] = np.sqrt(eigvals)
            except Exception:
                omega[ki, :] = np.inf
        
        return omega

    # Enforce exact D4 symmetry by constructing from top-left quadrant
    def enforce_d4_symmetry(cell):
        """
        Enforce D4 symmetry: cell == cell[::-1] row-wise and column-wise, and cell == its transpose.
        We take the top-left quadrant and mirror it.
        """
        cell = np.array(cell, dtype=int)
        half = size // 2
        # Take top-left quadrant
        tl = cell[:half, :half].copy()
        # Construct full cell
        cell_new = np.zeros((size, size), dtype=int)
        # Top-left
        cell_new[:half, :half] = tl
        # Top-right: mirror horizontally
        cell_new[:half, half:] = tl[:, ::-1]
        # Bottom-left: mirror vertically
        cell_new[half:, :half] = tl[::-1, :]
        # Bottom-right: mirror both
        cell_new[half:, half:] = tl[::-1, ::-1]
        return cell_new

    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """
        Create a cell with four rods centered in each quadrant.
        Enforces D4 symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        # Rod centers in normalized coordinates [-0.5, 0.5]
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        
        return enforce_d4_symmetry(cell).tolist()

    def make_central_ring_cell(inner_radius, outer_radius, exponent=2.0):
        """
        Create a cell with a central ring (annulus).
        Enforces D4 symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        R = np.sqrt(X**2 + Y**2)
        
        # Ring: inner_radius <= R <= outer_radius
        if outer_radius > 0 and inner_radius >= 0:
            val = np.abs(R/outer_radius)**exponent - np.abs(inner_radius/outer_radius)**exponent
            cell[val >= 0] = 1
        
        return enforce_d4_symmetry(cell).tolist()

    def make_staggered_rods_cell(rod_radius, exponent=2.0, offset=0.15):
        """
        Create a cell with staggered rods.
        Enforces D4 symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        # Rods at positions that maintain D4 symmetry
        # Centers: (±offset, ±offset) and (±(0.5-offset), ±(0.5-offset))
        centers = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                centers.append((sx*offset, sy*offset))
                centers.append((sx*(0.5-offset), sy*(0.5-offset)))
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        
        return enforce_d4_symmetry(cell).tolist()

    def score_cell(cell, n_pw=13, n_kpts=8):
        """
        Score a cell by computing the gap-midgap ratio between bands 4 and 5.
        Uses a smoother scoring function to guide optimization.
        """
        try:
            bands = compute_tm_bands(cell, n_pw=n_pw, n_kpts=n_kpts)
            # Extract bands 4 and 5 (index 3 and 4)
            omega_4 = bands[:, 3]
            omega_5 = bands[:, 4]
            
            max_omega_4 = np.max(omega_4)
            min_omega_5 = np.min(omega_5)
            
            if min_omega_5 > max_omega_4:
                gap = min_omega_5 - max_omega_4
                ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
            else:
                # Use a smooth penalty instead of hard -1.0
                overlap = max_omega_4 - min_omega_5
                midgap = 0.5 * (min_omega_5 + max_omega_4)
                ratio = -2.0 * overlap / midgap if midgap > 0 else -1.0
            return ratio
        except Exception:
            return -1.0

    # Generate and evaluate candidates
    best_score = -np.inf
    best_cell = None
    
    # Phase 1: Grid search over four-quadrant rods - expanded range
    radii = np.linspace(0.06, 0.20, 15)
    exponents = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    
    for r in radii:
        if time_left() < 15:
            break
        for exp in exponents:
            if time_left() < 5:
                break
            cell = make_four_quadrant_cell(r, exp)
            score = score_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Phase 2: Try central rings - expanded range
    if time_left() > 15:
        inner_radii = np.linspace(0.03, 0.18, 9)
        outer_radii = np.linspace(0.12, 0.28, 9)
        for ir in inner_radii:
            if time_left() < 10:
                break
            for orad in outer_radii:
                if time_left() < 5:
                    break
                if orad <= ir:
                    continue
                cell = make_central_ring_cell(ir, orad)
                score = score_cell(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Phase 3: Try staggered rods - expanded range
    if time_left() > 15:
        radii_s = np.linspace(0.04, 0.16, 11)
        offsets = np.linspace(0.08, 0.22, 9)
        for r in radii_s:
            if time_left() < 10:
                break
            for off in offsets:
                if time_left() < 5:
                    break
                cell = make_staggered_rods_cell(r, 2.0, off)
                score = score_cell(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Phase 4: Local refinement around the best candidate - expanded search
    if best_cell is not None and time_left() > 20:
        # Try small perturbations of the best cell
        best_arr = np.array(best_cell, dtype=int)
        half = size // 2
        tl_best = best_arr[:half, :half].copy()
        
        # Try flipping individual pixels in the top-left quadrant
        # Focus on pixels near the boundary of dielectric regions
        candidates = []
        for i in range(half):
            for j in range(half):
                # Check if this pixel is near a boundary
                neighbors = []
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < half and 0 <= nj < half:
                        neighbors.append(tl_best[ni, nj])
                if neighbors and tl_best[i, j] != np.mean(neighbors):
                    candidates.append((i, j))
        
        # Also add all pixels as candidates (for broader exploration)
        all_pixels = [(i, j) for i in range(half) for j in range(half)]
        candidates = candidates + all_pixels
        
        # Shuffle and try up to 150 perturbations
        np.random.seed(42)
        np.random.shuffle(candidates)
        
        for idx, (i, j) in enumerate(candidates[:150]):
            if time_left() < 5:
                break
            # Flip the pixel and its mirrors
            tl_new = tl_best.copy()
            tl_new[i, j] = 1 - tl_new[i, j]
            
            # Reconstruct full cell
            cell_new = np.zeros((size, size), dtype=int)
            cell_new[:half, :half] = tl_new
            cell_new[:half, half:] = tl_new[:, ::-1]
            cell_new[half:, :half] = tl_new[::-1, :]
            cell_new[half:, half:] = tl_new[::-1, ::-1]
            
            score = score_cell(cell_new.tolist())
            if score > best_score:
                best_score = score
                best_cell = cell_new.tolist()
                # Update tl_best to the new best
                tl_best = cell_new[:half, :half].copy()
        
        # Phase 5: Second round of local search with more aggressive flips
        if time_left() > 10:
            # Try flipping pairs of adjacent pixels
            for i in range(half - 1):
                if time_left() < 5:
                    break
                for j in range(half - 1):
                    if time_left() < 3:
                        break
                    # Try flipping 2x2 block
                    tl_new = tl_best.copy()
                    tl_new[i, j] = 1 - tl_new[i, j]
                    tl_new[i+1, j] = 1 - tl_new[i+1, j]
                    tl_new[i, j+1] = 1 - tl_new[i, j+1]
                    tl_new[i+1, j+1] = 1 - tl_new[i+1, j+1]
                    
                    cell_new = np.zeros((size, size), dtype=int)
                    cell_new[:half, :half] = tl_new
                    cell_new[:half, half:] = tl_new[:, ::-1]
                    cell_new[half:, :half] = tl_new[::-1, :]
                    cell_new[half:, half:] = tl_new[::-1, ::-1]
                    
                    score = score_cell(cell_new.tolist())
                    if score > best_score:
                        best_score = score
                        best_cell = cell_new.tolist()
                        tl_best = cell_new[:half, :half].copy()
    
    # If no cell was found, return a default
    if best_cell is None:
        best_cell = make_four_quadrant_cell(0.12, 2.0)
    
    # Final validation: ensure D4 symmetry
    best_arr = np.array(best_cell, dtype=int)
    best_arr = enforce_d4_symmetry(best_arr)
    
    return best_arr.tolist()

# EVOLVE-BLOCK-END
