import time
import math
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 pixel unit cell for a 2D square-lattice photonic crystal.
    Strategy:
    1. Start from a high-quality initial guess: a square frame (dielectric square with central air hole)
       or a cross shape, known to open TM band gaps.
    2. Use a fast approximate plane-wave method (smaller basis) to evaluate the gap.
    3. Perform stochastic local search (hill climbing) with pixel flips, respecting C4v symmetry.
    4. Use a deadline guard to stop before the 120s CPU budget.
    
    The key insight is that the gap between bands 4 and 5 is maximized by structures with
    strong anisotropy or specific symmetries that split degeneracies. A square frame with
    optimized hole size is a good starting point.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    lower_band = problem["lower_band"]
    polarization = problem["polarization"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Center of the cell
    center = (size - 1) / 2.0
    
    # --- Fast approximate band structure calculation ---
    # We use a plane-wave expansion with a smaller basis for speed.
    # The evaluator uses 289 plane waves (17x17). We use 25 (5x5) or 49 (7x7) for speed.
    # The gap ratio should correlate.
    
    def compute_gap_ratio(cell, n_pw=7):
        """
        Compute the gap-midgap ratio between bands 4 and 5 using a fast PW method.
        cell: 2D numpy array of 0/1
        n_pw: number of plane waves in each direction (odd)
        Returns the ratio, or a negative value if bands overlap.
        """
        # Build the dielectric permittivity grid
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        
        # Compute Fourier coefficients of 1/eps for TM mode
        # For TM: (1/eps) * laplacian
        # We need the inverse permittivity: 1/eps
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        # Fourier coefficients of inv_eps
        # G = (m, n) * 2*pi / a, where a is lattice constant (we set a=1)
        # For a 32x32 pixel cell, the pixel size is 1/32
        # The Fourier coefficients are:
        # f(G) = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i*G*x_i)
        
        # Plane wave indices
        # We use a symmetric set of plane waves
        pw_indices = list(range(-n_pw//2, n_pw//2 + 1))
        n_pw_actual = len(pw_indices)
        
        # Precompute exp terms for efficiency
        # x_i = (i + 0.5) / size, for i in 0..size-1
        x_coords = (np.arange(size) + 0.5) / size
        y_coords = x_coords  # same for y
        
        # Compute Fourier coefficients
        # f(m, n) = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i*2*pi*(m*x_i + n*y_j))
        # Use FFT for efficiency
        # Pad inv_eps to size x size (already is)
        # FFT of inv_eps gives coefficients at frequencies 0, 1, ..., N-1
        # We need coefficients at frequencies m, n where m, n are in pw_indices
        # Note: FFT gives f(k) = sum_{i} f_i * exp(-i*2*pi*k*i/N)
        # Our x_i = (i+0.5)/N, so exp(-i*2*pi*m*(i+0.5)/N) = exp(-i*pi*m/N) * exp(-i*2*pi*m*i/N)
        # So f(m) = exp(-i*pi*m/N) * FFT(inv_eps)[m mod N] / N^2
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        # Map plane wave indices to FFT indices
        # FFT index k corresponds to frequency k if k < N/2, or k-N if k >= N/2
        # We want frequencies m in pw_indices
        def get_fft_coeff(m, n):
            # m, n are the desired frequencies
            # FFT index: m mod N
            idx_m = m % size
            idx_n = n % size
            # Phase correction: exp(-i*pi*(m/n)/N)
            phase = np.exp(-1j * np.pi * (m + n) / size)
            return phase * fft_inv_eps[idx_m, idx_n] / (size * size)
        
        # Build the matrix for each k-point
        # The TM eigenvalue problem: (1/eps) * (-k^2) * c = omega^2 * c
        # In PW basis: sum_{G'} (1/eps)_{G-G'} * (k+G)^2 * c_{G'} = omega^2 * c_G
        # Where (1/eps)_{G-G'} are Fourier coefficients of 1/eps
        
        # We evaluate at high-symmetry points in the irreducible BZ
        # For square lattice: Gamma (0,0), M (pi,0), X (pi/2, pi/2) in normalized units
        # Actually, the irreducible BZ for square lattice is 1/8 of the full BZ
        # Key points: Gamma (0,0), M (1,0) [in units of pi/a], X (0.5, 0.5) [in units of pi/a]
        # Let's use k-points in units of 2*pi/a:
        # Gamma: (0, 0)
        # M: (1, 0) -> (pi/a, 0)
        # X: (0.5, 0.5) -> (pi/(2a), pi/(2a))
        
        k_points = [
            (0.0, 0.0),      # Gamma
            (1.0, 0.0),      # M
            (0.5, 0.5),      # X
            (0.7, 0.0),      # interior
            (0.3, 0.3),      # interior
        ]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # k in units of 2*pi/a
            # Actual k-vector: k = (kx, ky) * 2*pi
            kx_actual = kx * 2 * np.pi
            ky_actual = ky * 2 * np.pi
            
            # Build the matrix
            n = n_pw_actual
            mat = np.zeros((n, n), dtype=complex)
            
            for i, (mi, ni) in enumerate(pw_indices):
                Gx_i = mi * 2 * np.pi
                Gy_i = ni * 2 * np.pi
                kx_total = kx_actual + Gx_i
                ky_total = ky_actual + Gy_i
                k2_i = kx_total**2 + ky_total**2
                
                for j, (mj, nj) in enumerate(pw_indices):
                    Gx_j = mj * 2 * np.pi
                    Gy_j = nj * 2 * np.pi
                    
                    # Fourier coefficient of 1/eps at (G_i - G_j)
                    dmi = mi - mj
                    dni = ni - nj
                    coeff = get_fft_coeff(dmi, dni)
                    
                    mat[i, j] = coeff * k2_i
            
            # Solve eigenvalue problem
            # mat is Hermitian if inv_eps is real
            # Use eigh for Hermitian matrices
            try:
                eigenvalues = np.linalg.eigvalsh(mat)
                eigenvalues = np.sort(np.real(eigenvalues))
            except:
                continue
            
            # omega^2 = eigenvalue, so omega = sqrt(eigenvalue)
            # Bands 4 and 5 are indices 3 and 4 (0-indexed)
            # But we need to be careful: the bands are sorted by omega
            # omega_4 = sqrt(eigenvalues[3]), omega_5 = sqrt(eigenvalues[4])
            
            if len(eigenvalues) >= 5:
                omega4 = np.sqrt(max(0, eigenvalues[3]))
                omega5 = np.sqrt(max(0, eigenvalues[4]))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        ratio = gap / midgap if midgap > 0 else 0.0
        return ratio
    
    # --- Initial design: square frame ---
    def make_square_frame(W, H):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                dx = abs(i - center)
                dy = abs(j - center)
                is_outer = (dx < W) and (dy < W)
                is_inner = (dx < H) and (dy < H)
                if is_outer and not is_inner:
                    cell[i, j] = 1
        return cell
    
    # Try a few initial designs and pick the best
    initial_candidates = []
    for W in [12.0, 13.0, 14.0]:
        for H in [4.0, 5.0, 6.0, 7.0]:
            cell = make_square_frame(W, H)
            initial_candidates.append(cell)
    
    # Also try a cross shape
    def make_cross(width=6.0):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                dx = abs(i - center)
                dy = abs(j - center)
                if dx < width or dy < width:
                    cell[i, j] = 1
        return cell
    
    initial_candidates.append(make_cross(5.0))
    initial_candidates.append(make_cross(6.0))
    
    # Evaluate initial candidates
    best_cell = None
    best_score = -float('inf')
    for cell in initial_candidates:
        score = compute_gap_ratio(cell, n_pw=5)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # --- Local search: pixel flips ---
    # Respect C4v symmetry: flipping a pixel requires flipping its symmetric counterparts
    # Symmetry group: (i,j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
    #                 (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)
    
    def get_symmetric_pixels(i, j):
        """Get all symmetric counterparts of pixel (i, j)"""
        pixels = set()
        pixels.add((i, j))
        pixels.add((size-1-i, j))
        pixels.add((i, size-1-j))
        pixels.add((size-1-i, size-1-j))
        pixels.add((j, i))
        pixels.add((size-1-j, i))
        pixels.add((j, size-1-i))
        pixels.add((size-1-j, size-1-i))
        return pixels
    
    # Precompute symmetric pixel groups
    # Each group is a set of pixels that must be flipped together
    pixel_groups = {}
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) not in visited:
                group = get_symmetric_pixels(i, j)
                for p in group:
                    visited.add(p)
                pixel_groups[(i, j)] = group
    
    # Unique groups
    unique_groups = []
    seen_groups = set()
    for i in range(size):
        for j in range(size):
            group = tuple(sorted(pixel_groups[(i, j)]))
            if group not in seen_groups:
                seen_groups.add(group)
                unique_groups.append(group)
    
    # Local search
    current_cell = best_cell
    current_score = best_score
    best_cell_local = best_cell.copy()
    best_score_local = best_score
    
    # Use a smaller n_pw for speed during local search
    n_pw_search = 5
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Pick a random pixel group to flip
        idx = np.random.randint(len(unique_groups))
        group = unique_groups[idx]
        
        # Flip the pixels
        new_cell = current_cell.copy()
        for (pi, pj) in group:
            new_cell[pi, pj] = 1 - current_cell[pi, pj]
        
        # Evaluate
        new_score = compute_gap_ratio(new_cell, n_pw=n_pw_search)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            if new_score > best_score_local:
                best_score_local = new_score
                best_cell_local = new_cell.copy()
        else:
            # Accept with some probability (simulated annealing-like)
            # Simple: accept with small probability to escape local minima
            if np.random.random() < 0.05:
                current_cell = new_cell
                current_score = new_score
    
    return best_cell_local.tolist()
