import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics (2,0), (0,2), and (2,2) modes that directly 
    couple to the 4th and 5th TM bands. Optimizes amplitudes and phases 
    to maximize the spectral separation between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # C4v symmetric basis functions
    # Basis 1: cos(4piX) + cos(4piY)
    # Basis 2: cos(4pi(X+Y))
    # Basis 3: cos(4pi(X-Y))
    B1 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    B2 = np.cos(4 * np.pi * (X + Y))
    B3 = np.cos(4 * np.pi * (X - Y))
    
    # Fast PWE Surrogate Parameters
    # Use a slightly larger basis set if time allows, but 9x9 is a good start for speed
    M_PWE = 4 
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Precompute difference matrices for epsilon
    dx_matrix = Gx_flat[:, None] - Gx_flat[None, :]
    dy_matrix = Gy_flat[:, None] - Gy_flat[None, :]
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    # We need to sample the BZ well to find the min omega_5 and max omega_4
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= 1
    # Critical points: Gamma (0,0), X (0.5,0), M (0.5,0.5)
    # And points in between.
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.5, 0.25),
        (0.1, 0.1),
        (0.4, 0.1),
        (0.3, 0.2),
        (0.45, 0.35)
    ]
    
    # Precompute kG terms for each k-point to speed up loop
    kG_data = []
    for kx, ky in k_points:
        kGx = kx + Gx_flat
        kGy = ky + Gy_flat
        D = kGx**2 + kGy**2
        kG_data.append(D)
        
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        # The evaluator uses exact coefficients. We approximate with FFT.
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        # Map G indices to FFT indices
        # G indices are -M..M. FFT indices are 0..N-1.
        # For a given G, index in FFT is G % size.
        Gx_idx = Gx_flat % size
        Gy_idx = Gy_flat % size
        
        # We need eps(G_i - G_j).
        # dx_matrix contains Gx_i - Gx_j.
        # We need to look up fft_vals[dx, dy].
        # Since fft_vals is 2D, we can use advanced indexing.
        # dx_matrix and dy_matrix are (N_PW, N_PW).
        
        # Handle modulo correctly for negative indices
        # np.mod handles this, but we need to be careful with shape.
        eps_matrix = fft_vals[np.mod(dx_matrix, size), np.mod(dy_matrix, size)]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for D in kG_data:
            # A = D * E
            # D is diagonal (N_PW,). E is (N_PW, N_PW).
            A = D[:, None] * eps_matrix
            # Symmetrize
            A = (A + A.conj().T) / 2.0
            A = np.real(A)
            
            try:
                eigvals = np.linalg.eigvalsh(A)
            except np.linalg.LinAlgError:
                return -np.inf
                
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                
                if w4 > 0 and w5 > 0:
                    if w4 > max_omega_4:
                        max_omega_4 = w4
                    if w5 < min_omega_5:
                        min_omega_5 = w5
        
        if min_omega_5 == np.inf or max_omega_4 == 0.0:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        score = gap / midgap
        return score

    def generate_cell(C0, A1, A2, A3):
        """Generate a binary cell from Fourier parameters."""
        field = C0 + A1 * B1 + A2 * B2 + A3 * B3
        
        # Threshold to binary
        # We want to maintain the symmetry. Since B1, B2, B3 are C4v symmetric,
        # any linear combination is C4v symmetric.
        # Thresholding might break symmetry if the field crosses 0.5 asymmetrically?
        # No, if F(X,Y) is symmetric, F(X,Y) > 0.5 is symmetric.
        cell = (field > 0.5).astype(int)
        
        # Enforce strict symmetry just in case of floating point errors
        # Average over symmetries
        sym_cells = [
            cell,
            cell[:, ::-1],
            cell[::-1, :],
            cell.T,
            cell[::-1, ::-1],
            cell.T[::-1, :],
            cell[:, ::-1].T,
            cell[::-1, :].T
        ]
        avg_cell = np.mean(sym_cells, axis=0)
        cell_sym = (avg_cell >= 0.5).astype(int)
        
        return cell_sym

    def evaluate_params(C0, A1, A2, A3):
        cell = generate_cell(C0, A1, A2, A3)
        fill = np.sum(cell) / (size * size)
        if fill < 0.10 or fill > 0.90:
            return -np.inf
        return calculate_surrogate_gap(cell)

    best_score = -np.inf
    best_params = (0.5, 0.5, 0.5, 0.0)
    
    # Stage 1: Coarse Grid Search
    # C0: 0.4 to 0.6 (fill fraction control)
    # A1: -1.0 to 1.0
    # A2: -1.0 to 1.0
    # A3: -1.0 to 1.0
    
    # To save time, we'll do a coarse grid first
    C0_grid = np.linspace(0.35, 0.65, 7)
    A1_grid = np.linspace(-1.0, 1.0, 7)
    A2_grid = np.linspace(-1.0, 1.0, 7)
    A3_grid = np.linspace(-1.0, 1.0, 7)
    
    # This is 7^4 = 2401 evaluations.
    # Each evaluation involves 10 k-points * eigvalsh(81x81).
    # eigvalsh(81x81) is fast (~ms). 10 * 2401 * 1ms = 24s.
    # This is acceptable.
    
    for C0 in C0_grid:
        for A1 in A1_grid:
            for A2 in A2_grid:
                for A3 in A3_grid:
                    score = evaluate_params(C0, A1, A2, A3)
                    if score > best_score:
                        best_score = score
                        best_params = (C0, A1, A2, A3)
    
    # Stage 2: Fine-tune around best candidate
    C0_best, A1_best, A2_best, A3_best = best_params
    
    # Fine grid
    step = 0.2
    C0_fine = np.linspace(C0_best - step, C0_best + step, 5)
    A1_fine = np.linspace(A1_best - step, A1_best + step, 5)
    A2_fine = np.linspace(A2_best - step, A2_best + step, 5)
    A3_fine = np.linspace(A3_best - step, A3_best + step, 5)
    
    for C0 in C0_fine:
        for A1 in A1_fine:
            for A2 in A2_fine:
                for A3 in A3_fine:
                    score = evaluate_params(C0, A1, A2, A3)
                    if score > best_score:
                        best_score = score
                        best_params = (C0, A1, A2, A3)
                        
    # Stage 3: Even finer tuning
    step = 0.05
    C0_best, A1_best, A2_best, A3_best = best_params
    C0_fine2 = np.linspace(C0_best - step, C0_best + step, 5)
    A1_fine2 = np.linspace(A1_best - step, A1_best + step, 5)
    A2_fine2 = np.linspace(A2_best - step, A2_best + step, 5)
    A3_fine2 = np.linspace(A3_best - step, A3_best + step, 5)
    
    for C0 in C0_fine2:
        for A1 in A1_fine2:
            for A2 in A2_fine2:
                for A3 in A3_fine2:
                    score = evaluate_params(C0, A1, A2, A3)
                    if score > best_score:
                        best_score = score
                        best_params = (C0, A1, A2, A3)

    C0_best, A1_best, A2_best, A3_best = best_params
    best_cell = generate_cell(C0_best, A1_best, A2_best, A3_best)
    
    # Final check: if score is very bad, try a known good configuration
    # E.g. standard hole
    if best_score < 0.1:
         # Try a few hardcoded "good" structures
         test_configs = [
             (0.5, 0.8, 0.2, 0.0),
             (0.5, 0.5, 0.8, 0.0),
             (0.5, 0.0, 0.8, 0.8),
             (0.5, 0.8, 0.0, 0.8),
         ]
         for cfg in test_configs:
             score = evaluate_params(*cfg)
             if score > best_score:
                 best_score = score
                 best_params = cfg
                 
         if best_params != (C0_best, A1_best, A2_best, A3_best):
             C0_best, A1_best, A2_best, A3_best = best_params
             best_cell = generate_cell(C0_best, A1_best, A2_best, A3_best)

    if best_cell is None:
        # Fallback: Standard rod
        best_cell = []
        for i in range(size):
            xi = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                yj = (j + 0.5) / size - 0.5
                if xi*xi + yj*yj < 0.2**2:
                    row.append(1)
                else:
                    row.append(0)
            best_cell.append(row)
            
    return best_cell
