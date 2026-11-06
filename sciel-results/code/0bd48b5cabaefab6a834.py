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
    
    # Precompute basis functions for C4v-symmetric combinations
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # Fast PWE Surrogate Parameters - increased to M=5 for better accuracy
    M_PWE = 5  # Indices -5 to 5, so 11x11 = 121 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # More comprehensive k-points in the irreducible BZ (fractional coordinates)
    # Include high-symmetry points and interior points where band crossings occur
    k_points = [
        (0.0, 0.0),       # Gamma
        (0.5, 0.0),       # X
        (0.5, 0.5),       # M
        (0.25, 0.0),      # Midpoint Gamma-X
        (0.25, 0.25),     # Interior point
        (0.35, 0.15),     # Interior point
        (0.4, 0.2),       # Interior point
        (0.3, 0.3),       # Interior point
        (0.1, 0.1),       # Near Gamma
        (0.45, 0.05),     # Near X
        (0.45, 0.45),     # Near M
        (0.2, 0.1),       # Interior
        (0.35, 0.35),     # Interior
        (0.15, 0.25),     # Interior
        (0.4, 0.4),       # Interior
        (0.3, 0.1),       # Interior
        (0.25, 0.15),     # Interior
        (0.35, 0.25),     # Interior
        (0.2, 0.2),       # Interior
        (0.42, 0.08),     # Near X-M edge
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        # Precompute epsilon difference matrix for all G pairs
        # E[i,j] = eps(G_i - G_j)
        E_diff = np.zeros((N_PW, N_PW), dtype=complex)
        for i in range(N_PW):
            for j in range(N_PW):
                dGx = Gx_flat[i] - Gx_flat[j]
                dGy = Gy_flat[i] - Gy_flat[j]
                ix = int(dGx) % size
                iy = int(dGy) % size
                E_diff[i, j] = fft_vals[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = kGx**2 + kGy**2
            
            # A = D * E where D is diagonal
            A = D[:, None] * E_diff
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
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == np.inf or max_omega_4 == 0.0:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        score = gap / midgap
        return score

    def generate_cell(A20, A22, offset, phase20=0.0, phase22=0.0):
        """Generate a binary cell from Fourier parameters."""
        field = offset
        if A20 > 0:
            field += A20 * (np.cos(4 * np.pi * X + phase20) + np.cos(4 * np.pi * Y + phase20))
        if A22 > 0:
            field += A22 * (np.cos(4 * np.pi * (X + Y) + phase22) + np.cos(4 * np.pi * (X - Y) + phase22))
        
        cell_raw = (field > 0.5).astype(int)
        
        # Enforce C4v symmetry by averaging over all 8 symmetry operations
        sym_cells = [
            cell_raw,
            cell_raw[:, ::-1],
            cell_raw[::-1, :],
            cell_raw.T,
            cell_raw[::-1, ::-1],
            cell_raw.T[::-1, :],
            cell_raw[:, ::-1].T,
            cell_raw[::-1, :].T
        ]
        
        avg_cell = np.mean(sym_cells, axis=0)
        cell_sym = (avg_cell >= 0.5).astype(int)
        
        return cell_sym

    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Multi-stage optimization with improved coverage
    # Stage 1: Coarse grid search with wider range
    A20_coarse = np.linspace(0.05, 1.5, 12)
    A22_coarse = np.linspace(0.0, 1.2, 10)
    offset_coarse = np.linspace(0.25, 0.75, 11)
    
    for A20 in A20_coarse:
        for A22 in A22_coarse:
            for offset in offset_coarse:
                cell = generate_cell(A20, A22, offset)
                
                fill = np.sum(cell) / (size * size)
                if fill < 0.10 or fill > 0.90:
                    continue
                
                score = calculate_surrogate_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = (A20, A22, offset, 0.0, 0.0)
    
    # Stage 2: Fine-tune around best candidate with expanded phase search
    if best_cell is not None:
        A20_best, A22_best, offset_best, p20_best, p22_best = best_params
        
        A20_fine = np.linspace(max(0.02, A20_best - 0.2), min(1.8, A20_best + 0.2), 7)
        A22_fine = np.linspace(max(0.0, A22_best - 0.2), min(1.8, A22_best + 0.2), 7)
        offset_fine = np.linspace(max(0.20, offset_best - 0.1), min(0.80, offset_best + 0.1), 7)
        phase20_fine = np.linspace(-np.pi/4, np.pi/4, 5)
        phase22_fine = np.linspace(-np.pi/4, np.pi/4, 5)
        
        for A20 in A20_fine:
            for A22 in A22_fine:
                for offset in offset_fine:
                    for p20 in phase20_fine:
                        for p22 in phase22_fine:
                            cell = generate_cell(A20, A22, offset, p20, p22)
                            
                            fill = np.sum(cell) / (size * size)
                            if fill < 0.10 or fill > 0.90:
                                continue
                            
                            score = calculate_surrogate_gap(cell)
                            
                            if score > best_score:
                                best_score = score
                                best_cell = cell.tolist()
                                best_params = (A20, A22, offset, p20, p22)
    
    # Stage 3: Targeted configurations based on known good photonic crystal designs
    if best_cell is not None:
        test_configs = [
            (0.8, 0.5, 0.5, 0.0, 0.0),
            (0.6, 0.7, 0.45, 0.0, 0.0),
            (0.9, 0.3, 0.55, 0.0, 0.0),
            (0.7, 0.6, 0.5, 0.2, 0.0),
            (0.7, 0.6, 0.5, -0.2, 0.0),
            (0.5, 0.8, 0.4, 0.0, 0.3),
            (0.5, 0.8, 0.4, 0.0, -0.3),
            (1.0, 0.4, 0.6, 0.0, 0.0),
            (0.4, 1.0, 0.4, 0.0, 0.0),
            # Additional targeted configs for TM band gap
            (0.75, 0.45, 0.52, 0.1, 0.1),
            (0.65, 0.55, 0.48, -0.1, -0.1),
            (0.85, 0.35, 0.58, 0.15, 0.0),
            (0.55, 0.65, 0.42, 0.0, 0.2),
            (0.95, 0.25, 0.62, 0.0, 0.0),
            (0.45, 0.75, 0.38, 0.0, 0.0),
            (0.7, 0.5, 0.5, np.pi/6, 0.0),
            (0.7, 0.5, 0.5, 0.0, np.pi/6),
            (0.6, 0.6, 0.5, np.pi/8, np.pi/8),
            (0.8, 0.4, 0.5, -np.pi/8, -np.pi/8),
        ]
        
        for A20, A22, offset, p20, p22 in test_configs:
            cell = generate_cell(A20, A22, offset, p20, p22)
            
            fill = np.sum(cell) / (size * size)
            if fill < 0.10 or fill > 0.90:
                continue
            
            score = calculate_surrogate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.tolist()
                best_params = (A20, A22, offset, p20, p22)
    
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
