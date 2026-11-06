import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics (1,0), (2,0), and (2,2) modes that directly 
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
    # (1,0) + (0,1): cos(2πX) + cos(2πY)
    basis_10 = np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)
    
    # (2,0) + (0,2): cos(4πX) + cos(4πY)
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    
    # (2,2) + (2,-2): cos(4π(X+Y)) + cos(4π(X-Y))
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # Fast PWE Surrogate Parameters
    M_PWE = 5  # Indices -5 to 5, so 11x11 = 121 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Precompute difference matrices for epsilon coupling
    dx_matrix = (Gx_flat[:, None] - Gx_flat[None, :]).astype(int)
    dy_matrix = (Gy_flat[:, None] - Gy_flat[None, :]).astype(int)
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    # More points for better gap estimation
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.15, 0.0),
        (0.45, 0.05),
        (0.2, 0.1),
        (0.35, 0.35),
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = kGx**2 + kGy**2
            
            # Vectorized computation of epsilon matrix using precomputed differences
            E = fft_vals[np.mod(dx_matrix, size), np.mod(dy_matrix, size)]
            
            A = D[:, None] * E
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

    def generate_cell(A10, A20, A22, offset, phase10=0.0, phase20=0.0, phase22=0.0):
        """Generate a binary cell from Fourier parameters."""
        field = offset
        if A10 > 0:
            field += A10 * (np.cos(2 * np.pi * X + phase10) + np.cos(2 * np.pi * Y + phase10))
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

    def refine_cell(cell, n_iterations=50):
        """
        Local refinement: flip individual pixels in the symmetric fundamental domain
        to improve the gap score.
        """
        best = cell.copy()
        best_score = calculate_surrogate_gap(best)
        
        # Identify independent pixels under C4v symmetry
        # For a 32x32 grid centered at (15.5, 15.5), the fundamental domain
        # consists of pixels where i <= j and i + j <= 31 (upper-left triangle)
        # plus the diagonal and center lines
        
        for iteration in range(n_iterations):
            improved = False
            # Try flipping each pixel in the fundamental domain
            for i in range(size // 2):
                for j in range(size // 2):
                    if i > j:  # Skip by symmetry
                        continue
                    
                    # Get all symmetric equivalents of pixel (i,j)
                    positions = set()
                    positions.add((i, j))
                    positions.add((size-1-i, j))
                    positions.add((i, size-1-j))
                    positions.add((size-1-i, size-1-j))
                    positions.add((j, i))
                    positions.add((size-1-j, i))
                    positions.add((j, size-1-i))
                    positions.add((size-1-j, size-1-i))
                    
                    # Try flipping
                    trial = best.copy()
                    for (pi, pj) in positions:
                        trial[pi, pj] = 1 - trial[pi, pj]
                    
                    fill = np.sum(trial) / (size * size)
                    if fill < 0.10 or fill > 0.90:
                        continue
                    
                    score = calculate_surrogate_gap(trial)
                    if score > best_score + 1e-6:
                        best = trial
                        best_score = score
                        improved = True
            
            if not improved:
                break
        
        return best, best_score

    start_time = time.time()
    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Multi-stage optimization:
    # Stage 1: Coarse grid search over amplitudes and offset
    # Stage 2: Fine-tune around best candidate with phase optimization
    # Stage 3: Local pixel-level refinement
    
    # Stage 1: Coarse search
    A10_coarse = np.linspace(0.0, 0.8, 4)
    A20_coarse = np.linspace(0.1, 1.2, 7)
    A22_coarse = np.linspace(0.0, 1.0, 6)
    offset_coarse = np.linspace(0.30, 0.70, 8)
    
    for A10 in A10_coarse:
        for A20 in A20_coarse:
            for A22 in A22_coarse:
                for offset in offset_coarse:
                    if time.time() - start_time > 80:
                        break
                    cell = generate_cell(A10, A20, A22, offset)
                    
                    fill = np.sum(cell) / (size * size)
                    if fill < 0.10 or fill > 0.90:
                        continue
                    
                    score = calculate_surrogate_gap(cell)
                    
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
                        best_params = (A10, A20, A22, offset, 0.0, 0.0, 0.0)
    
    # Stage 2: Fine-tune around best candidate
    if best_cell is not None:
        A10_best, A20_best, A22_best, offset_best, p10_best, p20_best, p22_best = best_params
        
        # Fine grid around best values
        A10_fine = np.linspace(max(0.0, A10_best - 0.15), min(1.0, A10_best + 0.15), 4)
        A20_fine = np.linspace(max(0.05, A20_best - 0.15), min(1.5, A20_best + 0.15), 4)
        A22_fine = np.linspace(max(0.0, A22_best - 0.15), min(1.5, A22_best + 0.15), 4)
        offset_fine = np.linspace(max(0.25, offset_best - 0.08), min(0.75, offset_best + 0.08), 4)
        phase20_fine = np.linspace(-0.5, 0.5, 3)
        phase22_fine = np.linspace(-0.5, 0.5, 3)
        
        for A10 in A10_fine:
            for A20 in A20_fine:
                for A22 in A22_fine:
                    for offset in offset_fine:
                        for p20 in phase20_fine:
                            for p22 in phase22_fine:
                                if time.time() - start_time > 90:
                                    break
                                cell = generate_cell(A10, A20, A22, offset, 0.0, p20, p22)
                                
                                fill = np.sum(cell) / (size * size)
                                if fill < 0.10 or fill > 0.90:
                                    continue
                                
                                score = calculate_surrogate_gap(cell)
                                
                                if score > best_score:
                                    best_score = score
                                    best_cell = cell.copy()
    
    # Stage 3: Local pixel-level refinement
    if best_cell is not None and time.time() - start_time < 100:
        best_cell, best_score = refine_cell(best_cell, n_iterations=30)
    
    if best_cell is None:
        # Fallback: Standard rod
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            xi = (i + 0.5) / size - 0.5
            for j in range(size):
                yj = (j + 0.5) / size - 0.5
                if xi*xi + yj*yj < 0.2**2:
                    best_cell[i, j] = 1
        
    return best_cell.tolist()
