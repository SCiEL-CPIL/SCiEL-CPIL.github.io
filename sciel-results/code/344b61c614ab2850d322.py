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
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    # Focus on points where bands 4 and 5 are close
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.15, 0.15),
        (0.45, 0.05),
        (0.2, 0.4),
        (0.4, 0.4),
        (0.1, 0.3),
        (0.35, 0.05),
        (0.5, 0.1),
        (0.1, 0.1),
        (0.45, 0.45),
        (0.25, 0.1),
        (0.1, 0.25),
        (0.3, 0.1),
        (0.2, 0.3),
        (0.4, 0.1),
        (0.1, 0.4),
        (0.3, 0.2),
        (0.2, 0.2),
        (0.45, 0.25),
        (0.25, 0.45),
        (0.35, 0.35),
        (0.15, 0.05),
        (0.05, 0.15),
    ]
    
    # Precompute G-difference matrices for speed
    dGx = Gx_flat[:, None] - Gx_flat[None, :]
    dGy = Gy_flat[:, None] - Gy_flat[None, :]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        def get_eps(dx, dy):
            ix = int(dx) % size
            iy = int(dy) % size
            return fft_vals[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = kGx**2 + kGy**2
            
            # Vectorized computation of epsilon matrix
            E = np.zeros((N_PW, N_PW), dtype=complex)
            for i in range(N_PW):
                for j in range(N_PW):
                    E[i, j] = get_eps(Gx_flat[i] - Gx_flat[j], Gy_flat[i] - Gy_flat[j])
            
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

    def generate_cell(A20, A22, offset, phase20=0.0, phase22=0.0):
        """Generate a binary cell from Fourier parameters."""
        # Apply phases by rotating the basis functions
        # For C4v symmetry, we use cos with phase shifts
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

    def nelder_mead(params, bounds, max_iter=100, step_size=0.1):
        """Simple Nelder-Mead style local optimizer."""
        n = len(params)
        # Initialize simplex
        simplex = [np.array(params, dtype=float)]
        for i in range(n):
            trial = np.array(params, dtype=float)
            trial[i] += step_size
            simplex.append(trial)
        
        best_params = np.array(params, dtype=float)
        best_score = calculate_surrogate_gap(generate_cell(*params))
        
        for iteration in range(max_iter):
            # Evaluate all points
            scores = []
            for p in simplex:
                # Clip to bounds
                p_clipped = np.clip(p, bounds[0], bounds[1])
                s = calculate_surrogate_gap(generate_cell(*p_clipped))
                scores.append(s if np.isfinite(s) else -np.inf)
            
            # Sort simplex by score (worst first)
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            # Check convergence
            if abs(scores[-1] - scores[0]) < 1e-6:
                break
            
            # Reflection
            centroid = np.mean(simplex[:-1], axis=0)
            reflected = centroid + (centroid - simplex[-1])
            reflected = np.clip(reflected, bounds[0], bounds[1])
            ref_score = calculate_surrogate_gap(generate_cell(*reflected))
            
            if ref_score > scores[-2]:
                if ref_score > scores[0]:
                    # Expansion
                    expanded = centroid + 2 * (reflected - centroid)
                    expanded = np.clip(expanded, bounds[0], bounds[1])
                    exp_score = calculate_surrogate_gap(generate_cell(*expanded))
                    if exp_score > ref_score:
                        simplex[-1] = expanded
                    else:
                        simplex[-1] = reflected
                else:
                    simplex[-1] = reflected
            else:
                # Contraction
                contracted = centroid + 0.5 * (simplex[-1] - centroid)
                contracted = np.clip(contracted, bounds[0], bounds[1])
                con_score = calculate_surrogate_gap(generate_cell(*contracted))
                if con_score > scores[-1]:
                    simplex[-1] = contracted
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[-1] = simplex[0].copy()
            
            # Track best
            current_best = np.clip(simplex[0], bounds[0], bounds[1])
            current_score = calculate_surrogate_gap(generate_cell(*current_best))
            if current_score > best_score:
                best_score = current_score
                best_params = current_best.copy()
        
        return best_params, best_score

    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Multi-stage optimization:
    # Stage 1: Coarse grid search over amplitudes and offset
    # Stage 2: Fine-tune around best candidate with phase optimization
    # Stage 3: Local Nelder-Mead refinement
    
    # Stage 1: Coarse search
    A20_coarse = np.linspace(0.1, 1.4, 10)
    A22_coarse = np.linspace(0.0, 1.2, 8)
    offset_coarse = np.linspace(0.25, 0.75, 11)
    
    for A20 in A20_coarse:
        for A22 in A22_coarse:
            for offset in offset_coarse:
                cell = generate_cell(A20, A22, offset)
                
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                
                score = calculate_surrogate_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = (A20, A22, offset, 0.0, 0.0)
    
    # Stage 2: Fine-tune around best candidate
    if best_cell is not None:
        A20_best, A22_best, offset_best, p20_best, p22_best = best_params
        
        # Fine grid around best values
        A20_fine = np.linspace(max(0.05, A20_best - 0.2), min(1.6, A20_best + 0.2), 7)
        A22_fine = np.linspace(max(0.0, A22_best - 0.2), min(1.4, A22_best + 0.2), 7)
        offset_fine = np.linspace(max(0.2, offset_best - 0.1), min(0.8, offset_best + 0.1), 7)
        phase20_fine = np.linspace(-0.6, 0.6, 5)
        phase22_fine = np.linspace(-0.6, 0.6, 5)
        
        for A20 in A20_fine:
            for A22 in A22_fine:
                for offset in offset_fine:
                    for p20 in phase20_fine:
                        for p22 in phase22_fine:
                            cell = generate_cell(A20, A22, offset, p20, p22)
                            
                            fill = np.sum(cell) / (size * size)
                            if fill < 0.15 or fill > 0.85:
                                continue
                            
                            score = calculate_surrogate_gap(cell)
                            
                            if score > best_score:
                                best_score = score
                                best_cell = cell.tolist()
                                best_params = (A20, A22, offset, p20, p22)
    
    # Stage 3: Local refinement using Nelder-Mead
    if best_cell is not None:
        # Define bounds for parameters
        bounds = (np.array([0.05, 0.0, 0.2, -1.0, -1.0]),
                  np.array([1.6, 1.4, 0.8, 1.0, 1.0]))
        
        refined_params, refined_score = nelder_mead(best_params, bounds, max_iter=80, step_size=0.08)
        
        if refined_score > best_score:
            best_score = refined_score
            best_cell = generate_cell(*refined_params).tolist()
            best_params = tuple(refined_params)
        
        # Also try a few more targeted configurations known to be good for TM gaps
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
            (0.9, 0.6, 0.48, 0.1, 0.1),
            (0.85, 0.55, 0.52, -0.1, 0.15),
            (1.1, 0.35, 0.55, 0.0, 0.0),
            (0.75, 0.65, 0.45, 0.15, -0.1),
        ]
        
        for A20, A22, offset, p20, p22 in test_configs:
            cell = generate_cell(A20, A22, offset, p20, p22)
            
            fill = np.sum(cell) / (size * size)
            if fill < 0.15 or fill > 0.85:
                continue
            
            score = calculate_surrogate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.tolist()
                best_params = (A20, A22, offset, p20, p22)
        
        # Another round of Nelder-Mead from the new best
        if best_params is not None:
            refined_params2, refined_score2 = nelder_mead(best_params, bounds, max_iter=60, step_size=0.05)
            
            if refined_score2 > best_score:
                best_score = refined_score2
                best_cell = generate_cell(*refined_params2).tolist()
    
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
