import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell by optimizing a parametric 
    dielectric distribution defined by a superposition of low-order Fourier 
    harmonics (2,0), (0,2), (2,2), (4,0), (0,4), (4,4), and (4,-4) modes.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Precompute the basis functions for the harmonics
    # Mode (2,0) and (0,2) combined for C4v symmetry
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    
    # Mode (2,2) and (2,-2) combined for C4v symmetry
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # Mode (4,0) and (0,4) combined for C4v symmetry
    basis_40 = np.cos(8 * np.pi * X) + np.cos(8 * np.pi * Y)
    
    # Mode (4,4) and (4,-4) combined for C4v symmetry
    basis_44 = np.cos(8 * np.pi * (X + Y)) + np.cos(8 * np.pi * (X - Y))
    
    # Search parameters - expanded grid
    A20_vals = np.linspace(0.0, 1.0, 6)
    A22_vals = np.linspace(0.0, 1.0, 6)
    A40_vals = np.linspace(0.0, 0.5, 4)
    A44_vals = np.linspace(0.0, 0.5, 4)
    offset_vals = np.linspace(0.35, 0.65, 7)
    
    best_score = -np.inf
    best_cell = None
    
    # Fast PWE Surrogate Parameters
    # Use a larger number of plane waves for better accuracy
    M_PWE = 4  # Indices -4 to 4, so 9x9 = 81 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.4, 0.1)
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # 1. Compute Fourier coefficients of the pixel map
        # epsilon(G) = 1/N^2 sum epsilon(r) exp(-i G.r)
        # FFT gives sum. We divide by N^2.
        # The FFT output index [n, m] corresponds to frequency (n, m) with wrapping.
        # We need epsilon(Gx, Gy) for Gx, Gy in [-M, M].
        
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        # Helper to get epsilon(Gx, Gy) from FFT array
        # FFT indices are 0..N-1. 
        # Frequency k corresponds to index k if k >= 0, else N + k.
        def get_eps(Gx_val, Gy_val):
            ix = int(Gx_val) % size
            iy = int(Gy_val) % size
            return fft_vals[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build PWE matrix
            # A_{ij} = (k+Gi) . (k+Gj) * eps(Gi - Gj)
            # Gi, Gj are the plane wave vectors
            # k is the wave vector
            
            # Precompute (k+G) vectors
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            # Dot products (k+Gi).(k+Gj) can be computed via outer product
            # D_{ij} = (kx+Gx_i)(kx+Gx_j) + (ky+Gy_i)(ky+Gy_j)
            # This is (kx+Gx_i) @ (kx+Gx_j) + ...
            # Actually, D = outer(kGx, kGx) + outer(kGy, kGy)
            D = np.outer(kGx, kGx) + np.outer(kGy, kGy)
            
            # Compute epsilon matrix E_{ij} = eps(Gi - Gj)
            # Gi - Gj = (Gx_i - Gx_j, Gy_i - Gy_j)
            # We need to loop or vectorize. Loop is fine for 49x49.
            E = np.zeros((N_PW, N_PW), dtype=complex)
            for i in range(N_PW):
                for j in range(N_PW):
                    dx = Gx_flat[i] - Gx_flat[j]
                    dy = Gy_flat[i] - Gy_flat[j]
                    E[i, j] = get_eps(dx, dy)
            
            # Matrix A = D * E (element-wise multiplication)
            # Wait, the formula is sum_j (k+Gi).(k+Gj) eps(Gi-Gj) Hj = w^2 Hi
            # So A_{ij} = (k+Gi).(k+Gj) eps(Gi-Gj)
            A = D * E
            
            # A should be Hermitian. Due to numerical errors, take real part or symmetrize.
            # eps(G) is complex, but for symmetric real epsilon, eps(-G) = eps*(G).
            # The matrix should be Hermitian.
            A = (A + A.conj().T) / 2.0
            A = np.real(A) # Should be real if symmetric
            
            # Solve eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(A)
            except np.linalg.LinAlgError:
                return -np.inf
                
            # Sort eigenvalues
            eigvals = np.sort(eigvals)
            
            # We need bands 4 and 5 (1-indexed), so indices 3 and 4
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

    # Grid Search
    for A20 in A20_vals:
        for A22 in A22_vals:
            for A40 in A40_vals:
                for A44 in A44_vals:
                    for offset in offset_vals:
                        # Generate continuous field
                        field = offset + A20 * basis_20 + A22 * basis_22 + A40 * basis_40 + A44 * basis_44
                        
                        # Threshold to binary
                        cell_raw = (field > 0.5).astype(int)
                        
                        # Enforce C4v symmetry
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
                        
                        # Average and re-threshold
                        avg_cell = np.mean(sym_cells, axis=0)
                        cell_sym = (avg_cell >= 0.5).astype(int)
                        
                        # Quick check: if cell is empty or full, skip
                        fill = np.sum(cell_sym) / (size * size)
                        if fill < 0.15 or fill > 0.85:
                            continue
                        
                        score = calculate_surrogate_gap(cell_sym)
                        
                        if score > best_score:
                            best_score = score
                            best_cell = cell_sym.tolist()
    
    if best_cell is None:
        # Fallback: Standard rod
        best_cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    row.append(1)
                else:
                    row.append(0)
            best_cell.append(row)
            
    return best_cell
