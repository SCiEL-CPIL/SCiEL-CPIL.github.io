"""Photonic crystal design with continuous parameterization and local PW optimization."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

def _build_pw_solver(size, eps_low, eps_high, n_pw, k_points):
    """
    Build a fast plane-wave band solver for a 2D square lattice.
    Returns a function that given a permittivity field (size x size array)
    computes the gap-midgap ratio for bands 4 and 5.
    """
    half = n_pw // 2
    # Fourier indices: -half..half
    idx = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(idx, idx, indexing='ij')
    G2 = Gx**2 + Gy**2
    nG = len(Gx)
    
    # Precompute G vectors as 1D arrays
    Gx_flat = Gx.ravel()
    Gy_flat = Gy.ravel()
    G2_flat = G2.ravel()
    
    # k-points in irreducible BZ (fractional coordinates)
    # For square lattice, irreducible BZ is 1/8 wedge: 0 <= ky <= kx, 0 <= kx
    # 30 points covering boundary and interior
    if k_points is None:
        # Generate 30 points in irreducible BZ
        k_points = []
        # Along X: (k, 0)
        for i in range(7):
            k = i / 6.0 * 0.5
            k_points.append((k, 0.0))
        # Along M-X: (k, k) from (0,0) to (0.5, 0.5)
        for i in range(1, 7):
            k = i / 6.0 * 0.5
            k_points.append((k, k))
        # Along M-G: (0.5, k) from (0.5, 0.5) to (0.5, 0)
        for i in range(1, 7):
            k = 0.5 - i / 6.0 * 0.5
            k_points.append((0.5, k))
        # Interior points
        for i in range(1, 7):
            for j in range(1, 7):
                kx = i / 6.0 * 0.5
                ky = j / 6.0 * 0.5
                if ky <= kx and (kx, ky) not in k_points:
                    k_points.append((kx, ky))
        k_points = k_points[:30]
    
    def compute_bands(eps_field):
        """
        Compute TM band structure. eps_field is size x size array of 0/1.
        Returns (max_omega_4, min_omega_5) over all k-points.
        """
        # Compute Fourier coefficients of 1/eps for TM mode
        # eps(x,y) = eps_low + (eps_high - eps_low) * eps_field
        # For TM: need 1/eps in the eigenvalue problem
        eps_full = eps_low + (eps_high - eps_low) * eps_field
        inv_eps = 1.0 / eps_full
        
        # FFT to get Fourier coefficients
        # Pad to at least 2*size for accurate coefficients
        N_fft = 2 * size
        inv_eps_padded = np.zeros((N_fft, N_fft), dtype=np.complex128)
        inv_eps_padded[:size, :size] = inv_eps
        inv_eps_fft = np.fft.fft2(inv_eps_padded) / (N_fft * N_fft)
        
        # Map Fourier indices: our G indices are -half..half
        # FFT gives indices 0..N_fft-1, corresponding to -N_fft/2..N_fft/2-1
        # We need to extract the coefficients for our G indices
        eps_fourier = np.zeros(nG, dtype=np.complex128)
        for gi in range(nG):
            gx = Gx_flat[gi]
            gy = Gy_flat[gi]
            # FFT index: gx + N_fft/2, gy + N_fft/2 (mod N_fft)
            fft_ix = (gx + N_fft // 2) % N_fft
            fft_iy = (gy + N_fft // 2) % N_fft
            eps_fourier[gi] = inv_eps_fft[fft_ix, fft_iy]
        
        # For TM mode, the eigenvalue problem is:
        # sum_G' eps_{G-G'} (|G+k|^2) c_{G'} = omega^2 c_G
        # where eps_{G-G'} are Fourier coefficients of 1/eps
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for kx_frac, ky_frac in k_points:
            # k vector in units of 2*pi (lattice constant = 1)
            kx = 2.0 * math.pi * kx_frac
            ky = 2.0 * math.pi * ky_frac
            
            # Build the matrix
            # A[G, G'] = eps_{G-G'} * |G + k|^2
            # This is a circulant-like structure
            # |G + k|^2 = (Gx + kx)^2 + (Gy + ky)^2
            kvec2 = (Gx_flat + kx)**2 + (Gy_flat + ky)**2
            
            # The matrix is: M[i,j] = eps_fourier[i-j] * kvec2[j]  (cyclic)
            # Actually: M[i,j] = sum over G'' such that G'' = G_i - G_j of eps_{G''} * kvec2[j]
            # eps_{G_i - G_j} is the Fourier coefficient at index (G_i - G_j)
            
            # For efficiency, use the fact that M = D * C where D is diagonal (kvec2) 
            # and C is circulant with first column = eps_fourier
            # Actually M[i,j] = eps_{G_i - G_j} * kvec2[j]
            # So M = C * diag(kvec2) where C[i,j] = eps_{G_i - G_j}
            
            # Build circulant matrix C
            # C[i,j] = eps_fourier[index_of(G_i - G_j)]
            # G_i - G_j has components (Gx_i - Gx_j, Gy_i - Gy_j)
            # We need to find which index in our G array corresponds to this difference
            
            # Precompute a lookup: for each pair (dx, dy) where dx,dy in range, 
            # find the index
            # Gx values are -half..half, so Gx_i - Gx_j ranges from -2*half to 2*half
            # But we only have coefficients for -half..half
            # For differences outside this range, the coefficient is 0 (assuming 
            # the FFT captured enough)
            
            # Actually, let's build the matrix directly
            # M[i,j] = eps_fourier[idx(G_i - G_j)] * kvec2[j]
            
            # Create index mapping
            # Map (gx, gy) to index in Gx_flat/Gy_flat
            g_map = {}
            for gi in range(nG):
                g_map[(int(Gx_flat[gi]), int(Gy_flat[gi]))] = gi
            
            M = np.zeros((nG, nG), dtype=np.complex128)
            for i in range(nG):
                gx_i = int(Gx_flat[i])
                gy_i = int(Gy_flat[i])
                for j in range(nG):
                    gx_j = int(Gx_flat[j])
                    gy_j = int(Gy_flat[j])
                    dx = gx_i - gx_j
                    dy = gy_i - gy_j
                    key = (dx, dy)
                    if key in g_map:
                        coeff = eps_fourier[g_map[key]]
                    else:
                        coeff = 0.0
                    M[i, j] = coeff * kvec2[j]
            
            # Solve eigenvalue problem
            # M is not Hermitian in general, but for real eps it should be
            # Actually for TM with real eps, M should be symmetric (not necessarily Hermitian)
            # Let's use eigh if possible
            try:
                M_sym = (M + M.T.conj()) / 2
                eigvals = np.linalg.eigvalsh(M_sym)
                eigvals = np.real(eigvals)
                eigvals = np.sort(eigvals)
            except:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.real(eigvals)
                eigvals = np.sort(eigvals)
            
            # Band indices: 1-based, so band 4 = index 3, band 5 = index 4
            if len(eigvals) >= 5:
                omega_4 = math.sqrt(eigvals[3])
                omega_5 = math.sqrt(eigvals[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        gap = min_omega_5 - max_omega_4
        midgap = min_omega_5 + max_omega_4
        if midgap <= 0:
            return 0.0
        return 2.0 * gap / midgap
    
    return compute_bands


def _apply_symmetry(cell, size):
    """
    Apply square symmetry: cell must be invariant under:
    - row reversal
    - column reversal  
    - transpose
    We enforce by averaging over all 8 symmetry operations.
    """
    # Generate all 8 symmetry operations of the square
    def get_sym_ops(size):
        ops = []
        for flip_row in [False, True]:
            for flip_col in [False, True]:
                for transpose in [False, True]:
                    ops.append((flip_row, flip_col, transpose))
        return ops
    
    sym_cell = np.zeros((size, size), dtype=np.float64)
    count = 0
    for fr, fc, tr in _get_sym_ops(size):
        c = cell.copy()
        if tr:
            c = c.T
        if fr:
            c = c[::-1, :]
        if fc:
            c = c[:, ::-1]
        sym_cell += c
        count += 1
    
    sym_cell /= count
    # Threshold
    return (sym_cell > 0.5).astype(int)


def _get_sym_ops(size):
    """Return all 8 symmetry operations as (flip_row, flip_col, transpose) tuples."""
    ops = []
    for tr in [False, True]:
        for fr in [False, True]:
            for fc in [False, True]:
                # Check if this is a valid symmetry of the square
                # All 8 combinations are valid for the dihedral group D4
                ops.append((fr, fc, tr))
    return ops[:8]


def _generate_symmetric_cell(size, params):
    """
    Generate a symmetric cell from continuous parameters.
    params: array of Fourier coefficients for a smooth field.
    The field is defined on a continuous domain and sampled at pixel centers.
    """
    # Use a low-order Fourier series to define the dielectric field
    # f(x,y) = sum_{|mx|,|my| <= N} a_{mx,my} * cos(2*pi*mx*x) * cos(2*pi*my*y)
    # with symmetry constraints
    
    N = 2  # Maximum Fourier order
    # Symmetry: f(-x,y) = f(x,y), f(x,-y) = f(x,y), f(y,x) = f(x,y)
    # So we only need cos terms with non-negative indices, and mx >= my for full symmetry
    
    # Basis functions that respect D4 symmetry
    # cos(2*pi*m*x)*cos(2*pi*n*y) + cos(2*pi*m*y)*cos(2*pi*n*x) for m != n
    # cos(2*pi*m*x)*cos(2*pi*m*y) for m == n
    
    # Build basis
    basis_indices = []
    for m in range(N + 1):
        for n in range(m, N + 1):
            if m == n:
                basis_indices.append((m, n, 1))  # factor 1
            else:
                basis_indices.append((m, n, 2))  # factor 2 for symmetry
    
    n_params = len(basis_indices)
    
    # Sample at pixel centers
    x = np.linspace(-0.5, 0.5, size, endpoint=False)
    y = x
    
    field = np.zeros((size, size), dtype=np.float64)
    for p_idx, (m, n, factor) in enumerate(basis_indices):
        if p_idx >= len(params):
            break
        coeff = params[p_idx]
        cm = np.cos(2.0 * math.pi * m * x)
        cn = np.cos(2.0 * math.pi * n * x)
        if m == n:
            field += coeff * np.outer(cm, cn)
        else:
            field += coeff * (np.outer(cm, cn) + np.outer(cn, cm))
    
    # Add DC offset to control average filling fraction
    if len(params) > n_params:
        dc = params[n_params]
        field += dc
    
    # Threshold to binary
    cell = (field > 0.5).astype(int)
    
    # Enforce symmetry exactly
    cell = _apply_symmetry(cell, size)
    
    return cell


def _evaluate_cell(cell, size, eps_low, eps_high, n_pw, k_points):
    """Evaluate the gap-midgap ratio for a given cell."""
    # Build solver
    compute_bands = _build_pw_solver(size, eps_low, eps_high, n_pw, k_points)
    eps_field = cell.astype(np.float64)
    return compute_bands(eps_field)


def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing the TM band gap
    between bands 4 and 5.
    """
    t_start = time.time()
    DEADLINE = 220.0  # Safety margin before 230s limit
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem["lower_band"]  # 4
    polarization = problem.get("polarization", "TM")
    
    # Parameters for the continuous field
    # N=2 gives basis: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) = 6 params + DC = 7
    N_fourier = 2
    basis_indices = []
    for m in range(N_fourier + 1):
        for n in range(m, N_fourier + 1):
            basis_indices.append((m, n))
    n_basis = len(basis_indices)
    n_params = n_basis + 1  # + DC term
    
    # Initialize with a rod-like profile
    # DC term to set average filling
    params = np.zeros(n_params)
    params[-1] = 0.3  # DC offset
    params[0] = 0.2  # (0,0) term - actually this is redundant with DC
    # Let's use a different initialization
    params = np.array([0.0, 0.1, 0.05, 0.1, 0.0, 0.0, 0.25])
    
    # Generate initial cell
    cell = _generate_symmetric_cell(size, params)
    
    # Set up PW solver
    n_pw = 17  # 17x17 = 289 plane waves
    k_points = None  # Will be generated inside solver
    
    compute_bands = _build_pw_solver(size, eps_low, eps_high, n_pw, k_points)
    
    # Evaluate initial
    eps_field = cell.astype(np.float64)
    best_score = compute_bands(eps_field)
    best_params = params.copy()
    best_cell = cell.copy()
    
    # Gradient ascent optimization
    # Use finite differences for gradients
    step_size = 0.01
    learning_rate = 0.05
    max_iterations = 200
    
    for iteration in range(max_iterations):
        if time.time() - t_start > DEADLINE:
            break
        
        # Compute gradient via finite differences
        grad = np.zeros(n_params)
        for p in range(n_params):
            params_plus = params.copy()
            params_plus[p] += step_size
            cell_plus = _generate_symmetric_cell(size, params_plus)
            score_plus = compute_bands(cell_plus.astype(np.float64))
            
            params_minus = params.copy()
            params_minus[p] -= step_size
            cell_minus = _generate_symmetric_cell(size, params_minus)
            score_minus = compute_bands(cell_minus.astype(np.float64))
            
            grad[p] = (score_plus - score_minus) / (2.0 * step_size)
        
        # Gradient ascent step
        params_new = params + learning_rate * grad
        
        # Clip parameters to reasonable range
        params_new = np.clip(params_new, -1.0, 1.0)
        
        cell_new = _generate_symmetric_cell(size, params_new)
        score_new = compute_bands(cell_new.astype(np.float64))
        
        if score_new > best_score:
            best_score = score_new
            best_params = params_new.copy()
            best_cell = cell_new.copy()
            params = params_new.copy()
            # Reduce learning rate slightly
            learning_rate *= 0.95
        else:
            # Rejection: reduce learning rate
            learning_rate *= 0.8
            # Try smaller step
            if learning_rate < 0.001:
                break
        
        # Check time
        if time.time() - t_start > DEADLINE - 10:
            break
    
    # Final: ensure we return the best cell
    return best_cell.tolist()

# EVOLVE-BLOCK-END
