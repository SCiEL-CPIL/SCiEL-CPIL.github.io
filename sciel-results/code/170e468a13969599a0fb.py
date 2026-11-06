"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _make_four_rod_cell(radius=6.0, centers=None):
    """Create a 32x32 cell with 4 circular rods."""
    size = 32
    if centers is None:
        centers = [(8, 8), (8, 23), (23, 8), (23, 23)]
    r2 = radius * radius
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            for (cx, cy) in centers:
                dx = i - cx
                dy = j - cy
                if dx * dx + dy * dy <= r2:
                    cell[i, j] = 1
                    break
    return cell

def _tm_eigenvalues_fast(cell, n_pw=17, kx=0.0, ky=0.0, bands=6):
    """
    Solve the TM eigenvalue problem for a given k-point.
    Uses the Hermitian form: M_{mn} = |K_m| |K_n| (1/eps)_{m-n}
    """
    size = cell.shape[0]
    # Indices for plane waves: -8 to 8 for n_pw=17
    indices = np.arange(-n_pw // 2, n_pw - n_pw // 2)
    
    # Compute 1/eps in real space
    eps_real = np.where(cell > 0, 12.0, 1.0)
    inv_eps_real = 1.0 / eps_real
    
    # Compute Fourier coefficients of 1/eps (size 32x32)
    inv_eps_ft = np.fft.fft2(inv_eps_real) / (size * size)
    
    # Precompute G vectors and K magnitudes
    Gx = indices * 2 * np.pi / size
    Gy = indices * 2 * np.pi / size
    Gx2d, Gy2d = np.meshgrid(Gx, Gy, indexing='ij')
    
    # K_m = k + G_m
    Kx = kx + Gx2d
    Ky = ky + Gy2d
    K_mag = np.sqrt(Kx**2 + Ky**2)
    
    # Build matrix M
    # M is n_pw*n_pw x n_pw*n_pw
    M = np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=float)
    
    # Flatten indices for easier access
    # We iterate over m (row) and n (col)
    for im in range(n_pw):
        for in_ in range(n_pw):
            row = im * n_pw + in_
            K_m = K_mag[im, in_]
            
            for jm in range(n_pw):
                for jn in range(n_pw):
                    col = jm * n_pw + jn
                    K_n = K_mag[jm, jn]
                    
                    # Difference in reciprocal indices
                    dm = im - jm
                    dn = in_ - jn
                    
                    # Map to FFT index (0..31)
                    idx_m = dm % size
                    idx_n = dn % size
                    
                    coeff = inv_eps_ft[idx_m, idx_n].real # Should be real
                    
                    M[row, col] = K_m * K_n * coeff
    
    # Ensure symmetry
    M = (M + M.T) / 2.0
    
    # Get eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(M)
    except:
        eigvals = np.zeros(bands)
    
    return eigvals[:bands]

def _quick_gap_estimate(cell, n_pw=17):
    """
    Estimate the gap between bands 4 and 5 for a given cell.
    """
    size = cell.shape[0]
    k_factor = 2 * np.pi / size
    
    # Reduced set of k-points for speed
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # M
        (0.5, 0.5),      # K
        (0.25, 0.0),     
        (0.5, 0.25),     
        (0.25, 0.25),    
        (0.4, 0.0),
        (0.45, 0.0),
        (0.3, 0.3),
        (0.4, 0.4),
    ]
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    for kx, ky in k_points:
        kx_actual = kx * k_factor
        ky_actual = ky * k_factor
        eigvals = _tm_eigenvalues_fast(cell, n_pw=n_pw, kx=kx_actual, ky=ky_actual, bands=6)
        if len(eigvals) >= 5:
            omega4 = np.sqrt(max(eigvals[3], 0))
            omega5 = np.sqrt(max(eigvals[4], 0))
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
    
    if min_omega5 == np.inf or max_omega4 == 0:
        return -1.0
    
    gap_ratio = 2 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    return gap_ratio

def _random_symmetric_cell(size=32):
    """Generate a random symmetric binary cell."""
    quarter = np.random.randint(0, 2, (size//2, size//2))
    half = np.hstack([quarter, quarter[:, ::-1]])
    cell = np.vstack([half, half[::-1, :]])
    return cell

def _mutate_cell(cell, mutation_rate=0.1):
    """Randomly flip some pixels while preserving symmetry."""
    cell_mut = cell.copy()
    size = cell.shape[0]
    for _ in range(int(mutation_rate * size * size // 4)):
        i = np.random.randint(0, size//2)
        j = np.random.randint(0, size//2)
        val = 1 - cell_mut[i, j]
        cell_mut[i, j] = val
        cell_mut[i, size-1-j] = val
        cell_mut[size-1-i, j] = val
        cell_mut[size-1-i, size-1-j] = val
    return cell_mut

def design(problem):
    """
    Design a 32x32 unit cell with a 2x2 basis of dielectric rods.
    """
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_cell = _make_four_rod_cell(radius=6.0)
    best_score = _quick_gap_estimate(best_cell, n_pw=17)
    
    current_cell = best_cell.copy()
    current_score = best_score
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        iterations += 1
        
        mutation_rate = 0.05 if no_improve_count < 20 else 0.15
        candidate = _mutate_cell(current_cell, mutation_rate=mutation_rate)
        
        candidate_score = _quick_gap_estimate(candidate, n_pw=17)
        
        if candidate_score > current_score:
            current_cell = candidate
            current_score = candidate_score
            no_improve_count = 0
            
            if candidate_score > best_score:
                best_cell = candidate.copy()
                best_score = candidate_score
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            if time.time() > deadline - 20.0:
                break
            current_cell = _random_symmetric_cell(size)
            current_score = _quick_gap_estimate(current_cell, n_pw=17)
            no_improve_count = 0
        
        if iterations % 10 == 0:
            for r in [5.0, 5.5, 6.5, 7.0]:
                candidate = _make_four_rod_cell(radius=r)
                score = _quick_gap_estimate(candidate, n_pw=17)
                if score > best_score:
                    best_cell = candidate
                    best_score = score
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
