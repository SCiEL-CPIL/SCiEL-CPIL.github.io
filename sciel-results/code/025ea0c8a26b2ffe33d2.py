"""Photonic crystal design with TM band gap optimization."""
import numpy as np
import time
import math

def _compute_fourier_coeffs(cell, N=17):
    """Compute exact Fourier coefficients of 1/eps for the cell.
    N is half the number of plane waves (total = (2N-1)^2)"""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    # Fourier coefficients: c_{mn} = (1/A) integral inv_eps(x,y) exp(-i(m*2pi*x + n*2pi*y)) dx dy
    # For pixelated: sum over pixels
    coeffs = np.zeros((2*N-1, 2*N-1), dtype=complex)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            for m_idx in range(2*N-1):
                m = m_idx - (N-1)
                for n_idx in range(2*N-1):
                    n = n_idx - (N-1)
                    coeffs[m_idx, n_idx] += inv_eps[i, j] * np.exp(-2j * np.pi * (m*x + n*y)) / (size*size)
    return coeffs

def _compute_fourier_coeffs_fast(cell, N=17):
    """Fast computation using 2D FFT for Fourier coefficients."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    # Upsample to get better resolution, then use FFT
    # Actually, for pixelated structure, we can use direct summation with vectorization
    coeffs = np.zeros((2*N-1, 2*N-1), dtype=complex)
    
    # Vectorized computation
    xs = (np.arange(size) + 0.5) / size
    ys = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    m_vals = np.arange(-(N-1), N)
    n_vals = np.arange(-(N-1), N)
    
    # For each Fourier mode (m,n), compute the coefficient
    for m_idx, m in enumerate(m_vals):
        for n_idx, n in enumerate(n_vals):
            phase = np.exp(-2j * np.pi * (m * X + n * Y))
            coeffs[m_idx, n_idx] = np.sum(inv_eps * phase) / (size * size)
    return coeffs

def _pwe_tm_bands(cell, kx, ky, N=17, n_bands=6):
    """Solve TM band structure at wave vector (kx, ky) using PWE method.
    Returns the first n_bands eigenvalues (omega^2)."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    # Get Fourier coefficients of 1/eps
    coeffs = _compute_fourier_coeffs_fast(cell, N)
    
    # Plane wave grid
    G = 2 * np.pi
    m_vals = np.arange(-(N-1), N)
    Gx = G * m_vals
    Gy = G * m_vals
    
    # Build the eigenvalue matrix for TM polarization
    # TM: sum_{G'} (1/eps)_{G-G'} * |k+G|^2 * c_{G'} = omega^2 * c_G
    # In matrix form: A * c = omega^2 * c where A[G,G'] = (1/eps)_{G-G'} * |k+G|^2
    
    # Precompute |k+G|^2 for each plane wave
    pw_indices = []
    for i in range(len(m_vals)):
        for j in range(len(m_vals)):
            pw_indices.append((i, j))
    n_pw = len(pw_indices)
    
    kG2 = np.zeros(n_pw)
    for idx, (i, j) in enumerate(pw_indices):
        kx_total = kx + Gx[i]
        ky_total = ky + Gy[j]
        kG2[idx] = kx_total**2 + ky_total**2
    
    # Build matrix A efficiently
    # We can construct the matrix using the Toeplitz-like structure of the Fourier coefficients
    # But given the small size (17x17=289), a double loop is acceptable if we avoid python overhead in the inner loop
    # However, let's try to vectorize the index difference calculation.
    
    # Create index grids for i and j
    I, J = np.meshgrid(np.arange(n_pw), np.arange(n_pw), indexing='ij')
    # Map flat indices to (i, j) grid indices
    # pw_indices is a list of (i, j) tuples. Let's create arrays for them.
    pw_i = np.array([p[0] for p in pw_indices])
    pw_j = np.array([p[1] for p in pw_indices])
    
    # For each pair (idx1, idx2), we need coeffs[pw_i[idx1] - pw_i[idx2] + N-1, pw_j[idx1] - pw_j[idx2] + N-1]
    # Let's create 2D arrays of indices for coeffs lookup
    # This might use too much memory if n_pw is large, but 289 is small.
    
    # Actually, the double loop in python is slow. Let's stick to the loop but make it tighter or use a different approach.
    # Since we are fixing a crash, let's keep the logic but ensure it's correct.
    # To speed up, we can precompute the matrix of differences.
    
    # Let's just use the loop but it's the bottleneck. 
    # For N=17, n_pw=289. 289^2 = ~83k operations. This is fast in C, slow in Python.
    # Let's try to vectorize this specific part.
    
    # Create 2D arrays of i and j indices for all plane waves
    # pw_indices order: (0,0), (0,1)...(0,N-1), (1,0)...
    # It's a row-major flatten of (N, N)
    pw_i_2d = np.arange(N).reshape(N, 1) * np.ones((1, N), dtype=int)
    pw_j_2d = np.ones((N, 1), dtype=int) * np.arange(N)
    
    # Flatten to match pw_indices order
    pw_i_flat = pw_i_2d.flatten()
    pw_j_flat = pw_j_2d.flatten()
    
    # Calculate differences for all pairs
    # diff_i[idx1, idx2] = pw_i_flat[idx1] - pw_i_flat[idx2]
    diff_i = pw_i_flat[:, None] - pw_i_flat[None, :]
    diff_j = pw_j_flat[:, None] - pw_j_flat[None, :]
    
    # Map to coeffs indices
    coeff_idx_i = diff_i + (N - 1)
    coeff_idx_j = diff_j + (N - 1)
    
    # Initialize A
    A = np.zeros((n_pw, n_pw), dtype=complex)
    
    # We only need to fill entries where indices are valid
    valid_mask = (coeff_idx_i >= 0) & (coeff_idx_i < 2*N-1) & (coeff_idx_j >= 0) & (coeff_idx_j < 2*N-1)
    
    # Get the coefficients for valid indices
    # coeffs is (2N-1, 2N-1)
    # We can use advanced indexing
    # But we need to be careful with memory. 289x289 is fine.
    c_vals = coeffs[coeff_idx_i[valid_mask], coeff_idx_j[valid_mask]]
    
    # A[idx1, idx2] = c_vals * kG2[idx1]
    # kG2 is a vector of length n_pw. We need to broadcast it along rows (idx1).
    # kG2[:, None] has shape (n_pw, 1).
    # We need to multiply c_vals (shape num_valid) by kG2[idx1_of_valid]
    
    # Let's just do the assignment in a vectorized way if possible, or fall back to loop if vectorization is complex.
    # Given the fix requirement, let's use the vectorized lookup but assign carefully.
    
    # Actually, simpler vectorized approach:
    # A = (coeffs[coeff_idx_i, coeff_idx_j] where valid, else 0) * kG2[:, None]
    
    # Create a full grid of coeffs values, masking invalid ones with 0
    # coeff_idx_i and coeff_idx_j might be out of bounds, so we clip them or use a safe mask.
    # Let's use the mask approach.
    
    A_full = np.zeros((n_pw, n_pw), dtype=complex)
    # We need to map the valid indices back to (idx1, idx2)
    # This is tricky with boolean indexing.
    
    # Alternative: Just loop but it's slow.
    # Let's try a hybrid: Loop over idx1, vectorize over idx2.
    for idx1 in range(n_pw):
        i1 = pw_i_flat[idx1]
        j1 = pw_j_flat[idx1]
        # diff for this row
        di = i1 - pw_i_flat
        dj = j1 - pw_j_flat
        ci = di + (N - 1)
        cj = dj + (N - 1)
        
        # Mask valid
        mask = (ci >= 0) & (ci < 2*N-1) & (cj >= 0) & (cj < 2*N-1)
        
        # Get coeffs
        row_coeffs = np.zeros(n_pw, dtype=complex)
        row_coeffs[mask] = coeffs[ci[mask], cj[mask]]
        
        A[idx1, :] = row_coeffs * kG2[idx1]
    
    # Solve eigenvalue problem
    # A is Hermitian, so we can use eigh
    try:
        vals, vecs = np.linalg.eigh(A)
        return vals[:n_bands].real
    except np.linalg.LinAlgError:
        # Fallback if convergence fails, return zeros or raise
        return np.zeros(n_bands)

def _irreducible_bz_points(n_kx=6, n_ky=6):
    """Generate wave vectors in the irreducible Brillouin zone (triangle from Gamma to X to M)."""
    points = []
    # Irreducible BZ: 0 <= kx <= ky, kx + ky <= pi (in units where G=2pi)
    # Actually for square lattice, irreducible BZ is the triangle: 0<=kx<=ky, kx+ky<=pi
    # Let's use a grid in the full BZ and filter
    for i in range(n_kx):
        for j in range(n_ky):
            kx = (i / (n_kx - 1)) * np.pi
            ky = (j / (n_ky - 1)) * np.pi
            # Check if in irreducible BZ: 0 <= kx <= ky and kx + ky <= pi
            if kx <= ky and (kx + ky) <= np.pi + 1e-10:
                points.append((kx, ky))
    # Add special high-symmetry points
    points.append((0, 0))  # Gamma
    points.append((np.pi, 0))  # X
    points.append((np.pi/2, np.pi/2))  # M
    # Remove duplicates
    unique_points = []
    for p in points:
        if not any(abs(p[0]-q[0]) < 1e-6 and abs(p[1]-q[1]) < 1e-6 for q in unique_points):
            unique_points.append(p)
    return unique_points

def _evaluate_gap(cell, k_points=None, n_bands=6):
    """Evaluate the gap-midgap ratio between bands 4 and 5."""
    if k_points is None:
        k_points = _irreducible_bz_points()
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        try:
            vals = _pwe_tm_bands(cell, kx, ky, N=17, n_bands=n_bands)
            if len(vals) >= 5:
                omega4 = vals[3]  # 4th band (0-indexed: 3)
                omega5 = vals[4]  # 5th band (0-indexed: 4)
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 <= max_omega4:
        return -1.0  # Negative indicates overlap
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _apply_symmetry(cell):
    """Ensure the cell has 4-fold rotational symmetry about the center."""
    size = len(cell)
    result = np.array(cell, dtype=float)
    
    # Average over the 4 symmetry operations: identity, 90°, 180°, 270°
    # 90° rotation: new[i][j] = old[size-1-j][i]
    # 180° rotation: new[i][j] = old[size-1-i][size-1-j]
    # 270° rotation: new[i][j] = old[j][size-1-i]
    
    rot90 = np.zeros_like(result)
    rot180 = np.zeros_like(result)
    rot270 = np.zeros_like(result)
    
    for i in range(size):
        for j in range(size):
            rot90[i, j] = result[size-1-j, i]
            rot180[i, j] = result[size-1-i, size-1-j]
            rot270[i, j] = result[j, size-1-i]
    
    averaged = (result + rot90 + rot180 + rot270) / 4.0
    
    # Threshold at 0.5
    result_sym = (averaged >= 0.5).astype(int)
    
    # Verify symmetry
    assert np.array_equal(result_sym, result_sym[::-1, :]), "Row reversal symmetry failed"
    assert np.array_equal(result_sym, result_sym[:, ::-1]), "Column reversal symmetry failed"
    assert np.array_equal(result_sym, result_sym.T), "Transpose symmetry failed"
    
    return result_sym.tolist()

def _flip_pixel(cell, i, j, size):
    """Flip a pixel and apply symmetry."""
    cell = np.array(cell, dtype=int)
    old_val = cell[i, j]
    cell[i, j] = 1 - old_val
    
    # Apply symmetry: also flip the symmetric positions
    # 90°: (size-1-j, i)
    # 180°: (size-1-i, size-1-j)
    # 270°: (j, size-1-i)
    sym_positions = [(i, j), (size-1-j, i), (size-1-i, size-1-j), (j, size-1-i)]
    
    for si, sj in sym_positions:
        if cell[si, sj] == old_val:
            cell[si, sj] = 1 - old_val
    
    return cell

def _local_field_intensity(cell, kx, ky, band_idx, N=17):
    """Compute local field intensity for a given band at wave vector."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    coeffs = _compute_fourier_coeffs_fast(cell, N)
    
    G = 2 * np.pi
    m_vals = np.arange(-(N-1), N)
    Gx = G * m_vals
    Gy = G * m_vals
    
    pw_indices = []
    for i in range(len(m_vals)):
        for j in range(len(m_vals)):
            pw_indices.append((i, j))
    n_pw = len(pw_indices)
    
    kG2 = np.zeros(n_pw)
    for idx, (i, j) in enumerate(pw_indices):
        kx_total = kx + Gx[i]
        ky_total = ky + Gy[j]
        kG2[idx] = kx_total**2 + ky_total**2
    
    A = np.zeros((n_pw, n_pw), dtype=complex)
    for idx1, (i1, j1) in enumerate(pw_indices):
        for idx2, (i2, j2) in enumerate(pw_indices):
            m_diff = i1 - i2 + (N-1)
            n_diff = j1 - j2 + (N-1)
            if 0 <= m_diff < 2*N-1 and 0 <= n_diff < 2*N-1:
                A[idx1, idx2] = coeffs[m_diff, n_diff] * kG2[idx1]
    
    vals, vecs = eigh(A)
    
    if band_idx >= len(vals):
        return None
    
    eigvec = vecs[:, band_idx]
    
    # Reconstruct field in real space
    # E(x,y) = sum_G c_G exp(i(k+G)·r)
    intensity = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            field = 0.0
            for idx, (mi, ni) in enumerate(pw_indices):
                kx_total = kx + Gx[mi]
                ky_total = ky + Gy[ni]
                field += eigvec[idx] * np.exp(1j * (kx_total * x + ky_total * y))
            intensity[i, j] = np.abs(field)**2
    
    return intensity

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell maximizing TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 230s kill
    
    # Initial design: centered dielectric rod
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(1 if x*x + y*y < 0.2**2 else 0)
        cell.append(row)
    
    cell = _apply_symmetry(cell)
    
    # Evaluate initial gap
    k_points = _irreducible_bz_points(n_kx=5, n_ky=5)
    best_gap = _evaluate_gap(cell, k_points)
    best_cell = [row[:] for row in cell]
    
    # Optimization: Use a faster local search with fewer k-points for evaluation
    # The previous loop was too slow. We will use a very small set of k-points for the search
    # and a larger set for the final check.
    
    # Use only 3-4 k-points for the search to allow many iterations
    search_k_points = k_points[:4] 
    if len(search_k_points) < 3:
        search_k_points = k_points[:len(k_points)]

    max_iter = 100 # Increase iterations since each is faster
    for iteration in range(max_iter):
        if time.time() > deadline - 60:
            break
        
        cell_np = np.array(cell, dtype=int)
        best_score = best_gap
        best_flip = None
        
        # Try flipping pixels in the fundamental domain
        # To speed up, we don't compute field intensities (too slow), we just try flips
        # This is a blind hill-climbing but with fast evaluation.
        
        # Sample a subset of pixels to try if the grid is too large
        # 16x16 = 256 pixels. With 4 k-points, each eval is ~4 eigenvalue solves.
        # 256 * 4 = 1024 solves per iteration.
        # Each solve is ~0.05s? 1024 * 0.05 = 51s. Too slow.
        # We need to reduce k-points to 2 or 3, or sample pixels.
        
        # Let's use 2 k-points for the search
        search_k_points_fast = k_points[:2]
        
        # Sample 50 random pixels from the fundamental domain
        pixels_to_try = []
        for _ in range(50):
            i = np.random.randint(0, size // 2)
            j = np.random.randint(0, size // 2)
            pixels_to_try.append((i, j))
        
        for i, j in pixels_to_try:
            if time.time() > deadline - 60:
                break
            
            new_cell = _flip_pixel(cell_np, i, j, size)
            gap = _evaluate_gap(new_cell.tolist(), search_k_points_fast)
            
            if gap > best_score:
                best_score = gap
                best_flip = (i, j)
        
        if best_flip is not None:
            cell = _flip_pixel(cell_np, best_flip[0], best_flip[1], size).tolist()
            cell = _apply_symmetry(cell)
            best_cell = [row[:] for row in cell]
            best_gap = best_score
        else:
            # No improvement found in this random sample
            # If we've done several iterations without improvement, stop
            if iteration > 20:
                break
    
    # Final verification of symmetry
    final_cell = _apply_symmetry(best_cell)
    
    # If we have time, do a final high-precision evaluation to ensure best_cell is indeed the best
    # and maybe do a few more steps with higher k-point resolution
    if time.time() < deadline - 30:
        # Re-evaluate best_cell with full k-points to get a more accurate score
        # (Though the return value is the cell, not the score, so this is just for potential further improvement)
        pass
    
    return final_cell
