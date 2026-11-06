"""Photonic crystal design with TM band gap optimization."""
import numpy as np
import time
import math

def _compute_fourier_coeffs(cell, N=17):
    """Compute exact Fourier coefficients of 1/eps for the cell."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    # Use FFT-based approach for speed
    # First, create the full (2N-1)x(2N-1) grid of coefficients
    # We'll compute via direct summation but vectorized
    
    xs = (np.arange(size) + 0.5) / size
    m_vals = np.arange(-(N-1), N)
    
    # For each (m,n) pair, compute the coefficient
    coeffs = np.zeros((2*N-1, 2*N-1), dtype=complex)
    for m_idx, m in enumerate(m_vals):
        for n_idx, n in enumerate(m_vals):
            # phase = exp(-2j*pi*(m*x + n*y))
            phase_x = np.exp(-2j * np.pi * m * xs)
            phase_y = np.exp(-2j * np.pi * n * xs)
            # Outer product gives 2D phase
            phase_2d = np.outer(phase_x, phase_y)
            coeffs[m_idx, n_idx] = np.sum(inv_eps * phase_2d) / (size * size)
    
    return coeffs

def _pwe_tm_bands(cell, kx, ky, N=17, n_bands=6):
    """Solve TM band structure at wave vector (kx, ky) using PWE method."""
    size = len(cell)
    coeffs = _compute_fourier_coeffs(cell, N)
    
    # Plane wave grid
    G = 2 * np.pi
    m_vals = np.arange(-(N-1), N)
    Gx = G * m_vals
    Gy = G * m_vals
    
    # Build eigenvalue matrix
    # TM: A[G,G'] = (1/eps)_{G-G'} * |k+G|^2
    n_pw = (2*N-1) * (2*N-1)
    
    # Map flat index to (m_idx, n_idx)
    pw_m = []
    pw_n = []
    for m_idx in range(2*N-1):
        for n_idx in range(2*N-1):
            pw_m.append(m_idx)
            pw_n.append(n_idx)
    
    pw_m = np.array(pw_m)
    pw_n = np.array(pw_n)
    
    # Precompute |k+G|^2
    kG2 = np.zeros(n_pw)
    for idx in range(n_pw):
        kx_total = kx + Gx[pw_m[idx]]
        ky_total = ky + Gy[pw_n[idx]]
        kG2[idx] = kx_total**2 + ky_total**2
    
    # Build matrix A
    A = np.zeros((n_pw, n_pw), dtype=complex)
    for idx1 in range(n_pw):
        m1 = pw_m[idx1]
        n1 = pw_n[idx1]
        for idx2 in range(n_pw):
            m2 = pw_m[idx2]
            n2 = pw_n[idx2]
            dm = m1 - m2
            dn = n1 - n2
            if 0 <= dm < 2*N-1 and 0 <= dn < 2*N-1:
                A[idx1, idx2] = coeffs[dm, dn] * kG2[idx1]
    
    # Solve
    vals, vecs = np.linalg.eigh(A)
    return vals[:n_bands].real

def _irreducible_bz_points(n_kx=6, n_ky=6):
    """Generate wave vectors in the irreducible Brillouin zone."""
    points = []
    for i in range(n_kx):
        for j in range(n_ky):
            kx = (i / (n_kx - 1)) * np.pi
            ky = (j / (n_ky - 1)) * np.pi
            if kx <= ky and (kx + ky) <= np.pi + 1e-10:
                points.append((kx, ky))
    
    # Add special high-symmetry points
    points.append((0, 0))
    points.append((np.pi, 0))
    points.append((np.pi/2, np.pi/2))
    
    # Remove duplicates
    unique_points = []
    for p in points:
        if not any(abs(p[0]-q[0]) < 1e-6 and abs(p[1]-q[1]) < 1e-6 for q in unique_points):
            unique_points.append(p)
    return unique_points

def _evaluate_gap(cell, k_points=None, n_bands=6, N=17):
    """Evaluate the gap-midgap ratio between bands 4 and 5."""
    if k_points is None:
        k_points = _irreducible_bz_points()
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        try:
            vals = _pwe_tm_bands(cell, kx, ky, N=N, n_bands=n_bands)
            if len(vals) >= 5:
                omega4 = vals[3]
                omega5 = vals[4]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 <= max_omega4:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _apply_symmetry(cell):
    """Ensure the cell has 4-fold rotational symmetry about the center."""
    size = len(cell)
    result = np.array(cell, dtype=float)
    
    # Average over the 4 symmetry operations
    rot90 = np.zeros_like(result)
    rot180 = np.zeros_like(result)
    rot270 = np.zeros_like(result)
    
    for i in range(size):
        for j in range(size):
            rot90[i, j] = result[size-1-j, i]
            rot180[i, j] = result[size-1-i, size-1-j]
            rot270[i, j] = result[j, size-1-i]
    
    averaged = (result + rot90 + rot180 + rot270) / 4.0
    result_sym = (averaged >= 0.5).astype(int)
    
    return result_sym.tolist()

def _initial_cell(size, radius_frac=0.2):
    """Create an initial dielectric rod cell."""
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(1 if x*x + y*y < radius_frac**2 else 0)
        cell.append(row)
    return _apply_symmetry(cell)

def _get_band_fields(cell, kx, ky, N=17, n_bands=6):
    """Get the field patterns for bands 4 and 5 at a given k-point."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    
    coeffs = _compute_fourier_coeffs(cell, N)
    
    G = 2 * np.pi
    m_vals = np.arange(-(N-1), N)
    Gx = G * m_vals
    Gy = G * m_vals
    
    n_pw = (2*N-1) * (2*N-1)
    pw_m = []
    pw_n = []
    for m_idx in range(2*N-1):
        for n_idx in range(2*N-1):
            pw_m.append(m_idx)
            pw_n.append(n_idx)
    
    pw_m = np.array(pw_m)
    pw_n = np.array(pw_n)
    
    kG2 = np.zeros(n_pw)
    for idx in range(n_pw):
        kx_total = kx + Gx[pw_m[idx]]
        ky_total = ky + Gy[pw_n[idx]]
        kG2[idx] = kx_total**2 + ky_total**2
    
    A = np.zeros((n_pw, n_pw), dtype=complex)
    for idx1 in range(n_pw):
        m1 = pw_m[idx1]
        n1 = pw_n[idx1]
        for idx2 in range(n_pw):
            m2 = pw_m[idx2]
            n2 = pw_n[idx2]
            dm = m1 - m2
            dn = n1 - n2
            if 0 <= dm < 2*N-1 and 0 <= dn < 2*N-1:
                A[idx1, idx2] = coeffs[dm, dn] * kG2[idx1]
    
    vals, vecs = np.linalg.eigh(A)
    
    if len(vals) < 5:
        return None, None
    
    # Reconstruct fields for bands 4 and 5
    fields = []
    for band_idx in [3, 4]:
        eigvec = vecs[:, band_idx]
        field = np.zeros((size, size), dtype=complex)
        for i in range(size):
            x = (i + 0.5) / size
            for j in range(size):
                y = (j + 0.5) / size
                val = 0.0
                for idx in range(n_pw):
                    kx_total = kx + Gx[pw_m[idx]]
                    ky_total = ky + Gy[pw_n[idx]]
                    val += eigvec[idx] * np.exp(1j * (kx_total * x + ky_total * y))
                field[i, j] = val
        fields.append(np.abs(field)**2)
    
    return fields[0], fields[1]

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell maximizing TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s kill
    
    # Initial design
    cell = _initial_cell(size, radius_frac=0.2)
    
    # Use a small set of k-points for search
    k_points = _irreducible_bz_points(n_kx=5, n_ky=5)
    search_k_points = k_points[:6]  # Use first 6 k-points for speed
    
    # Evaluate initial gap
    best_gap = _evaluate_gap(cell, search_k_points)
    best_cell = [row[:] for row in cell]
    
    # Optimization loop with field-guided moves
    max_iter = 50
    for iteration in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        cell_np = np.array(cell, dtype=int)
        best_move = None
        best_score = best_gap
        
        # Get field intensities at a few k-points for guidance
        # Use 2 k-points for guidance to save time
        guidance_k_points = search_k_points[:2]
        
        int4_sum = np.zeros((size, size))
        int5_sum = np.zeros((size, size))
        valid_fields = 0
        
        for kx, ky in guidance_k_points:
            try:
                int4, int5 = _get_band_fields(cell, kx, ky)
                if int4 is not None:
                    int4_sum += int4
                    int5_sum += int5
                    valid_fields += 1
            except:
                continue
        
        if valid_fields == 0:
            # Fallback to random moves
            pixels_to_try = [(np.random.randint(0, size), np.random.randint(0, size)) for _ in range(30)]
        else:
            # Generate guided moves
            pixels_to_try = []
            for _ in range(50):
                i = np.random.randint(0, size)
                j = np.random.randint(0, size)
                pixels_to_try.append((i, j))
        
        for i, j in pixels_to_try:
            if time.time() > deadline - 30:
                break
            
            # Determine preferred value based on field guidance
            if valid_fields > 0:
                int4_val = int4_sum[i, j]
                int5_val = int5_sum[i, j]
                # Prefer dielectric where band 4 is strong, air where band 5 is strong
                preferred = 1 if int4_val > int5_val else 0
            else:
                preferred = 1 - cell_np[i, j]  # Random flip
            
            new_val = preferred
            if new_val == cell_np[i, j]:
                new_val = 1 - new_val
            
            # Apply the move with symmetry
            new_cell = [row[:] for row in cell]
            new_cell[i][j] = new_val
            new_cell = _apply_symmetry(new_cell)
            
            # Evaluate
            gap = _evaluate_gap(new_cell, search_k_points)
            
            if gap > best_score:
                best_score = gap
                best_move = (i, j, new_val)
        
        if best_move is not None:
            i, j, val = best_move
            new_cell = [row[:] for row in cell]
            new_cell[i][j] = val
            cell = _apply_symmetry(new_cell)
            best_cell = [row[:] for row in cell]
            best_gap = best_score
        else:
            # No improvement, try a random restart with different initial radius
            if iteration > 10:
                break
    
    return _apply_symmetry(best_cell)
