"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _make_symmetric_cell(size, base_pattern):
    """
    Given a 1/4 cell (top-left quadrant), expand to full symmetric cell.
    The full cell must be invariant under:
      - 180 degree rotation (cell == cell[::-1] row-wise and column-wise)
      - transpose (cell == cell.T)
    This means the cell has D4 symmetry about the center.
    For a 32x32 cell, the center is between pixels (15,15) and (16,16).
    Symmetry: cell[i][j] = cell[31-i][31-j] = cell[j][i] = cell[31-j][31-i]
    So we need to define the top-left 16x16 quadrant, then mirror.
    """
    half = size // 2
    cell = [[0] * size for _ in range(size)]
    for i in range(half):
        for j in range(half):
            val = base_pattern[i][j]
            cell[i][j] = val
            cell[i][size - 1 - j] = val
            cell[size - 1 - i][j] = val
            cell[size - 1 - i][size - 1 - j] = val
    return cell


def _compute_fourier_coefficients(cell, size, n_pw):
    """
    Compute 1/epsilon Fourier coefficients for the pixel map.
    n_pw: number of plane waves along each dimension (total = n_pw^2)
    Returns a 2D array of shape (n_pw, n_pw) with 1/epsilon Fourier coefficients.
    """
    eps_low = 1.0
    eps_high = 12.0
    inv_eps = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if cell[i][j] == 1:
                inv_eps[i][j] = 1.0 / eps_high
            else:
                inv_eps[i][j] = 1.0 / eps_low
    
    # Fourier coefficients: G indices from -(n_pw-1)//2 to (n_pw-1)//2
    # For n_pw = 17 (odd), G goes from -8 to 8
    half_n = (n_pw - 1) // 2
    G_indices = np.arange(-half_n, half_n + 1)
    
    # Compute via direct sum: c_G = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(Gx*i + Gy*j)/size)
    # Use 2D FFT for efficiency
    # Shift inv_eps so that the origin is at (0,0)
    # The Fourier coefficient at G is: c_G = (1/N^2) * sum_{x,y} f(x,y) * exp(-2*pi*i*(Gx*x + Gy*y)/N)
    # numpy fft2 gives: F[kx,ky] = sum_{x,y} f[x,y] * exp(-2*pi*i*(kx*x + ky*y)/N)
    # So c_G = F[mod(Gx, N), mod(Gy, N)] / N^2
    
    F = np.fft.fft2(inv_eps)
    N = size
    coeff = np.zeros((n_pw, n_pw), dtype=np.complex128)
    for idx, gx in enumerate(G_indices):
        for idy, gy in enumerate(G_indices):
            kx = gx % N
            ky = gy % N
            coeff[idx][idy] = F[kx][ky] / (N * N)
    
    return coeff


def _band_structure_tm(coeff, kx, ky, n_bands=5, n_pw=17):
    """
    Solve the TM band structure eigenvalue problem at wavevector (kx, ky).
    TM polarization: the eigenvalue problem is
    sum_G' [ (k+G)^2 * delta_{GG'} + (1/epsilon)_{G-G'} * (k+G')^2 ] A_{G'} = omega^2 A_G
    
    Actually, for TM: (k+G)^2 A_G - sum_{G'} (1/eps)_{G-G'} (k+G')^2 A_{G'} = omega^2 A_G
    Wait, let me be more careful.
    
    For TM polarization in 2D photonic crystals:
    The eigenvalue equation is:
    sum_{G'} [ (k+G)^2 delta_{GG'} - (1/eps)_{G-G'} (k+G')^2 ] A_{G'} = omega^2 A_G
    
    Actually the standard form is:
    (k+G)^2 A_G = omega^2 sum_{G'} (1/eps)_{G-G'} A_{G'}
    
    So: A = omega^2 * M * A where M_{GG'} = (1/eps)_{G-G'}
    Or equivalently: M^{-1} A = (1/omega^2) A
    
    Let me use the standard plane-wave method:
    For TM: sum_{G'} (1/eps)_{G-G'} (k+G')^2 A_{G'} = omega^2 A_G
    
    So the matrix is: H_{GG'} = (1/eps)_{G-G'} * (k+G')^2
    And we solve H * A = omega^2 * A
    """
    half_n = (n_pw - 1) // 2
    G_indices = np.arange(-half_n, half_n + 1)
    
    # Build the matrix H
    # H_{GG'} = (1/eps)_{G-G'} * (k+G')^2
    # (1/eps)_{G-G'} = coeff[idx(G-Gx)][idx(G-Gy)] where idx maps G value to array index
    # G-G' ranges from -2*half_n to 2*half_n, but coeff only has indices for -half_n to half_n
    # So we need to handle the case where |G-G'| > half_n -> coefficient is 0 (truncation)
    
    H = np.zeros((n_pw, n_pw), dtype=np.complex128)
    for i in range(n_pw):
        for j in range(n_pw):
            Gx = G_indices[i]
            Gy = G_indices[i]  # Wait, this is wrong. Let me use proper 2D indexing.
            # Actually for 2D, I need to index properly.
            pass
    
    # Let me redo this with proper 2D indexing
    H = np.zeros((n_pw * n_pw, n_pw * n_pw), dtype=np.complex128)
    
    # Precompute (k+G)^2 for each G
    kG_sq = np.zeros((n_pw, n_pw), dtype=np.float64)
    for i in range(n_pw):
        for j in range(n_pw):
            Gx = G_indices[i]
            Gy = G_indices[j]
            kGx = kx + Gx
            kGy = ky + Gy
            kG_sq[i][j] = kGx * kGx + kGy * kGy
    
    for i in range(n_pw):
        for j in range(n_pw):
            row = i * n_pw + j
            Gx_i = G_indices[i]
            Gy_i = G_indices[j]
            for ip in range(n_pw):
                for jp in range(n_pw):
                    col = ip * n_pw + jp
                    Gx_j = G_indices[ip]
                    Gy_j = G_indices[jp]
                    dGx = Gx_i - Gx_j
                    dGy = Gy_i - Gy_j
                    # Check if dG is within the range of coeff
                    if -half_n <= dGx <= half_n and -half_n <= dGy <= half_n:
                        coeff_idx_x = dGx + half_n
                        coeff_idx_y = dGy + half_n
                        H[row][col] = coeff[coeff_idx_x][coeff_idx_y] * kG_sq[ip][jp]
                    # else: H[row][col] = 0 (truncation)
    
    # Solve eigenvalue problem
    # H is Hermitian (should be), so use eigh
    try:
        eigenvalues, _ = np.linalg.eigh(H)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)
        return eigenvalues[:n_bands]
    except:
        return [0.0] * n_bands


def _compute_gap_ratio(cell, size, n_pw=17, n_bands=5):
    """
    Compute the gap-midgap ratio between bands 4 and 5 (0-indexed: 3 and 4).
    """
    coeff = _compute_fourier_coefficients(cell, size, n_pw)
    
    # Sample points in the irreducible Brillouin zone
    # For square lattice, IBZ is: 0 <= kx <= ky, kx + ky <= pi (in units of 2*pi/a)
    # Actually, the IBZ for square lattice is: -pi/2 <= kx <= pi/2, -pi/2 <= ky <= pi/2, and |kx| <= |ky|, kx+ky >= 0
    # Let me use: 0 <= kx <= ky, kx + ky <= pi (in units where a=1, so k goes from -pi to pi)
    # Wait, the lattice constant is 1 (32 pixels = 1 unit cell), so the BZ goes from -pi to pi.
    # IBZ: 0 <= kx <= ky <= pi/2 (for the first BZ with a=1)
    # Actually, let me use the standard IBZ: 0 <= kx <= ky, kx + ky <= pi
    
    # Sample 30 points in the IBZ
    points = []
    # Along kx axis (ky=0)
    for kx in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        points.append((kx, 0.0))
    # Along diagonal (kx=ky)
    for k in [0.0, 0.25, 0.5, 0.75, 1.0]:
        points.append((k, k))
    # Along ky axis (kx=0)
    for ky in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        points.append((0.0, ky))
    # Additional interior points
    points.append((0.5, 0.5))
    points.append((0.3, 0.7))
    points.append((0.7, 0.3))
    points.append((0.2, 0.8))
    points.append((0.8, 0.2))
    points.append((0.4, 0.6))
    points.append((0.6, 0.4))
    
    # Take first 30 unique points
    seen = set()
    unique_points = []
    for p in points:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    unique_points = unique_points[:30]
    
    max_band4 = 0.0
    min_band5 = float('inf')
    
    for kx, ky in unique_points:
        bands = _band_structure_tm(coeff, kx, ky, n_bands=n_bands, n_pw=n_pw)
        if len(bands) >= 5:
            max_band4 = max(max_band4, bands[3])
            min_band5 = min(min_band5, bands[4])
    
    if min_band5 == float('inf') or max_band4 == 0.0:
        return 0.0
    
    gap = min_band5 - max_band4
    midgap = (min_band5 + max_band4) / 2.0
    if midgap == 0:
        return 0.0
    
    return 2.0 * gap / midgap


def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell that maximizes the TM band gap
    between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Leave 50s margin for evaluator overhead
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Start with a known good design: a dielectric rod or air hole
    # The seed uses a dielectric rod of radius 0.2. Let's try several candidate designs
    # and evaluate them, then do local optimization.
    
    def _eval_cell(cell):
        """Evaluate the gap ratio for a given cell."""
        try:
            return _compute_gap_ratio(cell, size, n_pw=17, n_bands=5)
        except:
            return -1.0
    
    # Candidate 1: Seed design (dielectric rod radius 0.2)
    def _make_rod(radius):
        base = [[0] * (size // 2) for _ in range(size // 2)]
        for i in range(size // 2):
            for j in range(size // 2):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    base[i][j] = 1
        return _make_symmetric_cell(size, base)
    
    # Candidate 2: Air hole (complement of rod)
    def _make_air_hole(radius):
        base = [[0] * (size // 2) for _ in range(size // 2)]
        for i in range(size // 2):
            for j in range(size // 2):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    base[i][j] = 0
                else:
                    base[i][j] = 1
        return _make_symmetric_cell(size, base)
    
    # Candidate 3: Square dielectric block
    def _make_square(half_size):
        base = [[0] * (size // 2) for _ in range(size // 2)]
        for i in range(size // 2):
            for j in range(size // 2):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) < half_size and abs(y) < half_size:
                    base[i][j] = 1
        return _make_symmetric_cell(size, base)
    
    # Candidate 4: Cross pattern
    def _make_cross(arm_width):
        base = [[0] * (size // 2) for _ in range(size // 2)]
        for i in range(size // 2):
            for j in range(size // 2):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) < arm_width or abs(y) < arm_width:
                    base[i][j] = 1
        return _make_symmetric_cell(size, base)
    
    # Candidate 5: Ring (annulus)
    def _make_ring(inner_r, outer_r):
        base = [[0] * (size // 2) for _ in range(size // 2)]
        for i in range(size // 2):
            for j in range(size // 2):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                if inner_r * inner_r < r2 < outer_r * outer_r:
                    base[i][j] = 1
        return _make_symmetric_cell(size, base)
    
    # Evaluate several candidates
    candidates = []
    
    # Rods with different radii
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_make_rod(r))
    
    # Air holes with different radii
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_make_air_hole(r))
    
    # Squares
    for hs in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_square(hs))
    
    # Crosses
    for aw in [0.05, 0.1, 0.15, 0.2]:
        candidates.append(_make_cross(aw))
    
    # Rings
    for ir, orr in [(0.1, 0.3), (0.15, 0.35), (0.1, 0.25), (0.2, 0.4), (0.05, 0.3)]:
        candidates.append(_make_ring(ir, orr))
    
    # Evaluate all candidates
    best_score = -1.0
    best_cell = None
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline:
            break
        score = _eval_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            print(f"Candidate {idx}: score={score:.6f}", flush=True)
    
    # Local optimization: flip pixels in the best cell to improve the score
    if best_cell is not None and time.time() < deadline - 60.0:
        current_cell = [row[:] for row in best_cell]
        current_score = best_score
        half = size // 2
        
        # Try flipping individual pixels in the top-left quadrant
        for i in range(half):
            if time.time() > deadline:
                break
            for j in range(half):
                if time.time() > deadline:
                    break
                # Try flipping this pixel
                old_val = current_cell[i][j]
                current_cell[i][j] = 1 - old_val
                # Propagate symmetry
                current_cell[i][size - 1 - j] = current_cell[i][j]
                current_cell[size - 1 - i][j] = current_cell[i][j]
                current_cell[size - 1 - i][size - 1 - j] = current_cell[i][j]
                
                new_score = _eval_cell(current_cell)
                
                if new_score > current_score:
                    current_score = new_score
                    best_score = new_score
                    best_cell = [row[:] for row in current_cell]
                else:
                    # Revert
                    current_cell[i][j] = old_val
                    current_cell[i][size - 1 - j] = old_val
                    current_cell[size - 1 - i][j] = old_val
                    current_cell[size - 1 - i][size - 1 - j] = old_val
        
        # Second pass of local optimization
        if time.time() < deadline - 30.0:
            for i in range(half):
                if time.time() > deadline:
                    break
                for j in range(half):
                    if time.time() > deadline:
                        break
                    old_val = current_cell[i][j]
                    current_cell[i][j] = 1 - old_val
                    current_cell[i][size - 1 - j] = current_cell[i][j]
                    current_cell[size - 1 - i][j] = current_cell[i][j]
                    current_cell[size - 1 - i][size - 1 - j] = current_cell[i][j]
                    
                    new_score = _eval_cell(current_cell)
                    
                    if new_score > current_score:
                        current_score = new_score
                        best_score = new_score
                        best_cell = [row[:] for row in current_cell]
                    else:
                        current_cell[i][j] = old_val
                        current_cell[i][size - 1 - j] = old_val
                        current_cell[size - 1 - i][j] = old_val
                        current_cell[size - 1 - i][size - 1 - j] = old_val
    
    # If we haven't found anything better, fall back to seed
    if best_cell is None or best_score <= 0:
        # Return seed design
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
    
    return best_cell

# EVOLVE-BLOCK-END
