"""Combined program: fast vectorised band-structure evaluation + symmetry-aware local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_fourier_coeffs(cell, size, n_pw):
    """Compute Fourier coefficients of 1/eps for the cell using FFT."""
    eps = np.where(np.array(cell, dtype=float) > 0.5, 12.0, 1.0)
    inv_eps = 1.0 / eps
    fft_coeffs = np.fft.fftshift(np.fft.fft2(inv_eps)) / (size * size)
    
    half = n_pw // 2
    start = size // 2 - half
    end = size // 2 + half + 1
    block = fft_coeffs[start:end, start:end]
    
    # Return as a 2D array indexed by [gx+half, gy+half]
    return block

def _band_structure(cell, size, n_pw=13, n_bands=6, k_points=None):
    """Compute TM band structure at key points; return gap-midgap ratio."""
    if k_points is None:
        # 15 representative points in irreducible BZ
        k_points = [
            (0, 0), (1, 0), (0, 1), (0.5, 0.5),
            (0.3, 0.1), (0.1, 0.3), (0.4, 0.2), (0.2, 0.4),
            (0.7, 0.3), (0.3, 0.7), (0.8, 0.1), (0.1, 0.8),
            (0.6, 0.2), (0.2, 0.6), (0.9, 0.1),
        ]
    
    coeffs_block = _compute_fourier_coeffs(cell, size, n_pw)
    half = n_pw // 2
    N = n_pw * n_pw
    
    # Precompute G vectors
    Gx = np.arange(-half, half + 1)
    Gy = np.arange(-half, half + 1)
    GX, GY = np.meshgrid(Gx, Gy, indexing='ij')
    GX_flat = GX.flatten()
    GY_flat = GY.flatten()
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        # Diagonal matrix: (G+k)^2
        diag = ((GX_flat + kx)**2 + (GY_flat + ky)**2)
        
        # Build coupling matrix C: C[i,j] = (1/eps)_{G_i - G_j}
        # G_i - G_j has components (GX_flat[i]-GX_flat[j], GY_flat[i]-GY_flat[j])
        # We need coeffs_block at index [dgx+half, dgy+half]
        dgx = GX_flat[:, None] - GX_flat[None, :]
        dgy = GY_flat[:, None] - GY_flat[None, :]
        
        # Clip to valid range
        dgx_clipped = np.clip(dgx, -half, half)
        dgy_clipped = np.clip(dgy, -half, half)
        
        C = coeffs_block[dgx_clipped + half, dgy_clipped + half]
        
        # Solve: D @ A = omega^2 * C @ A  =>  D @ C^{-1} @ A = omega^2 A
        try:
            C_inv = np.linalg.inv(C)
            D_C_inv = np.diag(diag) @ C_inv
            D_C_inv = (D_C_inv + D_C_inv.conj().T) / 2
            eigvals = np.linalg.eigvalsh(D_C_inv)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= n_bands:
                omega4 = np.sqrt(max(eigvals[3], 0))
                omega5 = np.sqrt(max(eigvals[4], 0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 == float('inf') or max_omega4 <= 0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2
    if midgap <= 0:
        return 0.0
    return gap / midgap

def _get_sym_positions(i, j, size):
    """Return all symmetric positions for pixel (i,j)."""
    positions = [(i, j), (size-1-i, size-1-j), (j, i), (size-1-j, size-1-i)]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique

def _make_rod_cell(size, radius):
    """Create a symmetric rod cell."""
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < radius*radius:
                cell[i][j] = 1
    return cell

def _make_hole_cell(size, radius):
    """Create a cell with a dielectric background and a central air hole."""
    cell = [[1]*size for _ in range(size)]
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < radius*radius:
                cell[i][j] = 0
    return cell

def _make_square_rod(size, half_width):
    """Create a square rod."""
    cell = [[0]*size for _ in range(size)]
    center = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i - center + 0.5) < half_width and abs(j - center + 0.5) < half_width:
                cell[i][j] = 1
    return cell

def design(problem):
    """Symmetry-aware local search to maximize TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 230s wall-clock kill
    
    # Initialize with a standard rod design
    best_cell = _make_rod_cell(size, 0.2)
    best_score = _band_structure(best_cell, size, n_pw=13, n_bands=6)
    
    # Try several initial designs
    initial_designs = [
        _make_rod_cell(size, 0.15),
        _make_rod_cell(size, 0.25),
        _make_rod_cell(size, 0.3),
        _make_hole_cell(size, 0.15),
        _make_hole_cell(size, 0.2),
        _make_square_rod(size, 4),
        _make_square_rod(size, 6),
    ]
    
    for init_cell in initial_designs:
        if time.time() > deadline - 30:
            break
        score = _band_structure(init_cell, size, n_pw=13, n_bands=6)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in init_cell]
    
    # Local search with symmetry-aware flips
    cell = [row[:] for row in best_cell]
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 20:
        improved = False
        
        # Try a batch of random flips
        batch_size = 30
        for _ in range(batch_size):
            if time.time() > deadline - 20:
                break
            
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            
            positions = _get_sym_positions(i, j, size)
            
            # Get old values
            old_vals = [(pi, pj, cell[pi][pj]) for pi, pj in positions]
            
            # Flip
            new_val = 1 - cell[positions[0][0]][positions[0][1]]
            for pi, pj in positions:
                cell[pi][pj] = new_val
            
            score = _band_structure(cell, size, n_pw=13, n_bands=6)
            
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
                improved = True
                no_improve_count = 0
            else:
                # Revert
                for pi, pj, ov in old_vals:
                    cell[pi][pj] = ov
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Try a different initial design or larger perturbation
                no_improve_count = 0
                # Random restart with a perturbed version of best
                cell = [row[:] for row in best_cell]
                for _ in range(5):
                    i = np.random.randint(0, size)
                    j = np.random.randint(0, size)
                    positions = _get_sym_positions(i, j, size)
                    new_val = 1 - cell[positions[0][0]][positions[0][1]]
                    for pi, pj in positions:
                        cell[pi][pj] = new_val
                score = _band_structure(cell, size, n_pw=13, n_bands=6)
                if score > best_score:
                    best_score = score
                    best_cell = [row[:] for row in cell]
    
    return best_cell
# EVOLVE-BLOCK-END
