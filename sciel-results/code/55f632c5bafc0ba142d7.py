"""Photonic-crystal band-gap optimizer for TM 4-5 gap on 32x32 square lattice."""
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 220.0  # wall-clock limit; we'll use ~190s for safety


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _symmetrize(cell):
    """Given a 32x32 cell, enforce full D4 symmetry about center."""
    size = len(cell)
    arr = np.array(cell, dtype=np.float64)
    # The cell is invariant under: reverse rows, reverse cols, transpose
    # So: arr[i][j] == arr[31-i][j] == arr[i][31-j] == arr[j][i]
    symmetries = [
        arr,
        arr[::-1, :],
        arr[:, ::-1],
        arr[::-1, ::-1],
        arr.T,
        arr.T[::-1, :],
        arr.T[:, ::-1],
        arr.T[::-1, ::-1],
    ]
    avg = np.mean(symmetries, axis=0)
    return (avg >= 0.5).astype(np.int32)


def _get_fourier_coeffs(cell, N=17):
    """Compute exact Fourier coefficients of 1/eps for the cell.
    
    cell: 32x32 array of 0/1 (1 = dielectric with eps=12, 0 = air with eps=1)
    N: truncation number (N=17 for 289 plane waves)
    Returns: 2D array of shape (32,32) where index (kx, ky) gives coefficient for G=(kx,ky)
             (with kx,ky in 0..31 range, negative indices wrap around)
    """
    size = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    
    # For TM polarization, we need 1/eps(x,y)
    # 1/eps = 1 for air (cell=0), 1/12 for dielectric (cell=1)
    inv_eps = np.where(np.array(cell) == 1, 1.0/eps_high, 1.0/eps_low)
    
    # Use FFT for efficiency
    # The DFT gives us: F[kx, ky] = sum_{i,j} f[i,j] * exp(-i 2*pi (kx*i + ky*j)/size)
    # We need: sum_{i,j} f[i,j] * exp(-i 2*pi (nx*(i+0.5)/size + ny*(j+0.5)/size))
    # = exp(-i pi nx/size) * exp(-i pi ny/size) * F[nx mod size, ny mod size]
    
    fft = np.fft.fft2(inv_eps)
    fft = fft / (size * size)  # normalize
    
    # Precompute phase factors
    phase_x = np.exp(-1j * np.pi * np.arange(size) / size)
    phase_y = np.exp(-1j * np.pi * np.arange(size) / size)
    phase_2d = phase_x[:, np.newaxis] * phase_y[np.newaxis, :]
    
    return fft * phase_2d


def _build_matrix(coeffs, G_indices, kx, ky, N=17):
    """Build the eigenvalue matrix for TM bands at a given k-point.
    
    coeffs: 2D array of Fourier coefficients of 1/eps
    G_indices: list of G vector indices (nx, ny)
    kx, ky: k-point coordinates
    N: truncation number
    """
    size = 32
    num_pw = len(G_indices)
    
    # Precompute (k+G)^2 for each plane wave
    kg_sq = np.zeros(num_pw, dtype=np.float64)
    for i, (nx, ny) in enumerate(G_indices):
        kg_sq[i] = (nx + kx)**2 + (ny + ky)**2
    
    # Build A matrix: A[i,j] = (1/eps)_{G_i - G_j}
    A = np.zeros((num_pw, num_pw), dtype=np.complex128)
    for i in range(num_pw):
        ni_x, ni_y = G_indices[i]
        for j in range(num_pw):
            nj_x, nj_y = G_indices[j]
            dnx = ni_x - nj_x
            dny = ni_y - nj_y
            # Wrap to [0, size)
            idx_x = dnx % size
            idx_y = dny % size
            A[i, j] = coeffs[idx_x, idx_y]
    
    # We need to solve: D c = omega^2 A c
    # where D is diagonal with kg_sq
    # Equivalent to: A^{-1} D c = omega^2 c
    
    # Compute A^{-1} D efficiently
    # A is Hermitian positive definite
    # Use Cholesky decomposition for stability
    try:
        # A should be Hermitian
        A = (A + A.conj().T) / 2
        
        # Cholesky decomposition: A = L L^H
        L = np.linalg.cholesky(A + 1e-12 * np.eye(num_pw))
        
        # Solve L X = D (column by column, but D is diagonal so X = L^{-1} D)
        # Actually, we want A^{-1} D = L^{-H} L^{-1} D
        # Let's compute L^{-1} first
        L_inv = np.linalg.inv(L)
        
        # A_inv_D = L_inv^H @ L_inv @ D
        # D is diagonal with kg_sq
        # L_inv @ D = L_inv * kg_sq (scale columns)
        L_inv_D = L_inv * kg_sq[np.newaxis, :]
        A_inv_D = L_inv.conj().T @ L_inv_D
        
        # Ensure Hermitian
        A_inv_D = (A_inv_D + A_inv_D.conj().T) / 2
        
    except Exception:
        # Fallback to direct inverse
        A_inv = np.linalg.inv(A)
        A_inv_D = A_inv @ np.diag(kg_sq)
        A_inv_D = (A_inv_D + A_inv_D.conj().T) / 2
    
    return A_inv_D


def _band_structure(cell, kpoints=None, N=17):
    """Compute TM band structure using plane-wave method.
    
    Returns: list of (kx, ky, omega_4, omega_5) for each k-point
    """
    # Get Fourier coefficients
    coeffs = _get_fourier_coeffs(cell, N)
    
    # Plane wave indices: -N..N for both x and y
    G_indices = [(nx, ny) for nx in range(-N, N+1) for ny in range(-N, N+1)]
    
    # k-points: irreducible BZ for square lattice
    if kpoints is None:
        kpoints = []
        # Along Gamma-X: (0,0) to (1,0)
        for t in np.linspace(0, 1, 10):
            kpoints.append((t, 0))
        # Along X-M: (1,0) to (0.5,0.5)
        for t in np.linspace(0, 1, 10):
            kpoints.append((1 - 0.5*t, 0.5*t))
        # Along M-Gamma: (0.5,0.5) to (0,0)
        for t in np.linspace(0, 1, 10):
            kpoints.append((0.5*(1-t), 0.5*(1-t)))
    
    results = []
    for kx, ky in kpoints:
        # Build the matrix
        A_inv_D = _build_matrix(coeffs, G_indices, kx, ky, N)
        
        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(A_inv_D)
        eigvals = np.sort(eigvals.real)
        
        # omega_4 = eigvals[3], omega_5 = eigvals[4]
        if len(eigvals) < 5:
            continue
            
        omega_4 = math.sqrt(max(eigvals[3], 0))
        omega_5 = math.sqrt(max(eigvals[4], 0))
        
        results.append((kx, ky, omega_4, omega_5))
    
    return results


def _score_cell(cell):
    """Compute the gap-midgap ratio for a given cell."""
    try:
        results = _band_structure(cell)
        if len(results) < 2:
            return -1.0
        
        max_omega_4 = max(r[2] for r in results)
        min_omega_5 = min(r[3] for r in results)
        
        numerator = min_omega_5 - max_omega_4
        denominator = min_omega_5 + max_omega_4
        
        if denominator <= 0:
            return -1.0
        
        return 2 * numerator / denominator
    except Exception:
        return -1.0


def _make_circle_hole_cell(size, radius, center=(16.0, 16.0)):
    """Create a cell with a single circular air hole."""
    cell = np.ones((size, size), dtype=np.int32)
    cx, cy = center
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5 - cx) / size
            dy = (j + 0.5 - cy) / size
            if dx*dx + dy*dy < radius**2:
                cell[i, j] = 0
    return _symmetrize(cell.tolist())


def _make_square_hole_cell(size, half_width, center=(16.0, 16.0)):
    """Create a cell with a single square air hole."""
    cell = np.ones((size, size), dtype=np.int32)
    cx, cy = center
    for i in range(size):
        for j in range(size):
            dx = abs(i + 0.5 - cx) / size
            dy = abs(j + 0.5 - cy) / size
            if dx < half_width and dy < half_width:
                cell[i, j] = 0
    return _symmetrize(cell.tolist())


def _make_cross_cell(size, width, length):
    """Create a cross-shaped air hole (plus sign)."""
    cell = np.ones((size, size), dtype=np.int32)
    cx, cy = size // 2, size // 2
    for i in range(size):
        for j in range(size):
            # Horizontal bar
            if abs(i - cy) < width/2 and abs(j - cx) < length/2:
                cell[i, j] = 0
            # Vertical bar
            if abs(j - cx) < width/2 and abs(i - cy) < length/2:
                cell[i, j] = 0
    return _symmetrize(cell.tolist())


def _make_ring_cell(size, inner_radius, outer_radius, center=(16.0, 16.0)):
    """Create a ring-shaped air hole (annulus)."""
    cell = np.ones((size, size), dtype=np.int32)
    cx, cy = center
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5 - cx) / size
            dy = (j + 0.5 - cy) / size
            r2 = dx*dx + dy*dy
            if inner_radius**2 <= r2 < outer_radius**2:
                cell[i, j] = 0
    return _symmetrize(cell.tolist())


def _make_diamond_hole_cell(size, radius, center=(16.0, 16.0)):
    """Create a diamond-shaped (rotated square) air hole."""
    cell = np.ones((size, size), dtype=np.int32)
    cx, cy = center
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5 - cx) / size
            dy = (j + 0.5 - cy) / size
            # Diamond: |dx| + |dy| < radius
            if abs(dx) + abs(dy) < radius:
                cell[i, j] = 0
    return _symmetrize(cell.tolist())


def _make_staggered_holes_cell(size, radius, offset):
    """Create a cell with staggered air holes for better band gap."""
    cell = np.ones((size, size), dtype=np.int32)
    cx, cy = size // 2, size // 2
    
    # Main hole at center
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5 - cx) / size
            dy = (j + 0.5 - cy) / size
            if dx*dx + dy*dy < radius**2:
                cell[i, j] = 0
    
    # Additional small holes at (0.25, 0.25) type positions
    # These break symmetry but we'll symmetrize
    for dx_off in [0.25, -0.25]:
        for dy_off in [0.25, -0.25]:
            hx = (cx + dx_off * size) / size
            hy = (cy + dy_off * size) / size
            for i in range(size):
                for j in range(size):
                    dx = (i + 0.5) / size - hx
                    dy = (j + 0.5) / size - hy
                    if dx*dx + dy*dy < (radius * offset)**2:
                        cell[i, j] = 0
    
    return _symmetrize(cell.tolist())


def _hill_climb(cell, max_iters=100):
    """Simple hill climbing to improve the gap."""
    best_cell = [row[:] for row in cell]
    best_score = _score_cell(cell)
    
    for it in range(max_iters):
        if _time_left() < 30:
            break
        
        # Try a few mutations
        best_trial = None
        best_trial_score = best_score
        
        for _ in range(5):
            # Random pixel flip mutation
            trial = np.array(best_cell, dtype=np.int32)
            n_flips = np.random.randint(2, 8)
            for _ in range(n_flips):
                i = np.random.randint(0, 32)
                j = np.random.randint(0, 32)
                trial[i, j] = 1 - trial[i, j]
            trial = _symmetrize(trial.tolist())
            
            trial_score = _score_cell(trial)
            if trial_score > best_trial_score:
                best_trial_score = trial_score
                best_trial = trial
        
        if best_trial is not None:
            best_cell = best_trial
            best_score = best_trial_score
        else:
            # Stuck, try a bigger mutation
            trial = np.array(best_cell, dtype=np.int32)
            n_flips = np.random.randint(10, 25)
            for _ in range(n_flips):
                i = np.random.randint(0, 32)
                j = np.random.randint(0, 32)
                trial[i, j] = 1 - trial[i, j]
            trial = _symmetrize(trial.tolist())
            trial_score = _score_cell(trial)
            if trial_score > best_score:
                best_cell = trial
                best_score = trial_score
    
    return best_cell, best_score


def design(problem):
    """Design a 32x32 photonic crystal unit cell for maximum TM 4-5 band gap."""
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Generate a diverse set of initial candidates
    candidates = []
    
    # Single circular hole with various radii
    for radius in [0.12, 0.15, 0.18, 0.22, 0.25]:
        candidates.append(_make_circle_hole_cell(size, radius))
    
    # Square holes
    for half_width in [0.08, 0.12, 0.15, 0.18]:
        candidates.append(_make_square_hole_cell(size, half_width))
    
    # Cross shapes
    for width, length in [(0.08, 0.3), (0.1, 0.35), (0.12, 0.4)]:
        candidates.append(_make_cross_cell(size, width, length))
    
    # Ring shapes
    for inner, outer in [(0.1, 0.2), (0.12, 0.22), (0.15, 0.25)]:
        candidates.append(_make_ring_cell(size, inner, outer))
    
    # Diamond shapes
    for radius in [0.15, 0.2, 0.25]:
        candidates.append(_make_diamond_hole_cell(size, radius))
    
    # Staggered holes
    for radius, offset in [(0.12, 0.5), (0.15, 0.4), (0.1, 0.6)]:
        candidates.append(_make_staggered_holes_cell(size, radius, offset))
    
    # Evaluate all candidates
    best_cell = None
    best_score = -1.0
    
    for cell in candidates:
        if _time_left() < 15:
            break
        
        score = _score_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Local search from the best candidate
    if best_cell is not None and _time_left() > 60:
        best_cell, best_score = _hill_climb(best_cell, max_iterations=50)
    
    # Try local search from a few other good candidates
    if _time_left() > 90:
        # Find second-best candidate
        scored = []
        for cell in candidates:
            if _time_left() < 15:
                break
            score = _score_cell(cell)
            scored.append((score, cell))
        
        scored.sort(key=lambda x: -x[0])
        
        for score, cell in scored[1:4]:  # Try 2nd, 3rd, 4th best
            if _time_left() < 30:
                break
            if score > best_score * 0.8:  # Only if reasonably good
                trial_cell, trial_score = _hill_climb(cell, max_iterations=30)
                if trial_score > best_score:
                    best_score = trial_score
                    best_cell = trial_cell
    
    # Final symmetry check
    if best_cell is None:
        # Fallback: simple circular hole
        best_cell = _make_circle_hole_cell(size, 0.18)
    
    best_cell = _symmetrize(best_cell)
    
    return best_cell
