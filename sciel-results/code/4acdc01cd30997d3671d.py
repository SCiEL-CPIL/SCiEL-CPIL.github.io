"""Design a symmetric 32x32 photonic crystal unit cell for a large TM band gap."""
import time
import math
import numpy as np
from scipy.linalg import eigh

# Deadline guard: 230s wall-clock limit, return best with margin.
_START_TIME = time.time()
_DEADLINE = 215.0  # safety margin

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return _DEADLINE - _elapsed()

def _is_symmetric(cell):
    """Check 4-fold symmetry: flip rows, flip cols, transpose."""
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[n-1-i][j]:
                return False
            if cell[i][j] != cell[i][n-1-j]:
                return False
            if cell[i][j] != cell[j][i]:
                return False
    return True

def _compute_fourier_coeffs(cell, n_pw):
    """Compute Fourier coefficients of epsilon for a 32x32 pixel map.
    
    n_pw: number of plane waves per axis (odd, e.g., 17 for 289 total).
    Returns a 2D array of shape (n_pw, n_pw) with G indices from -(n_pw-1)//2 to +(n_pw-1)//2.
    """
    n = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0  # air
    eps[eps == 1] = 12.0  # dielectric
    
    half = (n_pw - 1) // 2
    G_vals = np.arange(-half, half + 1)
    
    coeffs = np.zeros((n_pw, n_pw), dtype=complex)
    for gi, gx in enumerate(G_vals):
        for gj, gy in enumerate(G_vals):
            # Fourier coefficient: (1/N^2) sum_{i,j} eps(i,j) exp(-i 2pi (gx*i + gy*j)/N)
            phase = np.exp(-2j * math.pi * (gx * np.arange(n)[:, None] + gy * np.arange(n)[None, :]) / n)
            coeffs[gi, gj] = np.sum(eps * phase) / (n * n)
    return coeffs

def _compute_band_gap(cell, n_pw=17, n_k=30):
    """Compute the gap-midgap ratio between bands 4 and 5 for TM polarization.
    
    Uses plane-wave method with exact Fourier coefficients.
    n_pw: plane waves per axis (17 gives 289 total)
    n_k: number of k-points
    Returns: (gap_ratio, max_omega_4, min_omega_5)
    """
    coeffs = _compute_fourier_coeffs(cell, n_pw)
    n = len(cell)
    half = (n_pw - 1) // 2
    G_vals = np.arange(-half, half + 1)
    
    # Build G vector list
    G_list = []
    for gx in G_vals:
        for gy in G_vals:
            G_list.append((gx, gy))
    G_list = np.array(G_list, dtype=float)
    nG = len(G_list)
    
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx >= 0, ky >= 0, kx+ky <= 1
    # Parametrize: kx = t1, ky = t1 + t2 where t1 in [0, 0.5], t2 in [0, 0.5-t1]
    # Actually for square lattice the IBZ is: 0 <= kx <= ky, kx >= 0, ky <= 0.5 (in units of 2pi/a)
    # More precisely: 0 <= kx <= ky, ky <= 0.5, kx >= 0
    # Let's use: kx = s, ky = s + t, s in [0, 0.25], t in [0, 0.5-s]
    # Or simpler: parametrize along the boundary and interior.
    
    # Generate k-points in the IBZ
    k_points = []
    # Along M-X: kx = 0, ky from 0 to 0.5
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.0, t))
    # Along X-G: kx from 0.5 to 0, ky = 0.5 (but in IBZ kx<=ky, so this is kx from 0 to 0.5, ky=0.5)
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0.5))
    # Along G-M: kx = ky, from 0 to 0.25 (since kx+ky <= 0.5 in the triangle, but actually for square IBZ it's kx=ky from 0 to 0.5)
    # Wait, the IBZ of square lattice is the triangle with vertices (0,0), (0.5,0), (0.5,0.5)
    # No, it's (0,0), (0.5,0), (0.5,0.5) - that's not right either.
    # The IBZ is: 0 <= kx <= ky, 0 <= ky <= 0.5. So it's a triangle with vertices (0,0), (0,0.5), (0.5,0.5).
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))
    
    # Remove duplicates and ensure we have ~30 points
    k_points = list(set(k_points))
    if len(k_points) > n_k:
        # Subsample
        indices = np.linspace(0, len(k_points)-1, n_k, dtype=int)
        k_points = [k_points[i] for i in indices]
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for kx, ky in k_points:
        # Build the TM matrix: (G-G0) x (G-G0) * epsilon_G-G0
        # For TM: [ (G - G0)^2 ] * [1/eps(G-G0)] 
        # Actually for TM polarization in 2D: the eigenvalue equation is
        # sum_{G'} (G-G')^2 / eps(G-G') * c_{G'} = omega^2 * c_G
        # where (G-G')^2 = (Gx-Gx')^2 + (Gy-Gy')^2 in units of 2pi/a
        
        # k is in units of 2pi/a
        k_vec = np.array([kx, ky])
        
        # Build matrix
        M = np.zeros((nG, nG), dtype=float)
        for i in range(nG):
            for j in range(nG):
                dG = G_list[i] - G_list[j]
                dG2 = dG[0]**2 + dG[1]**2
                # eps at G_i - G_j
                eps_coeff = coeffs[i, j].real  # should be real for symmetric eps
                if eps_coeff == 0:
                    M[i, j] = 0.0
                else:
                    # (k + G_i)^2 * eps_{G_i - G_j} ... 
                    # Actually the standard formulation:
                    # omega^2 c_G = sum_{G'} (k+G)^2 / eps(G-G') c_{G'}
                    # Wait, let me be more careful.
                    
                    # For TM: the wave equation gives
                    # (k+G)^2 c_G = sum_{G'} eps^{-1}_{G-G'} (k+G')^2 c_{G'}
                    # Hmm, this is getting complicated. Let me use the standard form.
                    
                    # Standard plane-wave method for TM:
                    # The eigenvalue problem is:
                    # sum_{G'} [(k+G_i) x (k+G_j)]^T [k+G_i] [k+G_j] / eps_{G_i-G_j} c_{G'} = omega^2 c_{G_i}
                    # For 2D TM (Ez nonzero):
                    # (k+G_i)^2 * c_{G_i} = sum_{G'} (1/eps_{G_i-G_j}) (k+G_j)^2 c_{G_j}
                    # No wait, that's not right either.
                    
                    # Let me use the correct formulation. For TM polarization (Ez):
                    # The equation is:
                    # sum_{G'} [(k+G_i)·(k+G_j)] / eps_{G_i-G_j} c_{G'} = omega^2 c_{G_i}
                    # where (k+G_i)·(k+G_j) = (kx+Gix)(kx+Gjx) + (ky+Gi y)(ky+Gjy)
                    
                    kGi = k_vec + G_list[i]
                    kGj = k_vec + G_list[j]
                    dot = kGi[0]*kGj[0] + kGi[1]*kGj[1]
                    if abs(eps_coeff) > 1e-10:
                        M[i, j] = dot / eps_coeff
                    else:
                        M[i, j] = 0.0
        
        # Solve eigenvalue problem
        try:
            eigvals, eigvecs = eigh(M)
            # Sort eigenvalues
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            
            # Bands 4 and 5 (1-indexed), so indices 3 and 4
            if len(eigvals) >= 5:
                omega_4 = math.sqrt(max(eigvals[3], 0))
                omega_5 = math.sqrt(max(eigvals[4], 0))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        except Exception:
            continue
    
    if min_omega_5 == float('inf') or max_omega_4 == 0:
        return 0.0, 0.0, 0.0
    
    gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
    return gap_ratio, max_omega_4, min_omega_5

def _make_cell_from_params(radius, shape='circle'):
    """Create a 32x32 cell with a central feature. Returns nested list."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    
    if shape == 'circle':
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < radius*radius:
                    cell[i][j] = 1
    elif shape == 'square':
        half = radius * size
        center = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - center + 0.5) < half and abs(j - center + 0.5) < half:
                    cell[i][j] = 1
    elif shape == 'ring':
        # Ring of dielectric
        r_inner = radius * 0.6
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x*x + y*y
                if r_inner*r_inner < r2 < radius*radius:
                    cell[i][j] = 1
    elif shape == 'cross':
        # Cross shape
        half_width = radius * size * 0.3
        center = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - center + 0.5) < half_width or abs(j - center + 0.5) < half_width:
                    cell[i][j] = 1
    elif shape == 'plus':
        # Plus/cross with variable arm width
        arm = int(radius * size * 0.4)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - center + 0.5) < arm or abs(j - center + 0.5) < arm:
                    cell[i][j] = 1
    
    # Ensure symmetry
    for i in range(size):
        for j in range(size):
            # Flip row symmetry
            if cell[i][j] != cell[size-1-i][j]:
                cell[size-1-i][j] = cell[i][j]
            # Flip col symmetry  
            if cell[i][j] != cell[i][size-1-j]:
                cell[i][size-1-j] = cell[i][j]
            # Transpose symmetry
            if cell[i][j] != cell[j][i]:
                cell[j][i] = cell[i][j]
    
    return cell

def _hill_climb(cell, n_iter=50):
    """Simple hill climbing: flip pixels and keep if gap improves."""
    size = len(cell)
    best_cell = [row[:] for row in cell]
    best_score = _compute_band_gap(best_cell)[0]
    
    # Collect candidate positions (only in upper-left quadrant for efficiency, then symmetrize)
    positions = []
    for i in range(size//2):
        for j in range(size//2):
            positions.append((i, j))
    
    np.random.seed(42)
    for it in range(n_iter):
        if _time_remaining() < 30:
            break
        
        # Randomly pick a position and flip
        idx = np.random.randint(len(positions))
        i, j = positions[idx]
        
        # Try flipping
        new_cell = [row[:] for row in best_cell]
        new_cell[i][j] = 1 - new_cell[i][j]
        
        # Symmetrize
        for ii in range(size):
            for jj in range(size):
                if new_cell[ii][jj] != new_cell[size-1-ii][jj]:
                    new_cell[size-1-ii][jj] = new_cell[ii][jj]
                if new_cell[ii][jj] != new_cell[ii][size-1-jj]:
                    new_cell[ii][size-1-jj] = new_cell[ii][jj]
                if new_cell[ii][jj] != new_cell[jj][ii]:
                    new_cell[jj][ii] = new_cell[ii][jj]
        
        score = _compute_band_gap(new_cell)[0]
        if score > best_score:
            best_cell = new_cell
            best_score = score
    
    return best_cell, best_score

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for max TM band gap (bands 4-5)."""
    size = problem.get("pixels", 32)
    
    # Start with several initial designs and evaluate
    candidates = []
    
    # Various radii for circle
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_make_cell_from_params(r, 'circle'))
    
    # Ring shapes
    for r in [0.3, 0.35, 0.4]:
        candidates.append(_make_cell_from_params(r, 'ring'))
    
    # Cross shapes
    for r in [0.3, 0.35, 0.4]:
        candidates.append(_make_cell_from_params(r, 'cross'))
    
    # Evaluate all candidates
    best_cell = None
    best_score = -1.0
    
    for cell in candidates:
        if _time_remaining() < 60:
            break
        try:
            score, _, _ = _compute_band_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
        except Exception:
            continue
    
    # Hill climb from the best
    if best_cell is not None and _time_remaining() > 30:
        best_cell, best_score = _hill_climb(best_cell, n_iter=30)
    
    # If we didn't find anything, fall back to the seed
    if best_cell is None:
        best_cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.04:
                    best_cell[i][j] = 1
    
    return best_cell
