"""Optimized photonic crystal design for TM band 4-5 gap."""
import time
import numpy as np
from scipy.optimize import minimize

def _compute_fourier_coeffs(cell, N=17):
    """Compute Fourier coefficients of 1/epsilon for the cell."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps = np.where(eps > 0, 12.0, 1.0)
    inv_eps = 1.0 / eps
    fft_2d = np.fft.fft2(inv_eps)
    fft_2d = np.fft.fftshift(fft_2d)
    fft_2d = fft_2d / (size * size)
    center = size // 2
    half = N - 1
    start = center - half
    end = center + half + 1
    coeffs = fft_2d[start:end, start:end].copy()
    return coeffs

def _solve_bands(coeffs, kx, ky, N=17):
    """Solve for TM bands at wavevector (kx, ky)."""
    size = 2 * N - 1
    G_list = [(i - (N-1), j - (N-1)) for i in range(size) for j in range(size)]
    M = len(G_list)
    A = np.zeros((M, M), dtype=float)
    for i in range(M):
        Gx_i, Gy_i = G_list[i]
        kx_minus_Gx = kx - Gx_i
        ky_minus_Gy = ky - Gy_i
        for j in range(M):
            Gx_j, Gy_j = G_list[j]
            G_diff_x = Gx_i - Gx_j
            G_diff_y = Gy_i - Gy_j
            ci = G_diff_x + (N-1)
            cj = G_diff_y + (N-1)
            if 0 <= ci < size and 0 <= cj < size:
                eps_coeff = coeffs[ci, cj].real
                A[i, j] = (kx_minus_Gx**2 + ky_minus_Gy**2) * eps_coeff
    eigvals = np.linalg.eigvalsh(A)
    eigvals = np.sort(eigvals)
    eigvals_pos = np.maximum(eigvals, 0)
    omegas = np.sqrt(eigvals_pos)
    return omegas

def _evaluate_gap(cell, N=17):
    """Evaluate the gap-midgap ratio for TM bands 4 and 5."""
    coeffs = _compute_fourier_coeffs(cell, N)
    
    kpoints = []
    for t in np.linspace(0, 1, 8):
        kpoints.append((t, 0.0))
    for t in np.linspace(0, 0.5, 8):
        kpoints.append((1.0, t))
    for t in np.linspace(0, 1, 8):
        kpoints.append((0.5*(1-t), 0.5*(1-t)))
    kpoints.append((0.3, 0.2))
    kpoints.append((0.5, 0.3))
    kpoints.append((0.7, 0.4))
    kpoints.append((0.4, 0.4))
    kpoints.append((0.6, 0.5))
    kpoints = kpoints[:30]
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in kpoints:
        omegas = _solve_bands(coeffs, kx, ky, N)
        if len(omegas) >= 5:
            max_omega4 = max(max_omega4, omegas[3])
            min_omega5 = min(min_omega5, omegas[4])
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap == 0:
        return -1.0
    return 2.0 * gap / midgap

def _make_cell(params, size=32):
    """Create a symmetric 2x2 supercell arrangement of dielectric rods in air.
    params: [r, offset, r_center]
    r: radius of main rods (in pixels)
    offset: offset from quadrant centers (in pixels), same for x and y to maintain transpose symmetry
    r_center: radius of central feature (0 = none)
    """
    r, offset, r_center = params
    cell = np.zeros((size, size))
    qc = size / 4.0  # 8.0
    
    # For full C4v symmetry with 4-fold + reflections + transpose:
    # Use positions that form a complete symmetry orbit
    # Base position: (qc + offset, qc + offset)
    # Orbit under C4v:
    positions = set()
    base = (qc + offset, qc + offset)
    # Generate all symmetry-equivalent positions
    for (x, y) in [base]:
        # Identity
        positions.add((x, y))
        # 90 deg rotation about center (16, 16)
        cx, cy = size/2, size/2
        xr = cx + (y - cy)
        yr = cy - (x - cx)
        positions.add((xr, yr))
        # 180 deg
        positions.add((2*cx - x, 2*cy - y))
        # 270 deg
        xr = cx - (y - cy)
        yr = cy + (x - cx)
        positions.add((xr, yr))
        # Reflections
        positions.add((2*cx - x, y))  # horizontal flip
        positions.add((x, 2*cy - y))  # vertical flip
        positions.add((y, x))  # diagonal
        positions.add((2*cx - y, 2*cy - x))  # anti-diagonal
    
    positions = list(positions)
    
    for i in range(size):
        for j in range(size):
            px = i + 0.5
            py = j + 0.5
            for (cx_pos, cy_pos) in positions:
                if (px - cx_pos)**2 + (py - cy_pos)**2 < r**2:
                    cell[i, j] = 1
                    break
    
    if r_center > 0.01:
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                if cell[i, j] == 0:
                    px = i + 0.5
                    py = j + 0.5
                    if (px - center)**2 + (py - center)**2 < r_center**2:
                        cell[i, j] = 1
    
    return cell.tolist()

def _verify_symmetry(cell):
    """Check if cell has required symmetries."""
    size = len(cell)
    # Vertical flip
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size-1-i][j]:
                return False
    # Horizontal flip  
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[i][size-1-j]:
                return False
    # Transpose
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[j][i]:
                return False
    return True

def design(problem):
    """Design a 32x32 photonic crystal cell maximizing TM band 4-5 gap."""
    t_start = time.time()
    time_limit = 110.0
    
    size = problem.get("pixels", 32)
    
    best_score = -1.0
    best_cell = None
    best_params = None
    
    # Parameter ranges: r in [3, 8] pixels, offset in [-3, 3], r_center in [0, 4]
    initial_configs = [
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [4.5, 1.0, 0.0],
        [4.5, 0.0, 2.0],
        [5.0, 0.0, 2.0],
        [5.0, 1.5, 0.0],
        [4.0, 0.0, 3.0],
        [5.5, 0.5, 1.0],
    ]
    
    for params in initial_configs:
        if time.time() - t_start > time_limit - 30:
            break
        cell = _make_cell(params, size)
        if not _verify_symmetry(cell):
            continue
        score = _evaluate_gap(cell, N=17)
        if score > best_score:
            best_score = score
            best_cell = cell
            best_params = params[:]
    
    if best_cell is None:
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
    
    # Local optimization
    if time.time() - t_start < time_limit - 25:
        try:
            def neg_score(params):
                if time.time() - t_start > time_limit - 10:
                    return 1e6
                r, offset, r_center = params
                if r < 2 or r > 9 or abs(offset) > 4 or r_center < 0 or r_center > 5:
                    return 1e6
                cell = _make_cell(params, size)
                if not _verify_symmetry(cell):
                    return 1e6
                s = _evaluate_gap(cell, N=17)
                return -s
            
            result = minimize(neg_score, best_params, method='Nelder-Mead',
                             options={'maxiter': 40, 'xatol': 0.1, 'fatol': 0.005,
                                      'adaptive': True, 'maxfev': 60})
            
            if result.fun < -best_score and result.fun < 1e5:
                best_score = -result.fun
                best_cell = _make_cell(result.x, size)
                best_params = result.x
        except Exception:
            pass
    
    # Discrete polish
    if best_cell is not None and time.time() - t_start < time_limit - 15:
        for dr in [-0.5, 0.5, -1.0, 1.0]:
            if time.time() - t_start > time_limit - 10:
                break
            params = best_params[:]
            params[0] = max(2.0, min(9.0, params[0] + dr))
            cell = _make_cell(params, size)
            if not _verify_symmetry(cell):
                continue
            score = _evaluate_gap(cell, N=17)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_params = params
    
    return best_cell
