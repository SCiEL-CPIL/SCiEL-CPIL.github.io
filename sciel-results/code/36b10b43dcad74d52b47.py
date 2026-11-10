"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _build_cell(size, params):
    """
    Build a 32x32 symmetric cell from parameters.
    params[0]: inner radius (fraction of cell)
    params[1]: outer radius (fraction of cell)
    params[2]: ring thickness parameter (for shaping)
    params[3]: squareness (0=circle, 1=square)
    """
    r_inner = max(0.0, min(0.5, params[0]))
    r_outer = max(r_inner, min(0.5, params[1]))
    thickness = max(0.0, min(1.0, params[2]))
    squareness = max(0.0, min(1.0, params[3]))
    
    cell = [[0] * size for _ in range(size)]
    half = size / 2.0
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            
            # Mix between circular and square distance
            r_sq = max(abs(x), abs(y))
            r_mix = (1 - squareness) * r + squareness * r_sq
            
            # Ring condition
            if r_inner <= r_mix <= r_outer:
                # Add some thickness modulation
                mid = (r_inner + r_outer) / 2
                half_width = (r_outer - r_inner) / 2
                if half_width > 0:
                    dist_from_mid = abs(r_mix - mid) / half_width
                    if dist_from_mid < thickness or thickness >= 1.0:
                        cell[i][j] = 1
            else:
                cell[i][j] = 0
    
    return cell


def _compute_gap_estimate(cell, size=32, eps_high=12.0, n_pw=17):
    """
    Quick estimate of the band gap between bands 4 and 5.
    Uses a simplified plane-wave method with fewer plane waves for speed.
    This is an approximation to guide the optimizer.
    """
    # Build epsilon grid
    eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            eps[i, j] = 1.0 if cell[i][j] == 0 else eps_high
    
    # Fourier coefficients of 1/eps for TM mode
    # We need Fourier coefficients of epsilon^{-1}
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i, j] = 1.0 / eps[i, j]
    
    # Compute Fourier coefficients
    G = []
    for nx in range(-n_pw, n_pw + 1):
        for ny in range(-n_pw, n_pw + 1):
            G.append((nx, ny))
    
    N = len(G)
    G_arr = np.array(G, dtype=float)
    
    # Fourier coefficients of inv_eps
    c_G = np.zeros(N, dtype=complex)
    for idx, (nx, ny) in enumerate(G):
        c_G[idx] = 0.0
        for i in range(size):
            for j in range(size):
                c_G[idx] += inv_eps[i, j] * np.exp(-2j * np.pi * (nx * i / size + ny * j / size))
        c_G[idx] /= (size * size)
    
    # Build the matrix for each k-point
    # For TM: (k+G)^2 * A_{G'} = sum_{G''} (k+G-G'')^2 / eps_{G-G''} * A_{G''}
    # where eps_{G} are Fourier coefficients of 1/eps
    
    # We'll evaluate at a few k-points
    k_points = []
    # X point: (pi/a, 0) in units of 1/a
    # M point: (pi/a, pi/a)
    # Gamma: (0,0)
    # K point: (pi/a, pi/(2a)) - midpoint of X-M
    k_points.append((np.pi, 0))
    k_points.append((0, np.pi))
    k_points.append((np.pi, np.pi))
    k_points.append((np.pi / 2, np.pi / 2))
    k_points.append((np.pi, np.pi / 2))
    k_points.append((np.pi / 2, 0))
    
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    for kx, ky in k_points:
        # Build matrix
        A = np.zeros((N, N), dtype=complex)
        for i, (Gi_x, Gi_y) in enumerate(G):
            ki_x = kx + Gi_x
            ki_y = ky + Gi_y
            for j, (Gj_x, Gj_y) in enumerate(G):
                Gdiff_x = Gi_x - Gj_x
                Gdiff_y = Gi_y - Gj_y
                # Find index of Gdiff in G array
                # Since G is symmetric, Gdiff should be in G
                # But we need to find it
                found = False
                for idx, (gx, gy) in enumerate(G):
                    if gx == Gdiff_x and gy == Gdiff_y:
                        A[i, j] = (ki_x ** 2 + ki_y ** 2) * c_G[idx]
                        found = True
                        break
                if not found:
                    # Gdiff not in our set, approximate as 0
                    A[i, j] = 0.0
        
        # Eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(np.real(eigvals))
            # Take the 5th and 4th bands (0-indexed: 3 and 4)
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                if omega_5 > 0 and omega_4 > 0:
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
        except:
            pass
    
    if min_omega_5 == float('inf') or max_omega_4 <= 0:
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap <= 0:
        return -1.0
    
    return gap / midgap


def design(problem):
    """
    Design a symmetric 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    Uses a ring-shaped dielectric structure optimized via parameter search.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with a reasonable initial guess
    # A ring structure is known to produce good band gaps in TM mode
    best_params = [0.15, 0.35, 1.0, 0.0]
    best_score = _compute_gap_estimate(_build_cell(size, best_params), size, eps_high)
    
    # Try several candidate structures
    candidates = [
        [0.15, 0.35, 1.0, 0.0],   # Thin ring
        [0.20, 0.40, 1.0, 0.0],   # Thicker ring
        [0.10, 0.30, 1.0, 0.0],   # Narrower ring
        [0.25, 0.45, 1.0, 0.0],   # Wider ring
        [0.15, 0.35, 1.0, 0.3],   # Slightly square
        [0.15, 0.35, 1.0, 0.5],   # More square
        [0.20, 0.40, 1.0, 0.3],   # Thicker, slightly square
        [0.10, 0.30, 1.0, 0.5],   # Narrow, more square
        [0.15, 0.30, 1.0, 0.0],   # Thin ring
        [0.20, 0.35, 1.0, 0.0],   # Medium ring
    ]
    
    for params in candidates:
        if time.time() > deadline:
            break
        cell = _build_cell(size, params)
        score = _compute_gap_estimate(cell, size, eps_high)
        if score > best_score:
            best_score = score
            best_params = params[:]
    
    # Local optimization around best params
    if time.time() < deadline - 30:
        def objective(params):
            if time.time() > deadline:
                return 1.0  # Large penalty
            cell = _build_cell(size, params)
            return -_compute_gap_estimate(cell, size, eps_high)
        
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.001}
            )
            if time.time() < deadline:
                opt_params = np.clip(result.x, [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 1.0, 1.0])
                cell_opt = _build_cell(size, opt_params)
                score_opt = _compute_gap_estimate(cell_opt, size, eps_high)
                if score_opt > best_score:
                    best_score = score_opt
                    best_params = opt_params.tolist()
        except:
            pass
    
    # Build final cell
    cell = _build_cell(size, best_params)
    
    # Ensure symmetry (should already be symmetric by construction, but verify)
    # The cell is built symmetrically, so this should be fine
    
    return cell

# EVOLVE-BLOCK-END
