"""Design a symmetric 32x32 photonic-crystal unit cell to maximize the TM band-4/5 gap."""
import time
import math
import numpy as np

_SIZE = 32
_EPS_HIGH = 12.0
_EPS_LOW = 1.0
_EPS_INV_MEAN = 0.5 * (1.0 / _EPS_HIGH + 1.0 / _EPS_LOW)
_EPS_INV_AMP = 0.5 * (1.0 / _EPS_HIGH - 1.0 / _EPS_LOW)

# Plane-wave basis: 289 = 17x17 grid of G vectors
_G_RANGE = 8
_GV = [(m, n) for m in range(-_G_RANGE, _G_RANGE+1) for n in range(-_G_RANGE, _G_RANGE+1)]
_NPW = len(_GV)  # 289

# Precompute phase factors
_PIXEL_X = np.array([(i + 0.5) / _SIZE - 0.5 for i in range(_SIZE)])
_PIXEL_Y = np.array([(j + 0.5) / _SIZE - 0.5 for j in range(_SIZE)])
_XY = np.meshgrid(_PIXEL_X, _PIXEL_Y, indexing='ij')
_X_FLAT = _XY[0].flatten()
_Y_FLAT = _XY[1].flatten()

_PHASE = np.zeros((_NPW, _SIZE * _SIZE), dtype=np.complex128)
for ig, (gm, gn) in enumerate(_GV):
    phase = 2.0 * np.pi * (gm * _X_FLAT + gn * _Y_FLAT)
    _PHASE[ig, :] = np.exp(1j * phase)

# Precompute G index lookup: (dm, dn) -> index in _GV
_G_INDEX = {}
for idx, (gm, gn) in enumerate(_GV):
    _G_INDEX[(gm, gn)] = idx

# Precompute difference indices for S matrix construction
# S[i,j] needs coeffs[G[i] - G[j]]
_DIFF_IDX = np.zeros((_NPW, _NPW), dtype=np.int32)
for i in range(_NPW):
    gi, gj_ = _GV[i]
    for j in range(_NPW):
        gj, gk = _GV[j]
        dm = gi - gj
        dn = gj_ - gk
        if -_G_RANGE <= dm <= _G_RANGE and -_G_RANGE <= dn <= _G_RANGE:
            _DIFF_IDX[i, j] = _G_INDEX[(dm, dn)]
        else:
            _DIFF_IDX[i, j] = -1

# K-points: 30 points in the irreducible BZ
# The irreducible BZ for square lattice (1/8 of full BZ):
# 0 <= kx <= ky, kx + ky <= 0.5 (in units of 2*pi/a)
_KPOINTS = []
# High-symmetry points
_KPOINTS.append((0.0, 0.0))
_KPOINTS.append((0.25, 0.25))
_KPOINTS.append((0.0, 0.5))
_KPOINTS.append((0.125, 0.125))
_KPOINTS.append((0.0, 0.25))
_KPOINTS.append((0.125, 0.25))
# Additional points on a grid in the triangle
for i in range(5):
    for j in range(5):
        u = i / 4.0
        v = j / 4.0
        if u + v <= 1.0:
            kx = u * 0.25
            ky = u * 0.25 + v * 0.5
            _KPOINTS.append((kx, ky))

# Deduplicate
_seen = set()
_unique_kpts = []
for p in _KPOINTS:
    key = (round(p[0], 6), round(p[1], 6))
    if key not in _seen:
        _seen.add(key)
        _unique_kpts.append(p)
_KPOINTS = _unique_kpts[:30]

def _compute_fourier_coeffs(cell_flat):
    """Compute Fourier coefficients of 1/eps for a given cell (flat array)."""
    g = 2.0 * cell_flat - 1.0
    coeffs = np.zeros(_NPW, dtype=np.complex128)
    coeffs[0] = _EPS_INV_MEAN
    for ig in range(1, _NPW):
        f_coeff = np.vdot(_PHASE[ig, :], g) / (_SIZE * _SIZE)
        coeffs[ig] = _EPS_INV_AMP * f_coeff
    return coeffs

def _build_S_matrix(coeffs):
    """Build the (1/eps) matrix S from Fourier coefficients."""
    S = np.zeros((_NPW, _NPW), dtype=np.complex128)
    valid = _DIFF_IDX >= 0
    S[valid] = coeffs[_DIFF_IDX[valid]]
    return S

def _tm_bands_fast(cell):
    """Compute TM band-4/5 gap ratio using pure NumPy solver.
    
    Uses the formulation: D * E = omega^2 * S * E
    where D is diagonal with (k+G)^2 and S is the 1/eps matrix.
    
    We solve the generalized eigenvalue problem using:
    1. Cholesky factorization of S (if possible) to convert to standard problem
    2. Or use inverse iteration / Lanczos for the lowest eigenvalues
    """
    cell_flat = np.array(cell, dtype=np.float64).flatten()
    coeffs = _compute_fourier_coeffs(cell_flat)
    S = _build_S_matrix(coeffs)
    
    # Ensure S is Hermitian
    S = 0.5 * (S + S.conj().T)
    
    band_4_max = -np.inf
    band_5_min = np.inf
    
    for kx, ky in _KPOINTS:
        # Build D diagonal: D[i] = (k+G_i)^2
        D_diag = np.zeros(_NPW, dtype=np.float64)
        for i in range(_NPW):
            gm, gn = _GV[i]
            kx_tot = kx + gm
            ky_tot = ky + gn
            D_diag[i] = kx_tot * kx_tot + ky_tot * ky_tot
        
        # Solve D E = omega^2 S E
        # Convert to standard eigenvalue problem: S^{-1} D E = omega^2 E
        # Use Cholesky: S = L L^H, then L^H^{-1} D L^{-1} y = omega^2 y
        try:
            # Add small regularization to ensure positive definiteness
            S_reg = S + 1e-10 * np.eye(_NPW)
            L = np.linalg.cholesky(S_reg)
            # Compute L^{-1} D L^{-H}
            # First compute L^{-1}
            L_inv = np.linalg.inv(L)
            M = L_inv.conj().T @ np.diag(D_diag) @ L_inv
            # M is Hermitian
            M = 0.5 * (M + M.conj().T)
            # Compute eigenvalues
            eigvals = np.linalg.eigvalsh(M)
            # Take 4th and 5th (0-indexed: 3, 4)
            if len(eigvals) >= 5:
                w4_sq = eigvals[3]
                w5_sq = eigvals[4]
                if w4_sq > 0 and w5_sq > 0:
                    w4 = math.sqrt(w4_sq)
                    w5 = math.sqrt(w5_sq)
                    band_4_max = max(band_4_max, w4)
                    band_5_min = min(band_5_min, w5)
        except Exception:
            # Fallback: use dense eig
            try:
                A = np.linalg.solve(S_reg, np.diag(D_diag))
                A = 0.5 * (A + A.conj().T)
                eigvals = np.linalg.eigvalsh(A)
                if len(eigvals) >= 5:
                    w4_sq = eigvals[3]
                    w5_sq = eigvals[4]
                    if w4_sq > 0 and w5_sq > 0:
                        w4 = math.sqrt(w4_sq)
                        w5 = math.sqrt(w5_sq)
                        band_4_max = max(band_4_max, w4)
                        band_5_min = min(band_5_min, w5)
            except Exception:
                continue
    
    if band_4_max <= 0 or band_5_min <= 0:
        return -1.0
    
    gap = band_5_min - band_4_max
    midgap = (band_5_min + band_4_max) / 2.0
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / (band_5_min + band_4_max)

def _enforce_symmetry(cell):
    """Enforce all symmetry constraints on a 32x32 cell."""
    cell = np.array(cell, dtype=np.int8)
    # Vertical mirror: cell[i][j] = cell[31-i][j]
    for i in range(_SIZE // 2):
        for j in range(_SIZE):
            cell[_SIZE-1-i, j] = cell[i, j]
    # Horizontal mirror: cell[i][j] = cell[i][31-j]
    for i in range(_SIZE):
        for j in range(_SIZE // 2):
            cell[i, _SIZE-1-j] = cell[i, j]
    # Transpose: cell[i][j] = cell[j][i]
    for i in range(_SIZE):
        for j in range(i+1, _SIZE):
            if cell[i, j] != cell[j, i]:
                # Set both to 1 if either is 1 (union)
                val = max(cell[i, j], cell[j, i])
                cell[i, j] = val
                cell[j, i] = val
    return cell.tolist()

def _make_cell_from_params(params):
    """Create a 32x32 cell from symmetric parameters.
    
    params: [r0, r1, w1, amod, fmod]
    """
    r0 = params[0] if len(params) > 0 else 0.2
    r1 = params[1] if len(params) > 1 else 0.0
    w1 = params[2] if len(params) > 2 else 0.0
    amod = params[3] if len(params) > 3 else 0.0
    fmod = int(params[4]) if len(params) > 4 else 0
    
    cell = np.zeros((_SIZE, _SIZE), dtype=np.int8)
    
    for i in range(_SIZE):
        for j in range(_SIZE):
            x = (i + 0.5) / _SIZE - 0.5
            y = (j + 0.5) / _SIZE - 0.5
            r = math.sqrt(x*x + y*y)
            theta = math.atan2(y, x)
            
            eff_r = r0 + amod * math.cos(fmod * theta) if fmod > 0 else r0
            in_shape = r < eff_r
            
            if r1 > 0 and w1 > 0:
                in_ring = abs(r - r1) < w1
                in_shape = in_shape or in_ring
            
            cell[i, j] = 1 if in_shape else 0
    
    return _enforce_symmetry(cell)

def _optimize_cell(initial_params, time_limit):
    """Optimize cell parameters using local search."""
    best_params = list(initial_params)
    best_score = _tm_bands_fast(_make_cell_from_params(best_params))
    
    n_params = len(best_params)
    step_sizes = [0.02, 0.02, 0.01, 0.02, 1]
    
    deadline = time.time() + time_limit
    
    while time.time() < deadline - 5.0:
        improved = False
        for p in range(n_params):
            for direction in [-1, 1]:
                trial_params = list(best_params)
                trial_params[p] += direction * step_sizes[p]
                
                if p == 0:
                    trial_params[p] = max(0.05, min(0.45, trial_params[p]))
                elif p == 1:
                    trial_params[p] = max(0.0, min(0.45, trial_params[p]))
                elif p == 2:
                    trial_params[p] = max(0.0, min(0.2, trial_params[p]))
                elif p == 3:
                    trial_params[p] = max(0.0, min(0.1, trial_params[p]))
                elif p == 4:
                    trial_params[p] = max(0, min(8, int(trial_params[p])))
                
                trial_cell = _make_cell_from_params(trial_params)
                trial_score = _tm_bands_fast(trial_cell)
                
                if trial_score > best_score:
                    best_params = trial_params
                    best_score = trial_score
                    improved = True
                    step_sizes[p] *= 1.2
                else:
                    step_sizes[p] *= 0.7
        
        if not improved:
            all_small = all(s < 0.001 for s in step_sizes[:4])
            if all_small:
                break
    
    return best_params, best_score

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 220.0
    
    size = problem.get("pixels", 32)
    if size != _SIZE:
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                cell[i][j] = 1 if x*x + y*y < 0.04 else 0
        return cell
    
    initial_params = [0.2, 0.0, 0.0, 0.0, 0]
    
    candidates = [
        [0.2, 0.0, 0.0, 0.0, 0],
        [0.25, 0.0, 0.0, 0.0, 0],
        [0.15, 0.0, 0.0, 0.0, 0],
        [0.2, 0.35, 0.05, 0.0, 0],
        [0.2, 0.0, 0.0, 0.03, 4],
        [0.22, 0.38, 0.04, 0.02, 4],
    ]
    
    best_params = initial_params
    best_score = -np.inf
    
    for init in candidates:
        if time.time() > deadline - 30.0:
            break
        remaining = deadline - time.time() - 10.0
        if remaining < 5.0:
            break
        params, score = _optimize_cell(init, min(20.0, remaining))
        if score > best_score:
            best_score = score
            best_params = params
    
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        params, score = _optimize_cell(best_params, remaining)
        if score > best_score:
            best_score = score
            best_params = params
    
    cell = _make_cell_from_params(best_params)
    cell = _enforce_symmetry(cell)
    
    return cell
