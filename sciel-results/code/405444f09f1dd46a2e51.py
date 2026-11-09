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
_GV = [(m, n) for m in range(-_G_RANGE, _G_RANGE + 1) for n in range(-_G_RANGE, _G_RANGE + 1)]
_NPW = len(_GV)  # 289

# Precompute phase factors for Fourier coefficient computation
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
_KPOINTS = []
_KPOINTS.append((0.0, 0.0))
_KPOINTS.append((0.25, 0.25))
_KPOINTS.append((0.0, 0.5))
_KPOINTS.append((0.125, 0.125))
_KPOINTS.append((0.0, 0.25))
_KPOINTS.append((0.125, 0.25))
for i in range(5):
    for j in range(5):
        u = i / 4.0
        v = j / 4.0
        if u + v <= 1.0:
            kx = u * 0.25
            ky = u * 0.25 + v * 0.5
            _KPOINTS.append((kx, ky))

_seen = set()
_unique_kpts = []
for p in _KPOINTS:
    key = (round(p[0], 6), round(p[1], 6))
    if key not in _seen:
        _seen.add(key)
        _unique_kpts.append(p)
_KPOINTS = _unique_kpts[:30]


def _compute_fourier_coeffs(cell_flat):
    """Compute Fourier coefficients of 1/eps for a given cell (flat array of 0/1)."""
    g = 2.0 * cell_flat - 1.0  # maps 0->-1, 1->+1
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
    """Compute TM band-4/5 gap ratio.
    
    Solves D E = omega^2 S E where D is diagonal with (k+G)^2 and S is 1/eps matrix.
    Returns the gap-midgap ratio: 2*(min_w5 - max_w4)/(min_w5 + max_w4)
    """
    cell_flat = np.array(cell, dtype=np.float64).flatten()
    coeffs = _compute_fourier_coeffs(cell_flat)
    S = _build_S_matrix(coeffs)
    S = 0.5 * (S + S.conj().T)
    
    band_4_max = -np.inf
    band_5_min = np.inf
    
    for kx, ky in _KPOINTS:
        D_diag = np.zeros(_NPW, dtype=np.float64)
        for i in range(_NPW):
            gm, gn = _GV[i]
            kx_tot = kx + gm
            ky_tot = ky + gn
            D_diag[i] = kx_tot * kx_tot + ky_tot * ky_tot
        
        try:
            S_reg = S + 1e-10 * np.eye(_NPW)
            L = np.linalg.cholesky(S_reg)
            L_inv = np.linalg.inv(L)
            M = L_inv.conj().T @ np.diag(D_diag) @ L_inv
            M = 0.5 * (M + M.conj().T)
            eigvals = np.linalg.eigvalsh(M)
            if len(eigvals) >= 5:
                w4_sq = eigvals[3]
                w5_sq = eigvals[4]
                if w4_sq > 0 and w5_sq > 0:
                    w4 = math.sqrt(w4_sq)
                    w5 = math.sqrt(w5_sq)
                    band_4_max = max(band_4_max, w4)
                    band_5_min = min(band_5_min, w5)
        except Exception:
            try:
                S_reg = S + 1e-10 * np.eye(_NPW)
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
    """Enforce all symmetry constraints on a 32x32 cell (vertical, horizontal mirror, transpose)."""
    cell = np.array(cell, dtype=np.int8)
    # Vertical mirror: cell[i][j] = cell[31-i][j]
    for i in range(_SIZE // 2):
        for j in range(_SIZE):
            cell[_SIZE - 1 - i, j] = cell[i, j]
    # Horizontal mirror: cell[i][j] = cell[i][31-j]
    for i in range(_SIZE):
        for j in range(_SIZE // 2):
            cell[i, _SIZE - 1 - j] = cell[i, j]
    # Transpose: cell[i][j] = cell[j][i]
    for i in range(_SIZE):
        for j in range(i + 1, _SIZE):
            val = max(cell[i, j], cell[j, i])
            cell[i, j] = val
            cell[j, i] = val
    return cell.tolist()


def _make_cell_from_params(params):
    """Create a 32x32 cell from symmetric parameters.
    
    params: [r0, r1, w1, amod, fmod]
    r0: radius of central disk
    r1: radius of ring
    w1: width of ring
    amod: amplitude of angular modulation
    fmod: frequency of angular modulation (0 = no modulation)
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
            r = math.sqrt(x * x + y * y)
            theta = math.atan2(y, x)
            
            eff_r = r0 + amod * math.cos(fmod * theta) if fmod > 0 else r0
            in_shape = r < eff_r
            
            if r1 > 0 and w1 > 0:
                in_ring = abs(r - r1) < w1
                in_shape = in_shape or in_ring
            
            cell[i, j] = 1 if in_shape else 0
    
    return _enforce_symmetry(cell)


def _optimize_cell(initial_params, time_limit):
    """Optimize cell parameters using local search with adaptive step sizes."""
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


def _pixel_flip_optimize(cell, time_limit):
    """Optimize cell by flipping individual pixels with simulated annealing.
    
    This explores the full design space without restrictive parameterization.
    Uses a fast evaluator to fit many iterations within the time budget.
    """
    cell_arr = np.array(cell, dtype=np.int8)
    best_cell = cell_arr.copy()
    best_score = _tm_bands_fast(best_cell.tolist())
    
    # Precompute which pixels are in the fundamental domain (upper-left quadrant)
    # Due to symmetry, we only need to flip pixels in rows 0-15, cols 0-15
    # and then propagate symmetry
    flip_indices = []
    for i in range(_SIZE // 2):
        for j in range(_SIZE // 2):
            flip_indices.append((i, j))
    
    # Precompute the symmetry mapping: for each (i,j) in fundamental domain,
    # which pixels get set when we flip
    sym_map = {}
    for i, j in flip_indices:
        targets = [(i, j)]
        # Horizontal mirror
        targets.append((i, _SIZE - 1 - j))
        # Vertical mirror
        targets.append((_SIZE - 1 - i, j))
        targets.append((_SIZE - 1 - i, _SIZE - 1 - j))
        # Transpose
        targets.append((j, i))
        targets.append((j, _SIZE - 1 - i))
        targets.append((_SIZE - 1 - j, i))
        targets.append((_SIZE - 1 - j, _SIZE - 1 - i))
        # Unique
        sym_map[(i, j)] = list(set(targets))
    
    # Simulated annealing
    deadline = time.time() + time_limit
    T = 0.5
    cooling = 0.9995
    current_cell = cell_arr.copy()
    current_score = best_score
    
    n_iter = 0
    while time.time() < deadline - 2.0 and T > 0.01:
        # Pick a random pixel in fundamental domain
        idx = np.random.randint(len(flip_indices))
        i, j = flip_indices[idx]
        
        # Flip the pixel and all symmetric counterparts
        targets = sym_map[(i, j)]
        old_val = current_cell[i, j]
        new_val = 1 - old_val
        for (ti, tj) in targets:
            current_cell[ti, tj] = new_val
        
        # Evaluate
        trial_score = _tm_bands_fast(current_cell.tolist())
        n_iter += 1
        
        delta = trial_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(T, 1e-6)):
            current_score = trial_score
            if trial_score > best_score:
                best_score = trial_score
                best_cell = current_cell.copy()
        else:
            # Revert
            for (ti, tj) in targets:
                current_cell[ti, tj] = old_val
        
        T *= cooling
    
    return best_cell.tolist(), best_score, n_iter


def _generate_random_cell():
    """Generate a random symmetric 32x32 cell."""
    cell = np.zeros((_SIZE, _SIZE), dtype=np.int8)
    for i in range(_SIZE // 2):
        for j in range(_SIZE // 2):
            val = 1 if np.random.random() < 0.35 else 0
            cell[i, j] = val
            cell[i, _SIZE - 1 - j] = val
            cell[_SIZE - 1 - i, j] = val
            cell[_SIZE - 1 - i, _SIZE - 1 - j] = val
            cell[j, i] = val
            cell[j, _SIZE - 1 - i] = val
            cell[_SIZE - 1 - j, i] = val
            cell[_SIZE - 1 - j, _SIZE - 1 - i] = val
    return cell.tolist()


def _generate_ring_cell(r0, r1, w1, n_holes=0, hole_r=0.05):
    """Generate a cell with central disk and/or ring, optionally with holes."""
    cell = np.zeros((_SIZE, _SIZE), dtype=np.int8)
    for i in range(_SIZE):
        for j in range(_SIZE):
            x = (i + 0.5) / _SIZE - 0.5
            y = (j + 0.5) / _SIZE - 0.5
            r = math.sqrt(x * x + y * y)
            
            in_shape = False
            if r0 > 0:
                in_shape = r < r0
            if r1 > 0 and w1 > 0:
                in_shape = in_shape or (abs(r - r1) < w1)
            
            # Optional holes at symmetric positions
            if n_holes > 0 and in_shape:
                for h in range(n_holes):
                    angle = 2 * math.pi * h / n_holes
                    hx = hole_r * 0.8 * math.cos(angle)
                    hy = hole_r * 0.8 * math.sin(angle)
                    hr = math.sqrt((x - hx) ** 2 + (y - hy) ** 2)
                    if hr < hole_r:
                        in_shape = False
                        break
            
            cell[i, j] = 1 if in_shape else 0
    return _enforce_symmetry(cell)


def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    size = problem.get("pixels", 32)
    if size != _SIZE:
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                cell[i][j] = 1 if x * x + y * y < 0.04 else 0
        return cell
    
    np.random.seed(42)
    
    best_cell = None
    best_score = -np.inf
    
    # Phase 1: Evaluate a diverse set of initial candidates
    candidates = []
    
    # Parameterized shapes
    param_candidates = [
        [0.2, 0.0, 0.0, 0.0, 0],
        [0.25, 0.0, 0.0, 0.0, 0],
        [0.15, 0.0, 0.0, 0.0, 0],
        [0.2, 0.35, 0.05, 0.0, 0],
        [0.2, 0.0, 0.0, 0.03, 4],
        [0.22, 0.38, 0.04, 0.02, 4],
        [0.18, 0.3, 0.06, 0.0, 0],
        [0.22, 0.4, 0.03, 0.0, 0],
    ]
    
    for params in param_candidates:
        if time.time() > deadline - 60.0:
            break
        cell = _make_cell_from_params(params)
        score = _tm_bands_fast(cell)
        candidates.append((cell, score))
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Ring-based candidates
    ring_params = [
        (0.15, 0.35, 0.04, 4, 0.04),
        (0.18, 0.38, 0.03, 6, 0.03),
        (0.12, 0.32, 0.05, 0, 0.0),
        (0.2, 0.42, 0.02, 4, 0.05),
    ]
    
    for r0, r1, w1, nh, hr in ring_params:
        if time.time() > deadline - 60.0:
            break
        cell = _generate_ring_cell(r0, r1, w1, nh, hr)
        score = _tm_bands_fast(cell)
        candidates.append((cell, score))
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Random candidates
    for _ in range(5):
        if time.time() > deadline - 60.0:
            break
        cell = _generate_random_cell()
        score = _tm_bands_fast(cell)
        candidates.append((cell, score))
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Phase 2: Local optimization on top candidates
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Optimize top parameterized candidates
    for params in param_candidates[:4]:
        if time.time() > deadline - 40.0:
            break
        remaining = deadline - time.time() - 10.0
        if remaining < 5.0:
            break
        opt_params, opt_score = _optimize_cell(params, min(15.0, remaining))
        opt_cell = _make_cell_from_params(opt_params)
        if opt_score > best_score:
            best_score = opt_score
            best_cell = opt_cell
    
    # Phase 3: Pixel-level optimization with simulated annealing
    if time.time() < deadline - 30.0 and best_cell is not None:
        remaining = deadline - time.time() - 5.0
        if remaining > 15.0:
            sa_cell, sa_score, n_iter = _pixel_flip_optimize(best_cell, remaining)
            if sa_score > best_score:
                best_score = sa_score
                best_cell = sa_cell
    
    # Phase 4: Try pixel optimization from a few other starting points
    if time.time() < deadline - 15.0 and len(candidates) > 1:
        for cell, _ in candidates[1:3]:
            if time.time() > deadline - 10.0:
                break
            remaining = deadline - time.time() - 5.0
            if remaining < 10.0:
                break
            sa_cell, sa_score, _ = _pixel_flip_optimize(cell, remaining)
            if sa_score > best_score:
                best_score = sa_score
                best_cell = sa_cell
    
    # Final safety: ensure we have a valid cell
    if best_cell is None:
        best_cell = _make_cell_from_params([0.2, 0.0, 0.0, 0.0, 0])
    
    # Final symmetry enforcement
    best_cell = _enforce_symmetry(best_cell)
    
    return best_cell
