import time
import numpy as np

_DEADLINE = 220.0
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _enforce_symmetry(cell):
    n = len(cell)
    arr = np.array(cell, dtype=float)
    ops = [
        arr,
        arr[:, ::-1].copy(),
        arr[::-1, :].copy(),
        arr[::-1, ::-1].copy(),
        arr.T.copy(),
        arr.T[:, ::-1].copy(),
        arr.T[::-1, :].copy(),
        arr.T[::-1, ::-1].copy(),
    ]
    avg = np.mean(ops, axis=0)
    return (avg > 0.5).astype(int).tolist()

def _fourier_coefficients(cell, eps_low, eps_high):
    n = len(cell)
    eps_map = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            eps_map[i, j] = eps_high if cell[i][j] == 1 else eps_low
    inv_eps = 1.0 / eps_map
    max_order = 8
    coeffs = {}
    for hx in range(-max_order, max_order + 1):
        for hy in range(-max_order, max_order + 1):
            phase_i = np.exp(-2j * np.pi * hx * np.arange(n) / n)
            phase_j = np.exp(-2j * np.pi * hy * np.arange(n) / n)
            phase = np.outer(phase_i, phase_j)
            coeff = np.sum(inv_eps * phase) / (n * n)
            coeffs[(hx, hy)] = coeff
    return coeffs

def _build_matrices(coeffs, kx, ky, n_pw=17):
    half = (n_pw - 1) // 2
    G_list = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_list.append((gx, gy))
    n = len(G_list)
    M = np.zeros((n, n), dtype=complex)
    for i, (gx, gy) in enumerate(G_list):
        kx_g = kx + gx
        ky_g = ky + gy
        k_sq = kx_g * kx_g + ky_g * ky_g
        for j, (gx2, gy2) in enumerate(G_list):
            dx = gx - gx2
            dy = gy - gy2
            c = coeffs.get((dx, dy), 0.0)
            M[i, j] = k_sq * c
    return M

def _compute_bands_fast(coeffs, kx, ky, n_pw=17, n_bands=6):
    M = _build_matrices(coeffs, kx, ky, n_pw)
    try:
        eigvals = np.linalg.eigvalsh(M)
    except:
        eigvals = np.linalg.eigvals(M)
        eigvals = np.sort(np.real(eigvals))
    return eigvals[:n_bands]

def _irreducible_bz_points(n=30):
    points = []
    for t in np.linspace(0, np.pi, 5):
        points.append((t, 0.0))
    for t in np.linspace(0, np.pi, 5):
        points.append((np.pi, t))
    for t in np.linspace(0, np.pi, 5):
        points.append((t, t))
    for t1 in np.linspace(0.2, 0.8, 3):
        for t2 in np.linspace(0.1, 0.5, 3):
            kx = t1 * np.pi
            ky = t2 * np.pi
            if ky <= kx:
                points.append((kx, ky))
    seen = set()
    unique_points = []
    for p in points:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
        if len(unique_points) >= n:
            break
    return unique_points

def _evaluate_cell(cell, eps_low, eps_high, n_pw=17, n_kpts=30, n_bands=6):
    coeffs = _fourier_coefficients(cell, eps_low, eps_high)
    kpts = _irreducible_bz_points(n_kpts)
    max_omega4 = 0.0
    min_omega5 = np.inf
    for (kx, ky) in kpts:
        eigvals = _compute_bands_fast(coeffs, kx, ky, n_pw, n_bands)
        if len(eigvals) >= 5:
            omega4 = np.sqrt(max(eigvals[3], 0))
            omega5 = np.sqrt(max(eigvals[4], 0))
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
    if min_omega5 == np.inf or max_omega4 == 0:
        return -1.0
    gap = min_omega5 - max_omega4
    midgap = 0.5 * (min_omega5 + max_omega4)
    if midgap <= 0:
        return -1.0
    return gap / midgap

def _generate_rod(radius, size=32):
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(1 if x * x + y * y < radius * radius else 0)
        cell.append(row)
    return cell

def _generate_ring(inner_r, outer_r, size=32):
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r2 = x * x + y * y
            row.append(1 if inner_r * inner_r < r2 < outer_r * outer_r else 0)
        cell.append(row)
    return cell

def _generate_cross(arm_width, arm_len, size=32):
    cell = [[0] * size for _ in range(size)]
    center = size // 2
    hw = arm_width // 2
    hl = arm_len // 2
    for i in range(size):
        for j in range(size):
            if (abs(i - center) <= hw and abs(j - center) <= hl) or \
               (abs(i - center) <= hl and abs(j - center) <= hw):
                cell[i][j] = 1
    return cell

def _generate_moire(freq1, freq2, angle, size=32):
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            v = np.sin(2 * np.pi * freq1 * x) + np.sin(2 * np.pi * freq2 * (x * np.cos(angle) + y * np.sin(angle)))
            row.append(1 if v > 0 else 0)
        cell.append(row)
    return cell

def _generate_noise_cell(seed, freq, persistence, lacunarity, threshold, size=32):
    rng = np.random.RandomState(seed)
    noise = np.zeros((size, size))
    amp = 1.0
    for octave in range(3):
        res = max(2, size // (2 ** octave))
        low_noise = rng.rand(res, res)
        upsampled = np.kron(low_noise, np.ones((size // res, size // res)))[:size, :size]
        noise += amp * upsampled
        amp *= persistence
    noise /= (1 + persistence + persistence ** 2)
    cell = (noise > threshold).astype(int).tolist()
    return cell

def _local_optimize(cell, eps_low, eps_high, max_flips=80, n_pw=17, n_kpts=10):
    n = len(cell)
    arr = np.array(cell, dtype=int)
    best_score = _evaluate_cell(arr.tolist(), eps_low, eps_high, n_pw, n_kpts)
    best_cell = arr.copy()
    rng = np.random.RandomState(42)
    for iteration in range(max_flips // 3):
        if _time_left() < 5:
            break
        improved = False
        for _ in range(3):
            i = rng.randint(0, n // 2)
            j = rng.randint(i, n - i)
            new_arr = arr.copy()
            new_arr[i, j] = 1 - new_arr[i, j]
            new_cell = _enforce_symmetry(new_arr.tolist())
            score = _evaluate_cell(new_cell, eps_low, eps_high, n_pw, n_kpts)
            if score > best_score:
                best_score = score
                best_cell = np.array(new_cell, dtype=int)
                improved = True
        if not improved and iteration > 30:
            break
    return best_cell.tolist(), best_score

def design(problem):
    global _start_time
    _start_time = time.time()
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    candidates = []
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append(_generate_rod(r, size))
    for inner, outer in [(0.1, 0.25), (0.15, 0.3), (0.2, 0.35), (0.1, 0.2), (0.15, 0.25), (0.2, 0.4), (0.05, 0.3)]:
        candidates.append(_generate_ring(inner, outer, size))
    for w, l in [(4, 8), (6, 10), (8, 12), (4, 12), (6, 14), (2, 10), (4, 14)]:
        candidates.append(_generate_cross(w, l, size))
    for f1, f2, ang in [(2, 3, 0.3), (2, 4, 0.5), (3, 5, 0.4), (2, 3, 0.7), (3, 4, 0.2), (4, 5, 0.6), (2, 5, 0.4)]:
        candidates.append(_generate_moire(f1, f2, ang, size))
    for seed in [1, 7, 42, 123, 999, 555, 777]:
        for thresh in [0.35, 0.45, 0.55, 0.65]:
            candidates.append(_generate_noise_cell(seed, 2, 0.5, 2.0, thresh, size))
    sym_candidates = [_enforce_symmetry(c) for c in candidates]
    best_cell = sym_candidates[0]
    best_score = -np.inf
    for c in sym_candidates:
        if _time_left() < 30:
            break
        score = _evaluate_cell(c, eps_low, eps_high, n_pw=17, n_kpts=8, n_bands=6)
        if score > best_score:
            best_score = score
            best_cell = c
    scored = []
    for c in sym_candidates:
        if _time_left() < 40:
            break
        score = _evaluate_cell(c, eps_low, eps_high, n_pw=17, n_kpts=12, n_bands=6)
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    top_candidates = [c for _, c in scored[:5]]
    for c in top_candidates:
        if _time_left() < 60:
            break
        opt_cell, opt_score = _local_optimize(c, eps_low, eps_high, max_flips=80, n_pw=17, n_kpts=10)
        if opt_score > best_score:
            best_score = opt_score
            best_cell = opt_cell
    best_cell = _enforce_symmetry(best_cell)
    return best_cell
