"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# ---------------------------------------------------------------------------
# Fast TM plane-wave scorer (kept from the current program, vectorised).
# ---------------------------------------------------------------------------
def _build_fourier(inv_eps, size, indices):
    """Fourier coefficients of inv_eps for the needed G vectors."""
    x_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, x_coords, indexing='ij')
    coeffs = {}
    for m in indices:
        for n in indices:
            phase = np.exp(-1j * (2 * np.pi * m * X + 2 * np.pi * n * Y))
            coeffs[(m, n)] = np.sum(inv_eps * phase) / (size * size)
    return coeffs

def _pw_solver(cell, n_pw=13, k_points=None):
    """Fast TM band solver. Returns (max_omega4, min_omega5)."""
    size = cell.shape[0]
    inv_eps = np.where(cell > 0.5, 1.0 / 12.0, 1.0)

    indices = list(range(-n_pw // 2, n_pw // 2 + 1))
    fourier_coeffs = _build_fourier(inv_eps, size, indices)

    if k_points is None:
        k_points = []
        for t in np.linspace(0, 1, 6):
            k_points.append((t * np.pi / 2, t * np.pi / 2))          # Gamma->X
        for t in np.linspace(0, 1, 6):
            k_points.append((np.pi / 2 + t * np.pi / 2,
                             np.pi / 2 - t * np.pi / 2))             # X->M
        for t in np.linspace(0, 1, 6):
            k_points.append((np.pi - t * np.pi, 0.0))                # M->Gamma
        k_points += [(0.3 * np.pi, 0.5 * np.pi),
                     (0.5 * np.pi, 0.7 * np.pi),
                     (0.2 * np.pi, 0.3 * np.pi),
                     (0.4 * np.pi, 0.15 * np.pi)]

    G_list = [(m, n) for m in indices for n in indices]
    N = len(G_list)

    max_omega4 = 0.0
    min_omega5 = np.inf

    for kx, ky in k_points:
        kxG = kx - 2 * np.pi * np.array([m for m, _ in G_list])
        kyG = ky - 2 * np.pi * np.array([n for _, n in G_list])
        A = (kxG ** 2 + kyG ** 2)
        B = np.empty((N, N), dtype=np.complex128)
        for i, (mi, ni) in enumerate(G_list):
            for j, (mj, nj) in enumerate(G_list):
                B[i, j] = fourier_coeffs.get((mi - mj, ni - nj), 0.0)
        try:
            from scipy.linalg import eig
            vals, _ = eig(A, B)
            vals_real = np.real(vals)
            vals_pos = vals_real[vals_real > 1e-10]
            vals_pos.sort()
            if len(vals_pos) >= 5:
                o4 = math.sqrt(vals_pos[3])
                o5 = math.sqrt(vals_pos[4])
                if o4 > max_omega4:
                    max_omega4 = o4
                if o5 < min_omega5:
                    min_omega5 = o5
        except Exception:
            continue

    return max_omega4, min_omega5

def _score_cell(cell):
    try:
        o4, o5 = _pw_solver(cell, n_pw=13)
        if o5 > o4 > 0:
            gap = o5 - o4
            mid = (o5 + o4) / 2.0
            if mid > 0:
                return gap / mid
    except Exception:
        pass
    return -1.0

# ---------------------------------------------------------------------------
# STRUCTURAL CHANGE: Fourier-synthesis representation.
# The dielectric map is a truncated Fourier series restricted to D4-symmetric
# modes (m+n even), so the cell is automatically 4-fold symmetric and the
# optimizer directly tunes the low-order scattering vectors.
# ---------------------------------------------------------------------------
_HARMONICS = [(0, 0), (1, 1), (2, 0), (0, 2), (2, 2), (1, 3), (3, 1), (4, 0), (0, 4)]

def _make_cell_fourier(coeffs, size=32):
    """Build a symmetric 32x32 cell from Fourier coefficients.
    coeffs: list of amplitudes aligned with _HARMONICS."""
    coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    field = np.zeros((size, size), dtype=np.float64)
    for (m, n), a in zip(_HARMONICS, coeffs):
        field += a * (np.cos(2 * np.pi * m * X) + np.cos(2 * np.pi * n * Y))
    cell = (field > 0).astype(np.float64)
    # Enforce exact D4 symmetry about the cell centre.
    c = cell
    c = (c + c[::-1].copy() + c[:, ::-1].copy() + c[::-1].copy()[:, ::-1].copy()
         + c.T + c[::-1].copy().T + c[:, ::-1].copy().T + c[::-1].copy()[:, ::-1].copy().T) / 8.0
    return (c > 0.5).astype(np.float64)

def _generate_initial_candidates(size=32):
    """Diverse Fourier initialisations spanning the structural families
    (central rod, ring, cross, checkerboard-like, multi-rod cluster)."""
    rng = np.random.RandomState(7)
    candidates = []
    # Deterministic structural seeds
    base = {
        'rod':      [1.0, 0.0, -0.6, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
        'ring':     [0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
        'cross':    [0.0, 0.0, 0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        'checker':  [0.0, 0.9, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
        'cluster':  [0.3, 0.4, 0.5, 0.5, 0.2, 0.3, 0.3, 0.0, 0.0],
        'annulus':  [0.0, -0.3, 0.7, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0],
        'grid':     [0.0, 0.0, 0.6, 0.6, 0.4, 0.0, 0.0, 0.5, 0.5],
    }
    candidates.extend(base.values())
    # Random perturbations of the seeds
    for name, c in base.items():
        for _ in range(3):
            candidates.append([v + rng.normal(0, 0.3) for v in c])
    # Fully random
    for _ in range(6):
        candidates.append([rng.normal(0, 0.6) for _ in _HARMONICS])
    return candidates

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200.0  # safety margin before 230s kill

    best_cell = _make_cell_fourier([1.0, 0.0, -0.6, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0], size)
    best_score = _score_cell(best_cell)

    rng = np.random.RandomState(42)

    # Evaluate a diverse set of initial Fourier candidates.
    for coeffs in _generate_initial_candidates(size):
        if time.time() > deadline:
            break
        try:
            cell = _make_cell_fourier(coeffs, size)
            sc = _score_cell(cell)
            if sc > best_score:
                best_score = sc
                best_cell = cell
        except Exception:
            continue

    # Iterative local refinement in Fourier-coefficient space.
    # Track the coefficients that produced the best cell by re-searching.
    def _refine(initial_coeffs, budget):
        cell = _make_cell_fourier(initial_coeffs, size)
        cur_score = _score_cell(cell)
        cur_coeffs = list(initial_coeffs)
        local_deadline = time.time() + budget
        step = 0.25
        for _ in range(400):
            if time.time() > local_deadline:
                break
            idx = rng.randint(len(_HARMONICS))
            new_coeffs = list(cur_coeffs)
            new_coeffs[idx] = cur_coeffs[idx] + rng.normal(0, step)
            new_coeffs[idx] = max(-2.0, min(2.0, new_coeffs[idx]))
            try:
                ncell = _make_cell_fourier(new_coeffs, size)
                nsc = _score_cell(ncell)
            except Exception:
                continue
            if nsc > cur_score:
                cur_coeffs = new_coeffs
                cur_score = nsc
                cell = ncell
                step = max(0.03, step * 0.92)
            else:
                step = min(0.5, step * 1.05)
        return cell, cur_score

    # Refine the best cell and a few strong seeds.
    seeds_to_refine = [
        [1.0, 0.0, -0.6, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.4, 0.5, 0.5, 0.2, 0.3, 0.3, 0.0, 0.0],
    ]
    for seed in seeds_to_refine:
        if time.time() > deadline - 15:
            break
        try:
            cell, sc = _refine(seed, 12.0)
            if sc > best_score:
                best_score = sc
                best_cell = cell
        except Exception:
            continue

    # Refine the current best cell itself (fit a coefficient set by random init).
    if time.time() < deadline - 15:
        try:
            fit_coeffs = [rng.normal(0, 0.5) for _ in _HARMONICS]
            cell, sc = _refine(fit_coeffs, 15.0)
            if sc > best_score:
                best_score = sc
                best_cell = cell
        except Exception:
            pass

    # Final safety: guarantee a valid, symmetric output.
    result = []
    for i in range(size):
        row = [1 if best_cell[i, j] > 0.5 else 0 for j in range(size)]
        result.append(row)
    return result
# EVOLVE-BLOCK-END
