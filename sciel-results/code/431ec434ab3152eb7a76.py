"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap."""
import time
import math
import numpy as np
from itertools import product

# Deadline guard: evaluator kills at 230s wall-clock; keep margin
_DEADLINE = 210.0  # seconds
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _enforce_d4(cell):
    """Enforce full D4 symmetry by mirroring the top-left quadrant."""
    size = len(cell)
    half = size // 2
    result = [[0] * size for _ in range(size)]
    # Define only the top-left quadrant (i < half, j < half)
    # Mirror to all 4 quadrants
    for i in range(half):
        for j in range(half):
            v = cell[i][j]
            result[i][j] = v
            result[i][size - 1 - j] = v
            result[size - 1 - i][j] = v
            result[size - 1 - i][size - 1 - j] = v
    return result


def _compute_fourier_coeffs(cell, N=17):
    """Compute Fourier coefficients of 1/epsilon for the cell using FFT."""
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0

    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)

    # Use FFT to compute Fourier coefficients efficiently
    # The pixel grid is size x size, representing one unit cell
    # Fourier coeff c_{mn} = (1/size^2) * sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(m*i/size + n*j/size))
    # This is just the 2D DFT normalized by size^2
    fft_2d = np.fft.fft2(inv_eps)
    # fft_2d[m, n] gives sum over i,j of inv_eps[i,j] * exp(-2*pi*i*(m*i + n*j)/size)
    # So c_{mn} = fft_2d[m, n] / size^2
    # But we need indices in range [-N+1, N-1]
    coeffs = {}
    for m in range(-N + 1, N):
        for n in range(-N + 1, N):
            # Map to FFT index: m mod size
            mi = m % size
            ni = n % size
            coeffs[(m, n)] = fft_2d[mi, ni] / (size * size)

    return coeffs


def _solve_tm_bands(cell, N_pw=17, kpts=None):
    """
    Solve TM bands at given k-points.
    Returns list of (omega4, omega5) pairs for each k-point.
    """
    size = len(cell)
    coeffs = _compute_fourier_coeffs(cell, N=N_pw)

    if kpts is None:
        # Generate k-points in irreducible BZ
        kpts = []
        # Along X: (kx, 0)
        for i in range(1, 9):
            kx = i * np.pi / 8.0
            kpts.append((kx, 0.0))
        # Along M-X: (pi, ky) with ky from 0 to pi
        for i in range(1, 9):
            ky = i * np.pi / 8.0
            kpts.append((np.pi, ky))
        # Along M-G: (kx, kx)
        for i in range(1, 9):
            kx = i * np.pi / 8.0
            kpts.append((kx, kx))
        # Interior points
        kpts.append((np.pi / 2, np.pi / 4))
        kpts.append((3 * np.pi / 4, np.pi / 4))
        kpts.append((np.pi / 2, np.pi / 2))

    g_vectors = []
    for m in range(-N_pw + 1, N_pw):
        for n in range(-N_pw + 1, N_pw):
            g_vectors.append((2 * np.pi * m, 2 * np.pi * n, m, n))

    npw = len(g_vectors)
    results = []

    for kx, ky in kpts:
        A = np.zeros((npw, npw), dtype=complex)
        for mi in range(npw):
            gx_m, gy_m, mm, mn = g_vectors[mi]
            kx_m = kx + gx_m
            ky_m = ky + gy_m
            km2 = kx_m * kx_m + ky_m * ky_m

            for ni in range(npw):
                gx_n, gy_n, nn, nnn = g_vectors[ni]
                dm = mm - nn
                dn = mn - nnn
                c = coeffs.get((dm, dn), 0.0)
                A[mi, ni] = c * km2

        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)
            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]
                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)
                    results.append((omega4, omega5))
        except:
            continue

    return results


def _compute_gap(cell, N_pw=17, kpts=None):
    """Compute the gap-midgap ratio between bands 4 and 5."""
    results = _solve_tm_bands(cell, N_pw=N_pw, kpts=kpts)
    if not results:
        return 0.0

    max_omega4 = max(r[0] for r in results)
    min_omega5 = min(r[1] for r in results)

    if min_omega5 <= 0 or max_omega4 <= 0:
        return 0.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap <= 0:
        return 0.0

    return 2.0 * gap / midgap


def _spectral_init(size=32, eps_high=12.0):
    """
    Spectral-domain initialization:
    Start with a reference structure (central disk), compute band-edge modes,
    extract energy density, and threshold to get binary pattern.
    """
    # Start with a reference: central disk
    ref_cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 * 0.2:
                ref_cell[i][j] = 1
    ref_cell = _enforce_d4(ref_cell)

    # Compute bands at Gamma point (k=0) for the reference
    kpts_gamma = [(0.0, 0.0)]
    results = _solve_tm_bands(ref_cell, N_pw=13, kpts=kpts_gamma)

    if not results:
        # Fallback: just use the reference
        return ref_cell

    # For now, use a different strategy: initialize with a pattern that has
    # high dielectric contrast in regions where the band-edge field energy is high.
    # Since computing eigenvectors is expensive, use a heuristic:
    # Start with a pattern that has 4-fold symmetric features at multiple scales.

    # Try several initial patterns and pick the best
    candidates = []

    # Pattern 1: Central disk + 4 corner dots
    for r_center in [0.15, 0.2, 0.25]:
        for r_corner in [0.05, 0.08, 0.1]:
            cell = [[0] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    # Central disk
                    if x * x + y * y < r_center * r_center:
                        cell[i][j] = 1
                    # 4 corner dots
                    for cx, cy in [(0.35, 0.35), (-0.35, 0.35), (0.35, -0.35), (-0.35, -0.35)]:
                        dx = x - cx
                        dy = y - cy
                        if dx * dx + dy * dy < r_corner * r_corner:
                            cell[i][j] = 1
            cell = _enforce_d4(cell)
            candidates.append(cell)

    # Pattern 2: Ring + cross
    for r_in, r_out in [(0.15, 0.3), (0.2, 0.35), (0.1, 0.25)]:
        for bar_w in [0.05, 0.1, 0.15]:
            cell = [[0] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    r = math.sqrt(x * x + y * y)
                    if r_in <= r < r_out:
                        cell[i][j] = 1
                    if abs(x) < bar_w / 2.0 and abs(y) < 0.4:
                        cell[i][j] = 1
                    if abs(y) < bar_w / 2.0 and abs(x) < 0.4:
                        cell[i][j] = 1
            cell = _enforce_d4(cell)
            candidates.append(cell)

    # Pattern 3: 4 rods along axes
    for rod_w in [0.05, 0.08, 0.12]:
        for rod_len in [0.3, 0.4, 0.5]:
            cell = [[0] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    if abs(x) < rod_w / 2.0 and abs(y) < rod_len / 2.0:
                        cell[i][j] = 1
                    if abs(y) < rod_w / 2.0 and abs(x) < rod_len / 2.0:
                        cell[i][j] = 1
            cell = _enforce_d4(cell)
            candidates.append(cell)

    # Pattern 4: Diamond / rotated square
    for r_diamond in [0.15, 0.2, 0.25]:
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Diamond: |x| + |y| < r
                if abs(x) + abs(y) < r_diamond:
                    cell[i][j] = 1
        cell = _enforce_d4(cell)
        candidates.append(cell)

    # Pattern 5: Checkerboard-like with larger blocks
    for block in [2, 3, 4]:
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                bi = i // block
                bj = j // block
                if (bi + bj) % 2 == 0:
                    cell[i][j] = 1
        cell = _enforce_d4(cell)
        candidates.append(cell)

    # Pattern 6: Concentric rings (multiple)
    for n_rings in [2, 3]:
        for base_r in [0.1, 0.15, 0.2]:
            for ring_width in [0.05, 0.08, 0.1]:
                cell = [[0] * size for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        r = math.sqrt(x * x + y * y)
                        for k in range(n_rings):
                            r_inner = base_r + k * ring_width
                            r_outer = r_inner + ring_width * 0.8
                            if r_inner <= r < r_outer:
                                cell[i][j] = 1
                cell = _enforce_d4(cell)
                candidates.append(cell)

    return candidates


def _local_binary_search(cell, size=32, n_iterations=500, N_pw=13, n_kpts=10, seed=42):
    """
    Local binary optimization: flip pixels that improve the gap.
    Only consider pixels in the top-left quadrant for D4 symmetry.
    """
    rng = np.random.RandomState(seed)
    half = size // 2

    # Get initial gap
    kpts = [(0.0, 0.0), (np.pi / 2, 0.0), (np.pi, 0.0), (np.pi / 2, np.pi / 2),
            (np.pi, np.pi / 2), (np.pi / 3, 0.0), (2 * np.pi / 3, 0.0),
            (np.pi / 4, np.pi / 4), (np.pi / 2, np.pi / 4), (3 * np.pi / 4, np.pi / 4)]

    current_gap = _compute_gap(cell, N_pw=N_pw, kpts=kpts)
    best_cell = [row[:] for row in cell]
    best_gap = current_gap

    # Get all positions in the top-left quadrant
    positions = [(i, j) for i in range(half) for j in range(half)]

    for it in range(n_iterations):
        if _remaining() < 30:
            break

        # Pick a random position to flip
        idx = rng.randint(len(positions))
        i, j = positions[idx]

        # Try flipping
        old_val = cell[i][j]
        cell[i][j] = 1 - old_val
        cell = _enforce_d4(cell)

        new_gap = _compute_gap(cell, N_pw=N_pw, kpts=kpts)

        if new_gap > best_gap:
            best_gap = new_gap
            best_cell = [row[:] for row in cell]
        else:
            # Revert
            cell[i][j] = old_val
            cell = _enforce_d4(cell)

    return best_cell, best_gap


def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell.
    Returns a nested list of 0/1 pixels.
    """
    global _start_time
    _start_time = time.time()

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")

    # Phase 1: Generate candidate initial patterns
    candidates = _spectral_init(size=size, eps_high=eps_high)

    # Phase 2: Evaluate each candidate with a quick band calculation
    best_cell = None
    best_score = -float('inf')

    # Quick k-points for screening
    quick_kpts = [(0.0, 0.0), (np.pi / 2, 0.0), (np.pi, 0.0), (np.pi / 2, np.pi / 2),
                  (np.pi / 4, np.pi / 4), (np.pi / 3, 0.0)]

    for idx, cell in enumerate(candidates):
        if _remaining() < 60:
            break
        try:
            score = _compute_gap(cell, N_pw=13, kpts=quick_kpts)
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
        except:
            continue

    if best_cell is None:
        # Fallback: central disk
        best_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.2 * 0.2:
                    best_cell[i][j] = 1
        best_cell = _enforce_d4(best_cell)

    # Phase 3: Local binary search to improve the best candidate
    if _remaining() > 60:
        n_iter = min(500, int(_remaining() / 0.5))  # ~0.5s per iteration estimate
        best_cell, best_score = _local_binary_search(
            best_cell, size=size, n_iterations=n_iter, N_pw=13, n_kpts=6
        )

    # Phase 4: If time remains, do a second round of local search with more k-points
    if _remaining() > 40:
        # Refine with more k-points
        refined_kpts = [(0.0, 0.0), (np.pi / 2, 0.0), (np.pi, 0.0), (np.pi / 2, np.pi / 2),
                        (np.pi, np.pi / 2), (np.pi / 3, 0.0), (2 * np.pi / 3, 0.0),
                        (np.pi / 4, np.pi / 4), (np.pi / 2, np.pi / 4), (3 * np.pi / 4, np.pi / 4)]
        final_score = _compute_gap(best_cell, N_pw=15, kpts=refined_kpts)

        if _remaining() > 30:
            # Another round of local search with refined k-points
            n_iter2 = min(200, int(_remaining() / 0.5))
            best_cell, final_score = _local_binary_search(
                best_cell, size=size, n_iterations=n_iter2, N_pw=15, n_kpts=10, seed=123
            )

    return best_cell
