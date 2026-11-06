"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap."""
import time
import math
import random
import numpy as np

_DEADLINE = 218.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _enforce_d4_strict(cell):
    """
    Enforce D4 symmetry by mirroring the top-left quadrant.
    The top-left 16x16 quadrant is the fundamental domain.
    Mirroring:
      - row i -> row 31-i  (vertical flip)
      - col j -> col 31-j  (horizontal flip)
      - (i,j) -> (j,i)     (transpose)
    We take the top-left quadrant and mirror it to fill the whole cell.
    """
    size = 32
    half = size // 2
    # The fundamental domain is the top-left half x half = 16x16
    # Actually for D4 of a 32x32 grid centered at (15.5, 15.5),
    # the independent region is i in [0,15], j in [0,15] (top-left quadrant)
    # Mirror rules:
    #   cell[i][j] = cell[31-i][j] = cell[i][31-j] = cell[31-i][31-j]
    #   cell[i][j] = cell[j][i] = cell[31-j][i] = cell[j][31-i] = cell[31-j][31-i]
    # So the independent region is i<=j, i<=15, j<=15 (upper-left triangle of top-left quadrant)
    # But simpler: just define by top-left 16x16 and mirror.
    # Actually the cleanest: define values for i in [0,15], j in [0,15] and mirror.
    # But (i,j) and (j,i) must agree, so we need i<=j within that quadrant.
    # Let's just use the top-left quadrant (16x16) and enforce all mirrors.
    
    result = [[0] * size for _ in range(size)]
    for i in range(half):
        for j in range(half):
            val = cell[i][j]
            # The 8 symmetric positions
            positions = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i),
            ]
            for pi, pj in positions:
                result[pi][pj] = val
    return result


def _compute_tm_gap(cell, N_pw=5, n_kpts=12):
    """
    TM band gap computation using plane-wave method.
    Returns the gap-midgap ratio for bands 4 and 5.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0

    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)

    fft_coeffs = np.fft.fft2(inv_eps)
    N = size

    def get_coeff(m, n):
        idx_m = m % N
        idx_n = n % N
        return fft_coeffs[idx_m, idx_n] / (N * N)

    g_indices = []
    for m in range(-N_pw + 1, N_pw):
        for n in range(-N_pw + 1, N_pw):
            g_indices.append((m, n))

    npw = len(g_indices)

    # Build S matrix (shared inverse epsilon)
    S = np.zeros((npw, npw), dtype=float)
    for i_idx, (mi, ni) in enumerate(g_indices):
        for j_idx, (mj, nj) in enumerate(g_indices):
            gm = mi - mj
            gn = ni - nj
            S[i_idx, j_idx] = get_coeff(gm, gn).real

    # k-points in the irreducible BZ: triangle with vertices (0,0), (1,0), (0.5,0.5)
    # Actually the irreducible BZ for square lattice is the triangle:
    # (0,0) -> (0.5, 0) -> (0.5, 0.5) -> (0,0)  ... wait, let me think.
    # The BZ of a square lattice with period 1 is [-0.5, 0.5] x [-0.5, 0.5].
    # The irreducible BZ (1/8 of the BZ due to D4 symmetry) is the triangle:
    # (0,0), (0.5, 0), (0.5, 0.5)
    # Wait, that's not right either. The D4 symmetry group of the square has 8 elements.
    # The BZ is [-0.5,0.5]^2. The irreducible wedge is 0 <= ky <= kx, kx <= 0.5, ky >= 0.
    # So vertices: (0,0), (0.5, 0), (0.5, 0.5).
    
    kpts = []
    # Along Gamma-X: (0,0) to (0.5, 0)
    for i in range(n_kpts // 4 + 1):
        kx = i / (n_kpts // 4) * 0.5
        kpts.append((kx, 0.0))
    # Along X-M: (0.5, 0) to (0.5, 0.5)
    for i in range(1, n_kpts // 4 + 1):
        ky = i / (n_kpts // 4) * 0.5
        kpts.append((0.5, ky))
    # Along M-Gamma: (0.5, 0.5) to (0, 0)
    for i in range(1, n_kpts // 4 + 1):
        t = i / (n_kpts // 4)
        kpts.append((0.5 * (1 - t), 0.5 * (1 - t)))
    # Interior points
    kpts.append((0.3, 0.1))
    kpts.append((0.4, 0.2))
    kpts.append((0.35, 0.15))
    kpts.append((0.45, 0.25))
    kpts.append((0.2, 0.05))
    kpts.append((0.3, 0.15))

    # Remove duplicates
    unique_kpts = []
    seen = set()
    for kx, ky in kpts:
        key = (round(kx, 6), round(ky, 6))
        if key not in seen:
            seen.add(key)
            unique_kpts.append((kx, ky))
    kpts = unique_kpts[:n_kpts + 6]

    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx, ky in kpts:
        km2 = np.zeros(npw)
        for idx, (m, n) in enumerate(g_indices):
            km2[idx] = (2.0 * m + kx) ** 2 + (2.0 * n + ky) ** 2

        A = S * km2[np.newaxis, :]

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

                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue

    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap <= 0:
        return 0.0

    return 2.0 * gap / midgap


def _make_cell_disk(r):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < r * r:
                cell[i][j] = 1
    return _enforce_d4_strict(cell)


def _make_cell_ring(r_in, r_out):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            if r_in <= r < r_out:
                cell[i][j] = 1
    return _enforce_d4_strict(cell)


def _make_cell_cross(bar_w, cross_l):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < bar_w / 2.0 and abs(y) < cross_l / 2.0:
                cell[i][j] = 1
            if abs(y) < bar_w / 2.0 and abs(x) < cross_l / 2.0:
                cell[i][j] = 1
    return _enforce_d4_strict(cell)


def _make_cell_hole_in_dielectric(hole_r):
    """Dielectric background with a circular air hole."""
    size = 32
    cell = [[1] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < hole_r * hole_r:
                cell[i][j] = 0
    return _enforce_d4_strict(cell)


def _make_cell_dielectric_holes(n_holes, hole_r, spacing):
    """Dielectric with periodic air holes in a sub-lattice."""
    size = 32
    cell = [[1] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Place holes at regular intervals
            for dx in np.arange(-0.5, 0.5 + 1e-9, spacing):
                for dy in np.arange(-0.5, 0.5 + 1e-9, spacing):
                    if (x - dx) ** 2 + (y - dy) ** 2 < hole_r ** 2:
                        cell[i][j] = 0
    return _enforce_d4_strict(cell)


def _make_cell_square_hole(half_size):
    """Dielectric with a square air hole in the center."""
    size = 32
    cell = [[1] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < half_size and abs(y) < half_size:
                cell[i][j] = 0
    return _enforce_d4_strict(cell)


def _make_cell_air_square(sq_size):
    """Air background with a square dielectric in the center."""
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < sq_size / 2.0 and abs(y) < sq_size / 2.0:
                cell[i][j] = 1
    return _enforce_d4_strict(cell)


def _local_search(cell, max_iter=5000, temperature=0.05):
    """
    Local search with simulated annealing over symmetric pixel groups.
    """
    size = 32
    best_cell = [row[:] for row in cell]
    best_score = _compute_tm_gap(best_cell)
    current_cell = [row[:] for row in cell]
    current_score = best_score
    
    # Precompute the independent pixel positions (top-left quadrant)
    independent = []
    for i in range(16):
        for j in range(16):
            independent.append((i, j))
    
    random.shuffle(independent)
    
    t = temperature
    cooling = 0.995
    
    for it in range(max_iter):
        if _remaining() < 15:
            break
        
        # Pick a random independent pixel to flip
        idx = random.randint(0, len(independent) - 1)
        i, j = independent[idx]
        
        # Flip this pixel and all its symmetric partners
        new_cell = [row[:] for row in current_cell]
        positions = [
            (i, j),
            (size - 1 - i, j),
            (i, size - 1 - j),
            (size - 1 - i, size - 1 - j),
            (j, i),
            (size - 1 - j, i),
            (j, size - 1 - i),
            (size - 1 - j, size - 1 - i),
        ]
        for pi, pj in positions:
            new_cell[pi][pj] = 1 - current_cell[pi][pj]
        
        new_score = _compute_tm_gap(new_cell)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / t):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
        
        t *= cooling
    
    return best_cell, best_score


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
    
    # Generate diverse initial candidates
    candidates = []
    
    # Central disks (air holes in dielectric)
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append(_make_cell_hole_in_dielectric(r))
    
    # Dielectric disks in air
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_cell_disk(r))
    
    # Rings
    for r_in, r_out in [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3), (0.25, 0.35), (0.3, 0.4), (0.15, 0.35)]:
        candidates.append(_make_cell_ring(r_in, r_out))
    
    # Cross shapes
    for bar_w, cross_l in [(0.1, 0.5), (0.15, 0.6), (0.2, 0.7), (0.1, 0.8), (0.2, 0.9), (0.15, 0.8)]:
        candidates.append(_make_cell_cross(bar_w, cross_l))
    
    # Square holes in dielectric
    for hs in [0.05, 0.1, 0.15, 0.2, 0.25]:
        candidates.append(_make_cell_square_hole(hs))
    
    # Dielectric squares in air
    for ss in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_cell_air_square(ss))
    
    # Dielectric with multiple holes
    for hole_r, spacing in [(0.05, 0.2), (0.05, 0.25), (0.07, 0.25), (0.05, 0.3), (0.08, 0.3)]:
        candidates.append(_make_cell_dielectric_holes(4, hole_r, spacing))
    
    # Score all candidates
    scores = []
    for cell in candidates:
        if _remaining() < 40:
            break
        score = _compute_tm_gap(cell)
        scores.append(score)
    
    scores = np.array(scores)
    
    # Find best initial candidate
    best_idx = np.argmax(scores)
    best_cell = candidates[best_idx]
    best_score = scores[best_idx]
    
    # Run local search from top candidates
    top_k = min(5, len(candidates))
    top_indices = np.argsort(scores)[-top_k:]
    
    for idx in top_indices:
        if _remaining() < 30:
            break
        cell = candidates[idx]
        # Run local search with different temperatures
        for temp in [0.03, 0.05, 0.08]:
            if _remaining() < 20:
                break
            ls_cell, ls_score = _local_search(cell, max_iter=2000, temperature=temp)
            if ls_score > best_score:
                best_score = ls_score
                best_cell = ls_cell
    
    # Final refinement from best
    if _remaining() > 25:
        ls_cell, ls_score = _local_search(best_cell, max_iter=3000, temperature=0.02)
        if ls_score > best_score:
            best_score = ls_score
            best_cell = ls_cell
    
    # Fallback
    if best_cell is None:
        best_cell = _make_cell_hole_in_dielectric(0.2)
    
    return best_cell
