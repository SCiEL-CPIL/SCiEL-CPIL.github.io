import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _make_symmetric_cell(size, base):
    """
    Enforce full square (C4v) symmetry: cell == cell[::-1] row/col and cell == cell.T.
    Majority vote over the 8 orbit positions.
    """
    b = np.array(base, dtype=np.int8)
    N = size
    cell = np.zeros((N, N), dtype=np.int8)
    processed = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if processed[i, j]:
                continue
            orbit = [
                (i, j), (N - 1 - i, j), (j, i), (N - 1 - i, N - 1 - j),
                (i, N - 1 - j), (j, N - 1 - i), (N - 1 - j, i), (N - 1 - j, N - 1 - i),
            ]
            unique_orbit = list(set(orbit))
            vals = [b[r, c] for r, c in unique_orbit]
            val = 1 if sum(vals) >= len(vals) / 2.0 else 0
            for r, c in unique_orbit:
                cell[r, c] = val
                processed[r, c] = True
    return cell.tolist()


def _gs_and_inv_eps(cell, size, n_pw, eps_high, eps_low):
    """Reciprocal vectors and exact Fourier coefficients of 1/eps (via FFT)."""
    G_range = range(-n_pw, n_pw + 1)
    Gs = np.array([(mx, my) for mx in G_range for my in G_range], dtype=float)
    cell_arr = np.array(cell, dtype=float)
    eps_arr = np.where(cell_arr > 0.5, eps_high, eps_low)
    inv_eps_arr = 1.0 / eps_arr
    inv_eps_fft = np.fft.fft2(inv_eps_arr)
    return Gs, inv_eps_fft


def _build_matrix(kx, ky, Gs, inv_eps_fft, size):
    k = np.array([kx, ky])
    kGs = k + Gs
    dot = np.outer(kGs[:, 0], kGs[:, 0]) + np.outer(kGs[:, 1], kGs[:, 1])
    dmx = Gs[:, 0][:, None] - Gs[:, 0][None, :]
    dmy = Gs[:, 1][:, None] - Gs[:, 1][None, :]
    inv_eps_mat = inv_eps_fft[dmx % size, dmy % size]
    A = dot * inv_eps_mat
    return (A + A.conj().T) / 2.0


def _k_points(n_kx=8, n_ky=8):
    pts = [(0.0, 0.0)]
    for t in np.linspace(0.1, 1.0, n_kx):
        pts.append((t * np.pi, 0.0))
    for t in np.linspace(0.1, 1.0, n_ky):
        pts.append((t * np.pi, t * np.pi))
    for t in np.linspace(0.1, 1.0, n_kx):
        pts.append((np.pi, t * np.pi))
    # interior of the irreducible triangle
    for t in np.linspace(0.1, 0.5, 4):
        pts.append((0.5 * np.pi, t * np.pi))
    return pts


def _compute_tm_bands_fast(cell, size, n_pw=8, n_kx=8, n_ky=8, eps_high=12.0, eps_low=1.0):
    """Fast plane-wave TM band gap (surrogate). Returns gap-midgap ratio."""
    Gs, inv_eps_fft = _gs_and_inv_eps(cell, size, n_pw, eps_high, eps_low)
    max_band4 = 0.0
    min_band5 = np.inf
    for kx, ky in _k_points(n_kx, n_ky):
        A = _build_matrix(kx, ky, Gs, inv_eps_fft, size)
        eigvals = np.real(np.linalg.eigvalsh(A))
        eigvals = eigvals[eigvals > 1e-10]
        if len(eigvals) >= 5:
            max_band4 = max(max_band4, eigvals[3])
            min_band5 = min(min_band5, eigvals[4])
    if min_band5 == np.inf or max_band4 == 0:
        return -1.0
    gap = min_band5 - max_band4
    midgap = 0.5 * (min_band5 + max_band4)
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap


def _band_eigenvector(cell, size, n_pw, eps_high, eps_low, band_idx, kx, ky):
    """Return (eigenvector, eigenvalue) of a given band (0-indexed) at (kx,ky)."""
    Gs, inv_eps_fft = _gs_and_inv_eps(cell, size, n_pw, eps_high, eps_low)
    A = _build_matrix(kx, ky, Gs, inv_eps_fft, size)
    w, V = np.linalg.eigh(A)
    w = np.real(w)
    V = np.real(V)
    pos = np.where(w > 1e-10)[0]
    if len(pos) <= band_idx:
        return None, None
    idx = pos[band_idx]
    return V[:, idx], w[idx]


def _field_intensity(evec, Gs, size):
    """|E(r)|^2 for each pixel, from a plane-wave eigenvector (real)."""
    evec = np.real(evec)
    G_range = range(-Gs.shape[0] // 2, Gs.shape[0] // 2 + 1)
    mx = Gs[:, 0].astype(int)
    my = Gs[:, 1].astype(int)
    i = np.arange(size)
    j = np.arange(size)
    I = np.zeros((size, size), dtype=float)
    for a in range(Gs.shape[0]):
        for b in range(Gs.shape[0]):
            coef = evec[a] * evec[b]
            if abs(coef) < 1e-12:
                continue
            phase = np.exp(2j * np.pi * ((mx[a] + mx[b]) * i[:, None] / size
                                         + (my[a] + my[b]) * j[None, :] / size))
            I += coef * phase
    I = np.real(I)
    return np.maximum(I, 0.0)


def _generate_initial_candidates(size):
    """Diverse, symmetric seed designs (dielectric rods and air holes)."""
    candidates = []
    center = size / 2.0 - 0.5

    def from_mask(mask):
        return _make_symmetric_cell(size, mask)

    def circle_mask(val, radius, cx=0.0, cy=0.0):
        m = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i - center) - cx
                y = (j - center) - cy
                if x * x + y * y <= radius * radius:
                    m[i, j] = val
        return m

    def superpose(masks):
        m = np.zeros((size, size), dtype=np.int8)
        for mm in masks:
            m = np.maximum(m, mm)
        return m

    # dielectric rods (1) on air
    for r in [2.0, 3.0, 4.0, 5.0, 6.0]:
        candidates.append(from_mask(circle_mask(1, r)))
    for d in [5.0, 6.0, 7.0, 8.0]:
        for r in [1.5, 2.0, 2.5]:
            m = superpose([circle_mask(1, r, d, 0), circle_mask(1, r, -d, 0),
                           circle_mask(1, r, 0, d), circle_mask(1, r, 0, -d)])
            candidates.append(from_mask(m))
    for d in [4.0, 5.0, 6.0]:
        for r in [1.5, 2.0]:
            m = superpose([circle_mask(1, r, d, d), circle_mask(1, r, -d, -d),
                           circle_mask(1, r, d, -d), circle_mask(1, r, -d, d)])
            candidates.append(from_mask(m))
    # air holes (0) on dielectric background
    for r in [2.0, 3.0, 4.0, 5.0]:
        m = np.ones((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i - center); y = (j - center)
                if x * x + y * y <= r * r:
                    m[i, j] = 0
        candidates.append(from_mask(m))
    return candidates


def design(problem):
    """Time-bounded field-intensity-driven (topological) search for the TM band gap."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)

    start_time = time.time()
    deadline = start_time + 200.0  # safety margin under the 230s kill
    n_pw = 8

    def now():
        return time.time()

    # ---- evaluate initial candidates ----
    candidates = _generate_initial_candidates(size)
    best_cell = None
    best_score = -1.0
    for cell in candidates:
        if now() > deadline:
            break
        try:
            score = _compute_tm_bands_fast(cell, size, n_pw=n_pw, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_cell = cell

    if best_cell is None:
        m = np.zeros((size, size), dtype=np.int8)
        c = size / 2.0 - 0.5
        for i in range(size):
            for j in range(size):
                if (i - c) ** 2 + (j - c) ** 2 <= 2.0 ** 2:
                    m[i, j] = 1
        best_cell = _make_symmetric_cell(size, m)
        return best_cell

    # ---- field-intensity-driven topological optimization ----
    # Sample a few k-points to build a contrast map between band 4 and band 5.
    sample_pts = _k_points(n_kx=5, n_ky=5)
    rng = np.random.default_rng(7)

    def contrast_map(cell):
        """Sum over k of |E4|^2 - |E5|^2 (positive => band4 wants energy, band5 doesn't)."""
        acc = np.zeros((size, size), dtype=float)
        for kx, ky in sample_pts:
            e4, _ = _band_eigenvector(cell, size, n_pw, eps_high, eps_low, 3, kx, ky)
            e5, _ = _band_eigenvector(cell, size, n_pw, eps_high, eps_low, 4, kx, ky)
            if e4 is None or e5 is None:
                continue
            acc += _field_intensity(e4, Gs_cache[0], size) - _field_intensity(e5, Gs_cache[0], size)
        return acc

    Gs_cache = (_gs_and_inv_eps(best_cell, size, n_pw, eps_high, eps_low)[0],)

    cur_cell = best_cell
    cur_score = best_score
    cur_arr = np.array(cur_cell, dtype=np.int8)

    for round_num in range(60):
        if now() > deadline:
            break
        # Build the field-contrast map for the current design
        try:
            C = contrast_map(cur_cell)
        except Exception:
            break
        # Normalize to [-1, 1]
        cmax = np.max(np.abs(C))
        if cmax < 1e-12:
            break
        C = C / cmax

        # Try several "move energy" directions:
        #  direction +1: make high-C pixels dielectric (help band4) and low-C air (help band5)
        #  direction -1: the opposite
        improved = False
        for direction in (1, -1):
            for step in range(12):
                if now() > deadline:
                    break
                # rank pixels by signed contrast
                flat = C.ravel()
                # candidate flips: pixels whose sign matches the direction we want
                # We flip a small batch of the most extreme pixels that are "wrong".
                # Wrong for direction +1: pixel is 1 but C<0 (band5 wants energy) or 0 but C>0.
                cur_flat = cur_arr.ravel()
                want = direction * flat  # positive => should be dielectric
                # pixels currently dielectric but want<0, or air but want>0
                bad = (cur_flat == 1) & (want < -0.2)
                bad = bad | (cur_flat == 0) & (want > 0.2)
                n_bad = int(np.sum(bad))
                if n_bad == 0:
                    break
                # flip a random subset of the worst offenders
                bad_idx = np.where(bad)[0]
                n_flip = max(1, min(4, n_bad))
                # choose the most extreme
                order = np.argsort(np.abs(want[bad_idx]))[::-1][:n_flip]
                flip_idx = bad_idx[order]
                mut = cur_flat.copy()
                mut[flip_idx] = 1 - mut[flip_idx]
                mut_cell = _make_symmetric_cell(size, mut.reshape(size, size))
                try:
                    score = _compute_tm_bands_fast(mut_cell, size, n_pw=n_pw,
                                                   eps_high=eps_high, eps_low=eps_low)
                except Exception:
                    score = -1.0
                if score > cur_score:
                    cur_cell = mut_cell
                    cur_arr = np.array(cur_cell, dtype=np.int8)
                    cur_score = score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_cell = cur_cell
        if not improved:
            # small random restart to escape local minima
            if now() > deadline:
                break
            mut = cur_arr.copy()
            for _ in range(3):
                ii = rng.integers(0, size); jj = rng.integers(0, size)
                mut[ii, jj] = 1 - mut[ii, jj]
            cur_cell = _make_symmetric_cell(size, mut)
            try:
                cur_score = _compute_tm_bands_fast(cur_cell, size, n_pw=n_pw,
                                                   eps_high=eps_high, eps_low=eps_low)
            except Exception:
                cur_score = -1.0
            if cur_score < best_score - 0.05:
                cur_cell = best_cell
                cur_arr = np.array(cur_cell, dtype=np.int8)
                cur_score = best_score

    return best_cell
# EVOLVE-BLOCK-END
