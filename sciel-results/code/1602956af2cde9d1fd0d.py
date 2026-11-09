import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

_SIZE = 32
_EPS_LOW = 1.0
_EPS_HIGH = 12.0
_NPW = 17          # 289 plane waves
_NK = 30           # k-points in irreducible BZ
_FUND = 160        # unique pixels in the C4v fundamental domain (8x16)
_DEADLINE = 210.0  # safety margin before 230s kill


# ----------------------------- geometry / symmetry -----------------------------
def _unique_positions():
    """Return the 160 unique pixel coordinates of the C4v fundamental domain.
    Fundamental domain: 0 <= x < 16, 0 <= y <= 15  (8 rows x 16 cols)."""
    positions = []
    for y in range(16):          # y = 0..15  (top half, including centre row 15)
        for x in range(8):       # x = 0..7   (left quarter)
            positions.append((y, x))
    return positions


def _orbit(y, x):
    """All 8 C4v-symmetric positions of pixel (y, x) in the 32x32 cell."""
    s = _SIZE
    return [
        (y, x),
        (s - 1 - y, x),
        (y, s - 1 - x),
        (s - 1 - y, s - 1 - x),
        (x, y),
        (s - 1 - x, y),
        (x, s - 1 - y),
        (s - 1 - x, s - 1 - y),
    ]


_POSITIONS = _unique_positions()
_ORBITS = [_orbit(y, x) for (y, x) in _POSITIONS]


def _bits_to_cell(bits):
    """Expand a 160-bit fundamental-domain array into the full 32x32 binary cell."""
    cell = np.zeros((_SIZE, _SIZE), dtype=np.int8)
    for i, (y, x) in enumerate(_POSITIONS):
        v = bits[i]
        for oy, ox in _ORBITS[i]:
            cell[oy, ox] = v
    return cell


# ----------------------------- precomputed evaluator data -----------------------------
def _build_k_points():
    """30 k-points over the irreducible BZ boundary/interior (units of 2pi/a)."""
    kpts = []
    # Gamma -> M : (t, 0), t in [0, 0.5]
    for t in np.linspace(0.0, 0.5, 10):
        kpts.append((t, 0.0))
    # M -> X : (0.5, t), t in [0, 0.5]
    for t in np.linspace(0.0, 0.5, 10):
        kpts.append((0.5, t))
    # interior line : (t, t), t in (0, 0.5)
    for t in np.linspace(0.05, 0.45, 10):
        kpts.append((t, t))
    return np.array(kpts, dtype=float)


_KPTS = _build_k_points()
_NK = len(_KPTS)

# G-vector grid: m, n in [-8, 8]  ->  17x17 = 289
_G_IDX = np.arange(-8, 9, dtype=float)
_GX, _GY = np.meshgrid(_G_IDX, _G_IDX, indexing='ij')
_GX = _GX.ravel()
_GY = _GY.ravel()
_NPW = len(_GX)

# (k+G)^2 per k-point and per G
_KPG2 = np.empty((_NK, _NPW), dtype=float)
for ik in range(_NK):
    kx, ky = _KPTS[ik]
    _KPG2[ik] = (kx + _GX) ** 2 + (ky + _GY) ** 2

# G_i - G_j difference grid (for exact Fourier coefficients of 1/eps)
_GDIFF_X = (_GX[:, None] - _GX[None, :]).astype(int)   # (NPW, NPW)
_GDIFF_Y = (_GY[:, None] - _GY[None, :]).astype(int)
# index into the 32x32 FFT array (mod 32)
_GDIFF_IDX = np.mod(_GDIFF_X, _SIZE) * _SIZE + np.mod(_GDIFF_Y, _SIZE)  # (NPW, NPW)


# ----------------------------- fast exact-coefficient TM evaluator -----------------------------
def _score_bits(bits):
    """Compute the gap-midgap ratio of the cell defined by `bits` (160-bit array).
    Uses the exact-coefficient plane-wave method with 289 PWs and 30 k-points,
    matching the evaluator's TM operator:  A = diag((k+G)^2) * (1/eps)_FFT."""
    cell = _bits_to_cell(bits)
    eps = np.where(cell == 1, _EPS_HIGH, _EPS_LOW)
    inv_eps = 1.0 / eps

    # Exact Fourier coefficients of 1/eps (periodic, 32x32 pixel map)
    fft_inv = np.fft.fft2(inv_eps)
    fft_inv /= (_SIZE * _SIZE)
    fft_flat = fft_inv.ravel()                      # (1024,)

    # Build the 289x289 coefficient matrix C[i,j] = (1/eps)_{G_i - G_j}
    C = fft_flat[_GDIFF_IDX]                        # (NPW, NPW) complex

    min_w5 = np.inf
    max_w4 = -np.inf

    for ik in range(_NK):
        # A = diag(kpg2) * C   (TM Galerkin matrix)
        A = _KPG2[ik, :, None] * C                  # (NPW, NPW)
        # A is Hermitian (C is Hermitian for real eps), use eigh
        vals = np.linalg.eigvalsh(A)
        vals = np.sort(np.real(vals))
        if vals[3] <= 0 or vals[4] <= 0:
            return -1.0
        w4 = np.sqrt(vals[3])
        w5 = np.sqrt(vals[4])
        if w4 > max_w4:
            max_w4 = w4
        if w5 < min_w5:
            min_w5 = w5

    if min_w5 == np.inf or max_w4 == -np.inf:
        return -1.0
    gap = min_w5 - max_w4
    mid = 0.5 * (min_w5 + max_w4)
    if mid <= 0:
        return -1.0
    return 2.0 * gap / mid


# ----------------------------- parallel workers -----------------------------
def _worker(args):
    """Evaluate a list of (bit-index, flipped) candidates in this process."""
    bits, idxs = args
    results = []
    for i in idxs:
        b = bits.copy()
        b[i] = 1 - b[i]
        s = _score_bits(b)
        results.append((i, s))
    return results


# ----------------------------- design -----------------------------
def design(problem):
    start = time.time()
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    global _EPS_LOW, _EPS_HIGH
    _EPS_LOW = eps_low
    _EPS_HIGH = eps_high

    def _elapsed():
        return time.time() - start

    # ---- initial candidates (all C4v-symmetric) ----
    def _rod(r):
        bits = np.zeros(_FUND, dtype=np.int8)
        for i, (y, x) in enumerate(_POSITIONS):
            # map fundamental (y in 0..15, x in 0..7) to full coords for distance
            fy = y if y <= 15 else y          # already top half
            fx = x
            # distance from cell centre (16,16) in full coords
            dy = fy - 16.0
            dx = fx - 16.0
            if dx * dx + dy * dy < r * r:
                bits[i] = 1
        return bits

    def _random_bits(seed):
        rng = np.random.RandomState(seed)
        return rng.randint(0, 2, _FUND).astype(np.int8)

    candidates = []
    # rod-like initial guesses (small and large)
    for r in [4.0, 6.0, 8.0, 10.0]:
        candidates.append(_rod(r))
    # a few random restarts
    for s in range(4):
        candidates.append(_random_bits(1000 + s))

    # score them (sequential, cheap enough)
    best_bits = None
    best_score = -np.inf
    for c in candidates:
        s = _score_bits(c)
        if s > best_score:
            best_score = s
            best_bits = c.copy()

    if best_bits is None:
        best_bits = _rod(6.0)
        best_score = _score_bits(best_bits)

    # ---- parallel coordinate-descent hill-climbing ----
    n_workers = max(1, min(8, np.cpu_count() or 1))
    idx_list = np.arange(_FUND)

    improved = True
    while improved and _elapsed() < _DEADLINE - 5.0:
        improved = False
        # shuffle candidate order for diversity
        rng = np.random.RandomState(int(_elapsed() * 1000) % (2 ** 31))
        perm = rng.permutation(_FUND)

        # chunk the 160 indices across workers
        chunks = [perm[k::n_workers] for k in range(n_workers)]

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_worker, (best_bits, ch)) for ch in chunks]
                all_results = []
                for f in futures:
                    all_results.extend(f.result())
        except Exception:
            # fallback: sequential
            all_results = []
            for i in perm:
                b = best_bits.copy()
                b[i] = 1 - b[i]
                all_results.append((i, _score_bits(b)))

        # apply the single best improving flip (coordinate descent)
        best_flip_idx = None
        best_flip_score = best_score
        for i, s in all_results:
            if s > best_flip_score:
                best_flip_score = s
                best_flip_idx = i

        if best_flip_idx is not None:
            best_bits[best_flip_idx] = 1 - best_bits[best_flip_idx]
            best_score = best_flip_score
            improved = True

    # ---- final binarized, strictly symmetric cell ----
    cell = _bits_to_cell(best_bits)
    out = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(cell[i, j]))
        out.append(row)
    return out
