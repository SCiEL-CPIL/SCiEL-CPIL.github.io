"""Combined: direct dielectric-rod geometry (inspiration 1) + fast, symmetry-aware local search (inspiration 2)."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Precomputation for a fast, self-consistent plane-wave TM evaluator.
# ---------------------------------------------------------------------------
SIZE = 32
EPS_HIGH = 12.0
EPS_LOW = 1.0

# Plane-wave grid: 17x17 with |G| <= 8.5
_HALF = 8
_G_LIST = []
for _gx in range(-_HALF, _HALF + 1):
    for _gy in range(-_HALF, _HALF + 1):
        if _gx * _gx + _gy * _gy <= 8.5 ** 2:
            _G_LIST.append((_gx, _gy))
_G_LIST = np.array(_G_LIST, dtype=int)
_NPW = len(_G_LIST)
_GX = _G_LIST[:, 0]
_GY = _G_LIST[:, 1]

# Fourier-coefficient index for each pair difference (G_i - G_j)
_coeff_idx = {}
for _i, (_gx, _gy) in enumerate(_G_LIST):
    _coeff_idx[(_gx, _gy)] = _i

_CIDX = np.full((_NPW, _NPW), -1, dtype=int)
for _i in range(_NPW):
    for _j in range(_NPW):
        _k = (_GX[_i] - _GX[_j], _GY[_i] - _GY[_j])
        if _k in _coeff_idx:
            _CIDX[_i, _j] = _coeff_idx[_k]
_VALID = _CIDX >= 0
_VALID_IDX = _CIDX[_VALID]

# 30 k-points on the boundary of the irreducible BZ (units of 1/a)
_kpts = []
for _i in range(11):
    _kpts.append((_i * 0.5 / 10.0, 0.0))
for _i in range(1, 11):
    _kpts.append((0.5, _i * 0.5 / 10.0))
for _i in range(1, 10):
    _t = _i / 10.0
    _kpts.append((0.5 * (1 - _t), 0.5 * (1 - _t)))
KPTS = np.array(_kpts[:30], dtype=float)

# Precompute (k + G)^2 for each k-point and each G
KG_SQ = np.zeros((len(KPTS), _NPW))
for _ki in range(len(KPTS)):
    _kx, _ky = KPTS[_ki]
    KG_SQ[_ki] = (_kx + _GX) ** 2 + (_ky + _GY) ** 2

# Symmetry orbits under C4v about the cell centre
_orbits = []
_visited = np.zeros((SIZE, SIZE), dtype=bool)
for _i in range(SIZE):
    for _j in range(SIZE):
        if not _visited[_i, _j]:
            _orb = [(_i, _j),
                    (_j, _i),
                    (SIZE - 1 - _i, SIZE - 1 - _j),
                    (SIZE - 1 - _j, SIZE - 1 - _i)]
            _uniq = []
            for _p in _orb:
                if not _visited[_p[0], _p[1]]:
                    _uniq.append(_p)
                    _visited[_p[0], _p[1]] = True
            _orbits.append(_uniq)

# ---------------------------------------------------------------------------
# Fast evaluator: takes a 32x32 0/1 array (1 = dielectric) and returns the
# gap-midgap ratio between TM bands 4 and 5.
# ---------------------------------------------------------------------------
def _score(cell_arr):
    """cell_arr: (32,32) int array, 1=dielectric, 0=air. Returns gap-midgap."""
    eps = np.where(cell_arr == 1, EPS_HIGH, EPS_LOW)
    mu = 1.0 / eps
    mu_fft = np.fft.fft2(mu) / (SIZE * SIZE)

    max_w4 = 0.0
    min_w5 = float('inf')

    for _ki in range(len(KPTS)):
        # Build the 1/eps Fourier matrix for this k
        c = np.empty(_NPW, dtype=float)
        for _i, (_gx, _gy) in enumerate(_G_LIST):
            c[_i] = mu_fft[_gy % SIZE, _gx % SIZE].real
        C = np.empty((_NPW, _NPW), dtype=float)
        C[_VALID] = c[_VALID_IDX]

        # M = C * (k+G_i)(k+G_j)  then symmetrize; eigenvalues = omega^2
        kg = KG_SQ[_ki]
        M = C * kg
        M = (M + M.T) * 0.5
        ev = np.linalg.eigvalsh(M)
        w4 = math.sqrt(max(0.0, ev[3]))
        w5 = math.sqrt(max(0.0, ev[4]))
        if w4 > max_w4:
            max_w4 = w4
        if w5 < min_w5:
            min_w5 = w5

    if min_w5 <= 0.0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)


# ---------------------------------------------------------------------------
# Geometry generators (dielectric background with air holes)
# ---------------------------------------------------------------------------
def _make_holes(radius, n_holes=4, hole_r=0.28):
    """Dielectric background with n_holes air holes at (±0.25, ±0.25)."""
    cell = np.ones((SIZE, SIZE), dtype=int)
    centers = []
    if n_holes == 4:
        centers = [(0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]
    elif n_holes == 2:
        centers = [(0.25, 0.0), (-0.25, 0.0)]
    elif n_holes == 1:
        centers = [(0.0, 0.0)]

    for _i in range(SIZE):
        for _j in range(SIZE):
            x = (_i + 0.5) / SIZE - 0.5
            y = (_j + 0.5) / SIZE - 0.5
            for cx, cy in centers:
                if (x - cx) ** 2 + (y - cy) ** 2 < hole_r ** 2:
                    cell[_i, _j] = 0
                    break
    return cell


def _make_ring(inner_r, outer_r):
    """Dielectric ring: 1 where inner_r < r < outer_r, else 0."""
    cell = np.zeros((SIZE, SIZE), dtype=int)
    for _i in range(SIZE):
        for _j in range(SIZE):
            x = (_i + 0.5) / SIZE - 0.5
            y = (_j + 0.5) / SIZE - 0.5
            r = math.sqrt(x * x + y * y)
            if inner_r < r < outer_r:
                cell[_i, _j] = 1
    return cell


def _make_cross(width):
    """Dielectric cross (plus sign) of given half-width."""
    cell = np.zeros((SIZE, SIZE), dtype=int)
    c = SIZE // 2
    for _i in range(SIZE):
        for _j in range(SIZE):
            if abs(_i - c + 0.5) < width or abs(_j - c + 0.5) < width:
                cell[_i, _j] = 1
    return cell


def _make_checkerboard(block):
    cell = np.zeros((SIZE, SIZE), dtype=int)
    for _i in range(SIZE):
        for _j in range(SIZE):
            if (_i // block + _j // block) % 2 == 0:
                cell[_i, _j] = 1
    return cell


def _make_square(half_w, inverted=False):
    cell = np.ones((SIZE, SIZE), dtype=int) if not inverted else np.zeros((SIZE, SIZE), dtype=int)
    for _i in range(SIZE):
        for _j in range(SIZE):
            x = (_i + 0.5) / SIZE - 0.5
            y = (_j + 0.5) / SIZE - 0.5
            if abs(x) < half_w and abs(y) < half_w:
                cell[_i, _j] = 0 if not inverted else 1
    return cell


# ---------------------------------------------------------------------------
# Symmetry enforcement
# ---------------------------------------------------------------------------
def _symmetrize(cell):
    """Enforce C4v symmetry: cell == reversed rows/cols and transpose."""
    out = np.zeros_like(cell)
    for _i in range(SIZE):
        for _j in range(SIZE):
            vals = [cell[_i, _j],
                    cell[_j, _i],
                    cell[SIZE - 1 - _i, SIZE - 1 - _j],
                    cell[SIZE - 1 - _j, SIZE - 1 - _i]]
            out[_i, _j] = 1 if sum(vals) >= 2 else 0
    return out


# ---------------------------------------------------------------------------
# Local search: hill-climbing with random orbit flips
# ---------------------------------------------------------------------------
def _local_search(cell, score, rng, deadline):
    best_cell = cell.copy()
    best_score = score
    stuck = 0

    while time.time() < deadline:
        n_flips = int(rng.integers(1, 4))
        flipped = []
        for _ in range(n_flips):
            oi = int(rng.integers(0, len(_orbits)))
            orb = _orbits[oi]
            flipped.append(orb)
            for (i, j) in orb:
                best_cell[i, j] = 1 - best_cell[i, j]

        s = _score(best_cell)
        if s > best_score:
            best_score = s
            stuck = 0
        else:
            for orb in flipped:
                for (i, j) in orb:
                    best_cell[i, j] = 1 - best_cell[i, j]
            stuck += 1
            if stuck > 500:
                # Perturb: flip a random set of orbits to escape local minimum
                for _ in range(int(rng.integers(3, 8))):
                    oi = int(rng.integers(0, len(_orbits)))
                    for (i, j) in _orbits[oi]:
                        best_cell[i, j] = 1 - best_cell[i, j]
                best_cell = _symmetrize(best_cell)
                best_score = _score(best_cell)
                stuck = 0

    return best_cell, best_score


# ---------------------------------------------------------------------------
# Main design function
# ---------------------------------------------------------------------------
def design(problem):
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    deadline = time.time() + 115.0  # safety margin before 120s CPU budget

    rng = np.random.default_rng(42)

    # Generate a diverse set of seed geometries
    seeds = []
    # Dielectric background with 4 air holes (inspiration 1's top strategy)
    for r in [0.22, 0.25, 0.28, 0.31, 0.34]:
        seeds.append(_make_holes(r, n_holes=4, hole_r=r))
    # 2 holes
    for r in [0.25, 0.30, 0.35]:
        seeds.append(_make_holes(r, n_holes=2, hole_r=r))
    # 1 central hole
    for r in [0.20, 0.25, 0.30]:
        seeds.append(_make_holes(r, n_holes=1, hole_r=r))
    # Rings
    seeds.append(_make_ring(0.10, 0.30))
    seeds.append(_make_ring(0.15, 0.35))
    seeds.append(_make_ring(0.20, 0.40))
    # Cross
    seeds.append(_make_cross(4))
    seeds.append(_make_cross(6))
    # Checkerboard
    seeds.append(_make_checkerboard(8))
    seeds.append(_make_checkerboard(4))
    # Squares
    seeds.append(_make_square(0.20, inverted=False))
    seeds.append(_make_square(0.25, inverted=True))

    # Evaluate all seeds
    best_score = -1.0
    best_cell = seeds[0].copy()
    for seed in seeds:
        if time.time() > deadline - 60.0:
            break
        s = _score(seed)
        if s > best_score:
            best_score = s
            best_cell = seed.copy()

    # Local search from the best seed
    if time.time() < deadline - 5.0:
        best_cell, best_score = _local_search(best_cell, best_score, rng, deadline - 5.0)

    # A second local search from a different promising seed (if time remains)
    if time.time() < deadline - 2.0 and len(seeds) > 1:
        # Pick the second-best seed
        scores = []
        for seed in seeds:
            s = _score(seed)
            scores.append(s)
        # Find the seed with the highest score that is not the current best
        sorted_indices = np.argsort(scores)[::-1]
        second_seed = None
        for idx in sorted_indices:
            if not np.allclose(seeds[idx], best_cell):
                second_seed = seeds[idx]
                break
        if second_seed is not None:
            c2, s2 = _local_search(second_seed, scores[sorted_indices[0]], rng, deadline - 2.0)
            if s2 > best_score:
                best_cell = c2
                best_score = s2

    # Final symmetry enforcement
    best_cell = _symmetrize(best_cell)

    return [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]

# EVOLVE-BLOCK-END
