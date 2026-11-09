"""Design a 32x32 C4v-symmetric photonic-crystal unit cell maximizing the TM 4-5 band gap."""
import math
import time
import random
import numpy as np
from scipy.linalg import eigh

_SIZE = 32
_NPW = 17  # 17x17 = 289 plane waves

# Precompute G vectors
_Gvecs = []
for _a in range(-_NPW // 2, _NPW // 2 + 1):
    for _b in range(-_NPW // 2, _NPW // 2 + 1):
        _Gvecs.append((_a, _b))
_NG = len(_Gvecs)  # 289

# Build C4v orbits
def _transforms(i, j):
    yield (i, j)
    yield (j, _SIZE - 1 - i)
    yield (_SIZE - 1 - i, _SIZE - 1 - j)
    yield (_SIZE - 1 - j, i)
    yield (i, _SIZE - 1 - j)
    yield (_SIZE - 1 - i, j)
    yield (j, i)
    yield (_SIZE - 1 - j, _SIZE - 1 - i)

_orbits = []
_seen = set()
for i in range(_SIZE):
    for j in range(_SIZE):
        if (i, j) in _seen:
            continue
        orbit = []
        for (ii, jj) in _transforms(i, j):
            if (ii, jj) not in _seen:
                _seen.add((ii, jj))
                orbit.append((ii, jj))
        _orbits.append(orbit)
_NUM_ORBITS = len(_orbits)

_cell_to_orbit = [[0] * _SIZE for _ in range(_SIZE)]
for _oi, _orb in enumerate(_orbits):
    for (_i, _j) in _orb:
        _cell_to_orbit[_i][_j] = _oi


def _build_cell(orbit_bits):
    cell = [[0] * _SIZE for _ in range(_SIZE)]
    for oi, bit in enumerate(orbit_bits):
        if bit:
            for (i, j) in _orbits[oi]:
                cell[i][j] = 1
    return cell


def _compute_tm_bands(cell_array, kx, ky):
    """Compute the first 5 TM band frequencies at wavevector (kx, ky)."""
    n = 32
    eps = 1.0 + 11.0 * cell_array
    epsinv = 1.0 / eps
    fft_vals = np.fft.fft2(epsinv) / (n * n)

    # Precompute |K+G|^2
    kplus_g_sq = np.array([(kx + gx) ** 2 + (ky + gy) ** 2 for (gx, gy) in _Gvecs])

    # Build symmetric matrix A[gi, gj] = sqrt(kplus_g_sq[gi] * kplus_g_sq[gj]) * epsinv[G_i - G_j]
    A = np.zeros((_NG, _NG), dtype=np.float64)
    for gi in range(_NG):
        gx_i, gy_i = _Gvecs[gi]
        for gj in range(_NG):
            gx_j, gy_j = _Gvecs[gj]
            dx = gx_i - gx_j
            dy = gy_i - gy_j
            e = fft_vals[dx % n, dy % n].real
            A[gi, gj] = math.sqrt(kplus_g_sq[gi] * kplus_g_sq[gj]) * e

    evals, _ = eigh(A, subset_by_index=[0, 4])
    return evals


# K-points in the irreducible BZ
_KPOINTS = []
for kx in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for ky in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        if kx <= ky + 1e-9:
            _KPOINTS.append((kx, ky))


def _compute_gap(cell_array):
    max_omega4 = 0.0
    min_omega5 = float('inf')
    for (kx, ky) in _KPOINTS:
        evals = _compute_tm_bands(cell_array, kx, ky)
        omega4 = evals[3]
        omega5 = evals[4]
        max_omega4 = max(max_omega4, omega4)
        min_omega5 = min(min_omega5, omega5)
    gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    return gap


def design(problem):
    start_time = time.time()
    deadline = start_time + 200

    # Initial guess: centered disk
    init_bits = [0] * _NUM_ORBITS
    for oi, orb in enumerate(_orbits):
        i, j = orb[0]
        x = (i + 0.5) / 32.0 - 0.5
        y = (j + 0.5) / 32.0 - 0.5
        if x * x + y * y < 0.25:
            init_bits[oi] = 1

    best_bits = list(init_bits)
    cell = _build_cell(best_bits)
    cell_array = np.array(cell, dtype=np.float64)
    best_score = _compute_gap(cell_array)

    rng = random.Random(42)
    step_size = 0.25
    stall = 0

    while time.time() < deadline:
        for _ in range(30):
            if time.time() > deadline:
                break
            oi = rng.randint(0, _NUM_ORBITS - 1)
            old = best_bits[oi]
            best_bits[oi] = 1 - old
            cell = _build_cell(best_bits)
            cell_array = np.array(cell, dtype=np.float64)
            score = _compute_gap(cell_array)
            if score > best_score + 1e-9:
                best_score = score
                stall = 0
                step_size = min(0.5, step_size * 1.05)
            else:
                best_bits[oi] = old
                stall += 1
                step_size = max(0.05, step_size * 0.97)

        if stall > 80:
            for oi in range(_NUM_ORBITS):
                if rng.random() < 0.15:
                    best_bits[oi] = 1 - best_bits[oi]
            stall = 0
            step_size = 0.25

    return _build_cell(best_bits)
