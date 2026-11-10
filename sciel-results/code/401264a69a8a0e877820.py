"""Evolved program: fast bit-parallel mono-4-AP counter + multi-start SA + greedy polish."""
import time
import numpy as np

N = 256
DEADLINE = 140.0          # safety margin before the 150 s wall-clock kill
_DEADLINE = time.time() + DEADLINE


def _time_left():
    return _DEADLINE - time.time()


def _build_aps():
    """All 4-term APs in 1..256 (0-indexed). Returns (masks, counts)."""
    masks = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            m = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d)) | (1 << (a + 3 * d))
            masks.append(m)
    masks = np.array(masks, dtype=np.uint64)
    # Precompute per-AP colour-count (number of 1-bits in each of the 4 positions)
    # We store the mask; counts computed on the fly.
    return masks


_APS = _build_aps()
# Precompute the 4 position bitmasks per AP for fast counting via lookup
# Actually simpler: count = popcount(m0 & M) for colour 0; mono if ==0 or ==4
# Use np.bitwise_and + a popcount. For speed precompute popcount of all 16 4-bit values?
# The 4 positions are not contiguous, so we can't use a 4-bit table directly.
# Instead compute per-AP: c0 = popcount(m0 & M), mono if c0==0 or c0==4.
# We need popcount of (m0 & M) which is a 256-bit & -> up to 4 bits set.
# Vectorize: for each AP, c0 = popcount(m0 & M). We can compute by summing 4 bit tests.
# Build the 4 index arrays.
def _build_idx():
    idx = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            idx.append((a, a + d, a + 2 * d, a + 3 * d))
    idx = np.array(idx, dtype=np.int64)
    return idx

_IDX = _build_idx()
_I0 = _IDX[:, 0]
_I1 = _IDX[:, 1]
_I2 = _IDX[:, 2]
_I3 = _IDX[:, 3]


def _count_mono(m0):
    """m0: np.uint64 array of 256 bits (colour-0 set). Return mono 4-AP count."""
    # colour of each position: 1 if in m0 else 0
    c0 = (m0[_I0] != 0)
    c1 = (m0[_I1] != 0)
    c2 = (m0[_I2] != 0)
    c3 = (m0[_I3] != 0)
    # mono if all equal
    mono = ((c0 == c1) & (c1 == c2) & (c2 == c3))
    return int(mono.sum())


def _to_mask(bits):
    """bits: array of 256 0/1 -> uint64 array of 256 (bit i set if bits[i]==1)."""
    m = np.zeros(256, dtype=np.uint64)
    for i in range(256):
        if bits[i]:
            m[i] = 1 << i
    return m


def _bits_from_mask(m):
    out = np.zeros(256, dtype=np.int8)
    for i in range(256):
        out[i] = 1 if (m[i] != 0) else 0
    return out


def _flip(m, pos):
    m[pos] = (1 << pos) if m[pos] == 0 else 0
    return m


def _greedy_polish(m, max_passes=40):
    """Greedy: repeatedly flip any single bit that reduces mono count."""
    cur = _count_mono(m)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if _time_left() < 3.0:
            break
        # try each position
        best_gain = 0
        best_pos = -1
        for i in range(256):
            old = m[i]
            m[i] = (1 << i) if old == 0 else 0
            nv = _count_mono(m)
            gain = cur - nv
            if gain > best_gain:
                best_gain = gain
                best_pos = i
            m[i] = old
        if best_pos >= 0:
            m[best_pos] = (1 << best_pos) if m[best_pos] == 0 else 0
            cur -= best_gain
            improved = True
    return m, cur


def _sa(m0, iters, t0, t1):
    """Simulated annealing with 1-bit and 3-bit flips. Returns (mask, count)."""
    m = m0.copy()
    cur = _count_mono(m)
    rng = np.random.default_rng()
    n_iter = 0
    for it in range(iters):
        if _time_left() < 4.0:
            break
        frac = it / max(1, iters)
        T = t0 + (t1 - t0) * frac
        # choose flip size
        if rng.random() < 0.35:
            size = 3
        else:
            size = 1
        pos = rng.integers(0, 256, size=size)
        for p in pos:
            m[p] = (1 << p) if m[p] == 0 else 0
        nv = _count_mono(m)
        d = nv - cur
        if d <= 0 or (T > 0 and rng.random() < np.exp(-d / T)):
            cur = nv
        else:
            for p in pos:
                m[p] = (1 << p) if m[p] == 0 else 0
        n_iter += 1
    return m, cur


def construct():
    rng = np.random.default_rng(12345)
    best_m = None
    best_c = 10 ** 9

    # Seed: alternate blocks of two (the reference), plus random starts.
    seeds = []
    # reference
    ref = np.zeros(256, dtype=np.int8)
    for i in range(256):
        ref[i] = (i // 2) % 2
    seeds.append(_to_mask(ref))
    # a few structured seeds
    for period in (1, 2, 3, 4, 5, 7, 8):
        s = np.zeros(256, dtype=np.int8)
        for i in range(256):
            s[i] = (i // period) % 2
        seeds.append(_to_mask(s))
    # random seeds
    for _ in range(6):
        s = rng.integers(0, 2, 256).astype(np.int8)
        seeds.append(_to_mask(s))

    for seed in seeds:
        if _time_left() < 5.0:
            break
        # quick SA then polish
        m, c = _sa(seed, iters=4000, t0=2.0, t1=0.05)
        m, c = _greedy_polish(m, max_passes=30)
        if c < best_c:
            best_c = c
            best_m = m.copy()

    # Final long polish on the best found
    if best_m is not None and _time_left() > 3.0:
        best_m, best_c = _greedy_polish(best_m, max_passes=60)

    bits = _bits_from_mask(best_m)
    return [int(x) for x in bits]
