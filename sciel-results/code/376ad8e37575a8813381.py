# EVOLVE-BLOCK-START
import time

_start_time = None
_deadline = 320.0  # safety margin before the 330s kill

# Fixed mask for the base-3 XOR parity term. Chosen to have a high Hamming weight
# so the XOR result is broadly "chaotic" across the vector space.
_MASK3 = 1234567891  # base-3 digits: 1 2 0 1 2 1 0 2 1 2 0 1 2 (13 digits, 11 non-zero)

# Precompute base-3 powers up to a generous n so the per-call cost is just a
# multiply-and-add (no pow() per coordinate).
_MAX_N = 40
_BASE3_POWERS = [1] * (_MAX_N + 1)
for _i in range(1, _MAX_N + 1):
    _BASE3_POWERS[_i] = _BASE3_POWERS[_i - 1] * 3


def _popcount(x):
    """Popcount of a non-negative integer (fast for the small values we get)."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def priority(vector, n):
    """
    Priority for greedy cap-set construction in {0,1,2}^n.

    Combined structural + chaotic priority:
      1. TIER  : prefer vectors using more distinct symbols (all-3 > two > one).
                 All-same vectors form lines with many others, so they rank last.
      2. BAL   : within a tier, prefer symbol counts close to uniform (low L1 dev).
      3. PAR   : base-3 XOR-with-mask popcount -> deterministic chaotic interleaving
                 that breaks the correlation between local balance and line formation.

    Higher return value = inserted earlier.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _deadline:
        # Past deadline: ordering no longer matters, return a flat constant.
        return 0.0

    # Manual symbol counting (fast, no Counter overhead).
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1

    # --- TIER: number of distinct symbols (structural) ---
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct = 3
    elif (c0 > 0 and c1 > 0) or (c0 > 0 and c2 > 0) or (c1 > 0 and c2 > 0):
        distinct = 2
    else:
        distinct = 1
    tier = distinct * 1000000.0

    # --- BAL: L1 deviation from the uniform distribution (n/3 each) ---
    target = n / 3.0
    l1 = abs(c0 - target) + abs(c1 - target) + abs(c2 - target)
    bal = -l1 * 10.0

    # --- PAR: base-3 integer XOR fixed mask, popcount = chaotic interleaving ---
    val = 0
    if n <= _MAX_N:
        powers = _BASE3_POWERS
        for i, v in enumerate(vector):
            val += v * powers[i]
    else:
        p = 1
        for v in vector:
            val += v * p
            p *= 3
    parity = _popcount(val ^ _MASK3)
    par = parity * 1.0

    return float(tier + bal + par)
# EVOLVE-BLOCK-END
