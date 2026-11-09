import zlib
import time

# EVOLVE-BLOCK-START
_START_TIME = time.monotonic()
_DEADLINE = _START_TIME + 320.0  # safety margin before 330s kill

def priority(vector, n):
    # Deadline guard: if we are too close to the wall-clock limit,
    # return a neutral value to avoid any further work in the evaluator.
    if time.monotonic() > _DEADLINE:
        return 0.0

    # Structural ranking: prefer vectors whose coordinates are "balanced"
    # across {0,1,2}.  A balanced vector is less likely to form a 3-term
    # arithmetic progression with other balanced vectors, so the greedy
    # process can accept more vectors before the cap-set constraint binds.
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

    # Primary key: product of counts (maximized when counts are equal).
    # Secondary key: sum of squared deviations from n/3 (minimized when balanced).
    # Tertiary key: penalty for repeated coordinates (smooth AP-avoidance).
    #   A coordinate value appearing k times contributes C(k,2) potential
    #   "midpoints" that could form an AP with another vector sharing that
    #   value in the same coordinate.  Minimizing this is a smooth surrogate
    #   for keeping the set cap-free.
    # Quaternary key: deterministic CRC32 hash to break ties pseudo-randomly.
    balance_product = float(c0 * c1 * c2)
    # Higher-order balance term: product of (count - mean)^4 penalizes
    # strong imbalance more aggressively than variance alone.
    mean = n / 3.0
    variance_penalty = (c0 - mean) ** 2 + (c1 - mean) ** 2 + (c2 - mean) ** 2
    quartic_penalty = abs(c0 - mean) ** 4 + abs(c1 - mean) ** 4 + abs(c2 - mean) ** 4
    # Repeated-coordinate penalty: sum over values of C(k,2)
    repeat_penalty = float(c0 * (c0 - 1) + c1 * (c1 - 1) + c2 * (c2 - 1))
    h = zlib.crc32(bytes(vector))

    # Combine: dominant structural term, then variance, then quartic,
    # then repeat penalty, then hash.
    # Use large multipliers to ensure lexicographic-like ordering of the keys.
    score = (balance_product * 1e14
             - variance_penalty * 1e8
             - quartic_penalty * 1e4
             - repeat_penalty * 1e2
             + float(h))
    return score
# EVOLVE-BLOCK-END
