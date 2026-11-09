"""Cap set greedy-priority design.

The evaluator calls priority(vector, n) once per candidate vector and inserts
vectors greedily in descending priority order, keeping the set a cap set.
We therefore return a STATIC score per vector that encodes a good insertion
order.  The score favours vectors that:
  * have a balanced digit composition (maximises the cap-set density region),
  * have a "structured" per-coordinate spread (low variance of digit counts),
  * have a low, deterministic hash (tie-breaker that is stable across calls).
"""
import time

# Deadline guard: never run the evaluator past ~300s of our own work.
_DEADLINE = time.time() + 290.0


def _composition(v):
    """Return (c0, c1, c2) counts of digits 0,1,2 in the tuple v."""
    c0 = c1 = c2 = 0
    for x in v:
        if x == 0:
            c0 += 1
        elif x == 1:
            c1 += 1
        else:
            c2 += 1
    return c0, c1, c2


def _balance_score(c0, c1, c2, n):
    """Higher = closer to the balanced (n/3, n/3, n/3) composition.

    Balanced compositions sit near the centre of the affine cube where the
    cap-set density is highest, so a greedy pass that visits balanced vectors
    first tends to keep more of them.
    """
    # Distance from the ideal balanced point, normalised.
    d = abs(c0 - c1) + abs(c1 - c2) + abs(c0 - c2)
    # d in [0, 2n]; 0 is perfectly balanced.
    return -d / (2.0 * n) if n > 0 else 0.0


def _spread_score(v, n):
    """Reward vectors whose per-coordinate digits are 'spread out'.

    A vector that uses all three digits across its coordinates tends to
    participate in fewer 3-term arithmetic progressions with other balanced
    vectors, so it is a safer early insertion.  We measure the variance of
    the digit values; higher variance (more spread) gets a small bonus.
    """
    if n == 0:
        return 0.0
    mean = sum(v) / n
    var = 0.0
    for x in v:
        dx = x - mean
        var += dx * dx
    var /= n
    # var in [0, 4/3]; normalise to [0,1]
    return var / (4.0 / 3.0)


def _hash_score(v, n):
    """Deterministic low hash as a tie-breaker.

    Using a small, well-mixed hash keeps the order stable and avoids
    pathological clustering of equal-priority vectors.
    """
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) & 0xFFFFFFFF
    # Map to [0,1); we SUBTRACT it so lower-hash vectors come first.
    return -(h / 4294967295.0) * 0.001  # tiny weight


def priority(vector, n):
    """Return a float priority; higher is inserted earlier.

    Static, O(n) per call, well within the 300s budget for any realistic n.
    """
    # Deadline guard (should essentially never trigger for O(n) work).
    if time.time() > _DEADLINE:
        return 0.0

    if n <= 0:
        return 0.0

    c0, c1, c2 = _composition(vector)

    # Primary: balance (dominant term).
    s = _balance_score(c0, c1, c2, n)

    # Secondary: spread (small weight so it only breaks near-ties).
    s += 0.05 * _spread_score(vector, n)

    # Tertiary: deterministic hash tie-breaker (very small weight).
    s += _hash_score(vector, n)

    return float(s)
