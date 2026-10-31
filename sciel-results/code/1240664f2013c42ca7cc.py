import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (combined design, smooth surrogates):
    1. Primary: Center score — product of distances to all-zeros and all-twos.
       Vectors near the center of the cube are geometrically less likely to
       form 3-term APs with boundary vectors. This is a single smooth
       geometric signal that naturally rewards balanced, multi-symbol vectors.
    2. Secondary: Shannon entropy of the symbol distribution. Smoother than
       the discrete distinct-count, captures the degree of balance.
    3. Tertiary: Symmetry-breaking 2-weight. Favors vectors with more 2s to
       prevent clustering in the {0,1}^n sub-cube.
    4. Quaternary: Variance-based danger proxy. High coordinate variance
       indicates more "extreme" coordinates, structurally more likely to
       complete an affine line.
    5. Quinary: Polynomial rolling hash for deterministic tie-breaking.

    All components computed in O(n) time with minimal per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting in one pass
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

    # Factor 1: Center score — product of distances to all-zeros and all-twos.
    # dist_to_zero = c1 + 2*c2 (L1 distance from (0,0,...,0) in {0,1,2}^n)
    # dist_to_two  = 2*c0 + c1 (L1 distance from (2,2,...,2))
    # Product peaks at the center (all 1s) and is zero at the corners.
    dist_to_zero = c1 + 2 * c2
    dist_to_two = 2 * c0 + c1
    center_score = dist_to_zero * dist_to_two

    # Factor 2: Shannon entropy of the symbol distribution.
    # H = -sum(p_i * log(p_i)), with 0*log(0) = 0.
    # Max entropy log(3) ≈ 1.0986 when perfectly balanced.
    entropy = 0.0
    if c0 > 0:
        p0 = c0 / n
        entropy -= p0 * math.log(p0)
    if c1 > 0:
        p1 = c1 / n
        entropy -= p1 * math.log(p1)
    if c2 > 0:
        p2 = c2 / n
        entropy -= p2 * math.log(p2)
    entropy_score = entropy * 500.0

    # Factor 3: Symmetry-breaking 2-weight.
    # Favors vectors with more 2s to break the ternary grid symmetry
    # and prevent the greedy algorithm from clustering in {0,1}^n.
    weight_score = c2 * 50.0

    # Factor 4: Variance-based danger proxy.
    # Compute sum and sum-of-squares for variance.
    s = c1 + 2 * c2
    s2 = c1 + 4 * c2
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 3000.0

    # Factor 5: Polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = center_score + entropy_score + weight_score + danger_score + hash_score
    return float(total)
