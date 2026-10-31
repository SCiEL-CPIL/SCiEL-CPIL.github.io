import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (Refined Smooth Surrogates):
    1. Primary: Smooth quadratic distinct-symbol score.
    2. Secondary: Entropy-based balance score. Maximizes the Shannon entropy of the
       symbol distribution, which is a smoother and more geometrically meaningful
       measure of balance than L2 deviation.
    3. Tertiary: Variance-based danger proxy. Penalizes high variance to avoid
       "extreme" vectors that cluster in corners and form lines more easily.
    4. Quaternary: Skewness-based symmetry breaking. Uses the third moment to
       break the 0/2 symmetry more intelligently than a simple count of 2s.
    5. Quinary: Enhanced hash tie-breaking. Increased weight of the polynomial
       hash to break the large plateaus created by the distinct-symbol score,
       ensuring more diverse exploration within the high-priority set.

    All components computed in O(n) time.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    s3 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
            s3 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4
            s3 += 8

    # Factor 1: Smooth quadratic distinct-symbol score.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Entropy-based balance score.
    # H = - sum p_i log(p_i). We want to maximize this.
    # Add a small epsilon to avoid log(0).
    eps = 1e-9
    p0 = c0 / n + eps
    p1 = c1 / n + eps
    p2 = c2 / n + eps
    # Normalize probabilities roughly (adding eps shifts sum slightly, but for large n it's fine)
    # Strictly, p should sum to 1. Let's normalize properly.
    sum_p = p0 + p1 + p2
    p0 /= sum_p
    p1 /= sum_p
    p2 /= sum_p
    
    entropy = -(p0 * math.log(p0) + p1 * math.log(p1) + p2 * math.log(p2))
    # Max entropy is log(3) ~ 1.0986.
    # Scale to comparable magnitude.
    entropy_score = entropy * 1000.0

    # Factor 3: Variance-based danger proxy.
    mean = s / n
    variance = s2 / n - mean * mean
    # Stronger penalty than the net effect of the previous version (-2000),
    # but we removed the redundant algebraic term.
    danger_score = -variance * 3000.0

    # Factor 4: Skewness-based symmetry breaking.
    # Skewness = E[(X - mu)^3].
    # For {0,1,2}, this helps distinguish between "more 0s" and "more 2s".
    # We compute the third central moment.
    # E[X^3] - 3mu E[X^2] + 2 mu^3
    e3 = s3 / n
    e2 = s2 / n
    skewness = e3 - 3 * mean * e2 + 2 * mean * mean * mean
    # We want to break symmetry. A positive skewness (more 2s) or negative (more 0s)?
    # Let's favor a specific direction to break the 0/2 symmetry deterministically.
    # Favouring 2s slightly (positive skewness) to match the previous "weight_score" intent.
    skewness_score = skewness * 50.0

    # Factor 5: Enhanced hash tie-breaking.
    # Increased weight to break plateaus from distinct_score more effectively.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = (h / 0x7FFFFFFF) * 10.0

    # Combined priority: higher is added first.
    total = distinct_score + entropy_score + danger_score + skewness_score + hash_score
    return float(total)
