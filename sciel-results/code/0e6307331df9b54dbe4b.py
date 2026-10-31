"""Cap set priority: center-distance + smooth surrogates for geometric separation."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy:
    1. Primary: Product of distances to all-zeros and all-twos vectors.
       Vectors geometrically central in the ternary grid are less likely to
       form 3-term affine lines with boundary vectors.
    2. Smooth diversity surrogate: product of counts / n^3, which is smooth
       and zero when any symbol is missing.
    3. Smooth balance penalty: variance of the three counts, penalizing
       imbalanced distributions.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Safety: return a reasonable default if deadline exceeded
        return 0.0

    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1

    # Product of distances to all-zeros and all-twos
    dist_to_zero = count_1 + 2 * count_2
    dist_to_two = 2 * count_0 + count_1
    center_score = dist_to_zero * dist_to_two

    # Smooth diversity surrogate: (c0 * c1 * c2) / n^3
    # This is 0 when any count is 0, and maximized when all are ~n/3
    if n > 0:
        diversity_smooth = (count_0 * count_1 * count_2) / (n * n * n)
    else:
        diversity_smooth = 0.0

    # Smooth balance penalty: variance of the three counts
    # Mean count = n/3, variance measures deviation from uniform
    mean_count = n / 3.0
    var_counts = ((count_0 - mean_count) ** 2 +
                  (count_1 - mean_count) ** 2 +
                  (count_2 - mean_count) ** 2) / 3.0
    # Normalize: max variance when one count is n, others 0: (2n^2/3)/3 = 2n^2/9
    if n > 0:
        max_var = 2.0 * n * n / 9.0
        balance_penalty = var_counts / max_var if max_var > 0 else 0.0
    else:
        balance_penalty = 0.0

    # Combine: center score dominates, diversity encourages all symbols,
    # balance penalty discourages extreme imbalance
    return center_score + 3.0 * diversity_smooth * n - 1.5 * balance_penalty * n
# EVOLVE-BLOCK-END
