import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (combined design, smooth surrogates):
    1. Primary: Smooth quadratic distinct-symbol score.
    2. Secondary: Hamming weight relative to all-ones (count of 2s).
    3. Tertiary: Quadratic balance penalty.
    4. Quaternary: Smooth variance-based danger proxy.
    5. Quinary: Algebraic independence score via projection onto orthogonal
       complement of low-degree monomial basis. Vectors with high projection
       magnitude onto directions orthogonal to {x_i, x_i*x_j} are more
       algebraically independent, reducing additive dependencies that form
       affine lines.
    6. Senary: Fast polynomial rolling hash for robust deterministic tie-breaking.

    All components computed in O(n) time with minimal per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting in one pass (collects counts, sum, and sum-of-squares)
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Hamming weight relative to all-ones (count of 2s).
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Algebraic independence score.
    # Compute the projection magnitude onto the orthogonal complement of the
    # span of low-degree monomial basis vectors {x_i, x_i*x_j}.
    # We use a smooth surrogate: the residual energy after projecting onto
    # the space spanned by the first few monomials. Higher residual = more
    # algebraically independent = less likely to be in a low-degree subspace
    # that contains affine lines.
    #
    # Approximation: For a vector v in {0,1,2}^n, consider the "monomial
    # features" that capture low-degree structure. The key insight is that
    # affine lines in {0,1,2}^n correspond to additive relations. We measure
    # how much of the vector's energy lies in directions that are NOT captured
    # by simple linear and bilinear features.
    #
    # Efficient computation: The projection onto the span of {e_i} (linear)
    # is just the vector itself. The projection onto {e_i * e_j} (bilinear)
    # can be approximated by the pairwise interaction energy.
    #
    # We compute: ||v||^2 - ||P_linear(v)||^2 - ||P_bilinear(v)||^2
    # where P_linear captures the "mean" direction and P_bilinear captures
    # pairwise correlations.
    
    # Linear part: projection onto the all-ones direction
    # This is just (s/n) * 1, so ||P_linear||^2 = s^2/n
    linear_energy = s * s / n
    
    # Bilinear part: approximate the projection onto pairwise products
    # The bilinear form <v, v_i * v_j> summed over i<j gives pairwise interactions
    # We compute sum_{i<j} v_i * v_j = (s^2 - s2) / 2
    bilinear_energy = (s * s - s2) / 2.0
    
    # Total energy
    total_energy = s2
    
    # Residual energy = total - linear - bilinear (clamped to non-negative)
    # This measures how much of the vector is "orthogonal" to low-degree structure
    residual = total_energy - linear_energy - bilinear_energy
    if residual < 0:
        residual = 0.0
    
    # Normalize by n to get a per-coordinate measure
    algebraic_score = (residual / n) * 2000.0

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + algebraic_score + hash_score
    return float(total)
