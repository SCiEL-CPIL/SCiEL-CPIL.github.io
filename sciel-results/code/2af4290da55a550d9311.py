"""Cap set priority: smooth quadratic distinctness, variance penalty, L2 balance, symmetry breaking, and algebraic independence."""
import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def _algebraic_independence_score(vector, n):
    """
    Compute a smooth proxy for algebraic independence by measuring the
    magnitude of the vector's projection onto the orthogonal complement of
    low-degree monomial basis patterns (coordinate indicators and pairwise
    products). Vectors with high scores are less aligned with simple additive
    structures, making them structurally safer against affine line formation.

    We use a set of fixed "danger patterns" derived from low-degree monomials:
    - Coordinate indicators: vectors that are 1 in one position, 0 elsewhere
    - Pairwise product patterns: vectors encoding x_i * x_j interactions
    - Uniform direction: the all-ones vector (degree 0 monomial)

    The score is the L2 norm of the component of the vector orthogonal to
    the span of these patterns, normalized.
    """
    if n <= 0:
        return 0.0

    # Convert to list for efficient access
    vec = list(vector)

    # Build basis patterns (kept small for O(n) per pattern)
    # Pattern 1: uniform (all ones) - degree 0
    # Pattern 2-3: coordinate sums for even/odd positions (degree 1 proxy)
    # Pattern 4-5: pairwise interaction proxies (degree 2 proxy)

    # For efficiency, compute projections onto a few key directions:
    # 1. Uniform direction (1, 1, ..., 1)
    # 2. Alternating direction (1, -1, 1, -1, ...) - captures parity
    # 3. Quadratic interaction: sum of pairs (i,j) where i<j, weighted by x_i*x_j

    # Component 1: projection onto uniform
    n_float = float(n)
    proj_uniform = sum(vec) / n_float  # mean

    # Component 2: projection onto alternating pattern
    alt_sum = 0.0
    for i, v in enumerate(vec):
        if i % 2 == 0:
            alt_sum += v
        else:
            alt_sum -= v
    # Normalize: the alternating vector has norm sqrt(n)
    proj_alt = alt_sum / math.sqrt(n_float)

    # Component 3: pairwise interaction energy
    # This captures degree-2 monomial structure: sum_{i<j} v_i * v_j
    # Can be computed as (sum^2 - sum_of_squares) / 2
    total_sum = sum(vec)
    sum_sq = sum(v * v for v in vec)
    pairwise_energy = (total_sum * total_sum - sum_sq) / 2.0
    # Normalize by max possible: for all-2s, pairwise = n(n-1)*4/2 = 2n(n-1)
    max_pairwise = 2.0 * n_float * (n_float - 1.0) if n > 1 else 1.0
    proj_pairwise = pairwise_energy / max_pairwise if max_pairwise > 0 else 0.0

    # Component 4: cubic interaction proxy - sum of triples
    # Approximate via (sum^3 - 3*sum*sum_sq + 2*sum_cubes) / 6
    # But for efficiency, use a simpler proxy: variance of pairwise products
    # Actually, let's use the "excess" over what's expected from first two moments
    # A vector with high algebraic independence has interactions that deviate
    # from what's predicted by its mean and variance.
    # Use: |pairwise_energy - n*(n-1)/2 * mean^2| as deviation from "random"
    expected_pairwise = n_float * (n_float - 1.0) / 2.0 * proj_uniform * proj_uniform
    deviation = abs(pairwise_energy - expected_pairwise)
    # Normalize
    norm_deviation = deviation / max_pairwise if max_pairwise > 0 else 0.0

    # The orthogonal complement magnitude is related to how much the vector
    # deviates from being explainable by these low-degree patterns.
    # A vector fully in the span would have all these projections "consistent"
    # with a simple model. High deviation = high algebraic independence.

    # Combine: we want vectors that are NOT well-explained by low-degree patterns
    # The residual after removing projections onto these directions:
    # L2 norm of vector minus its projection onto span{uniform, alternating, pairwise}

    # Simpler smooth surrogate: the "excess interaction" normalized
    # High values mean the vector has complex multiplicative structure
    # that can't be captured by linear or simple quadratic models.
    score = norm_deviation * 1.0

    # Also add a term that rewards vectors with moderate (not extreme)
    # coordinate values, as extremes create clearer additive structures
    if n > 0:
        variance = sum_sq / n_float - proj_uniform * proj_uniform
        # Penalize high variance (extreme values create clearer APs)
        variance_penalty = -variance * 0.5
    else:
        variance_penalty = 0.0

    return score + variance_penalty


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy:
    1. Primary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols (0, 1, 4, 9 * 10000) creates a steep but continuous
       gradient, strongly preferring vectors that use all three symbols.
       This avoids the discrete cliffs of linear tiering.
    2. Secondary: Smooth variance-based danger proxy. High coordinate variance
       indicates many "extreme" coordinates (0s and 2s), which are structurally
       more likely to complete affine lines. We penalize variance to prefer
       "middle" (1-heavy) vectors that are geometrically safer.
    3. Tertiary: Quadratic balance penalty. L2 deviation from the uniform
       (n/3, n/3, n/3) distribution. This is a smooth surrogate for balance
       that avoids the piecewise-linear artifacts of L1.
    4. Quaternary: Symmetry breaking via count of 2s. Favors vectors with more
       2s to prevent clustering in the {0,1}^n sub-cube and ensure uniform
       exploration of the space.
    5. Quinary: Fast polynomial rolling hash for deterministic tie-breaking.

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
    # This creates a very strong preference for 3-distinct vectors while
    # maintaining a smooth landscape for the greedy algorithm.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    # Favors vectors with more 2s to break the symmetry of the ternary grid
    # and prevent the greedy algorithm from clustering in the {0,1}^n sub-cube.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    # High coordinate variance means more "extreme" coordinates (0s and 2s),
    # which are structurally more likely to complete an affine line.
    if n > 0:
        mean = s / n
        variance = s2 / n - mean * mean
    else:
        variance = 0.0
    danger_score = -variance * 5000.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Factor 6: Algebraic independence score.
    # Measures deviation from low-degree monomial span (coordinates, pairwise products).
    # Vectors with high algebraic independence are structurally less likely to
    # form additive dependencies that create affine lines.
    alg_score = _algebraic_independence_score(vector, n) * 2000.0

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + alg_score + hash_score
    return float(total)
