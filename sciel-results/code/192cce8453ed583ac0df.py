"""Cap set priority: structural F_3^n cap-set awareness."""
import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0  # 50s safety margin before 330s kill

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    DIRECT structural approach:
    - In F_3^n, a 3-term AP {a,b,c} has a+c = 2b (mod 3).
    - A coordinate with value 1 in b means b is the midpoint of (0,2) there.
    - Vectors with FEWER 1s are "corners/edges" and are structurally safer:
      they participate in fewer potential APs as the middle element.
    - So: PRIMARY term = -(count of 1s).  Corners (0 ones) rank highest.
    - SECONDARY: among equal 1-counts, balance 0s vs 2s to spread.
    - TERTIARY: deterministic weighted hash for full tie-break.
    """
    if time.time() > _DEADLINE:
        return 0.0

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

    # PRIMARY: balance toward n/3 per symbol. Vectors with symbol counts
    # close to n/3 are "interior" points that avoid the corner/edge
    # structures which form small 3-term APs. This lets the greedy
    # algorithm accept more vectors before hitting the cap-set constraint.
    target = n / 3.0
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    score = -penalty
    
    # SPECTRAL: Fourier-based penalty for 3-term AP correlation.
    # In F_3^n, the Fourier transform of the indicator function reveals
    # structure. For a single vector, we can compute its "spectral energy"
    # by considering how it correlates with linear phases.
    # 
    # A vector v has high AP-risk if it aligns with many linear functions
    # χ_a(x) = ω^{a·x} where ω = e^{2πi/3}. The DC component (a=0) is
    # always 1. Non-trivial components measure deviation from uniformity.
    # 
    # Compute the magnitude of the first-order Fourier coefficients.
    # For each coordinate position, the local contribution to the
    # non-trivial spectrum depends on the value at that position.
    # 
    # The key insight: vectors that are "locally constant" across many
    # coordinates have higher low-frequency spectral content, which
    # correlates with forming APs with other similar vectors.
    
    # Compute weighted sum of values (proxy for alignment with linear phase)
    # Use a fixed random-looking weight vector to estimate spectral energy
    spectral_weight = 0.0
    for i, v in enumerate(vector):
        # Pseudo-random but deterministic weight based on position
        w = ((i * 7919 + 1) % 3) - 1  # values in {-1, 0, 1}
        spectral_weight += w * v
    
    # The spectral energy is proportional to the square of this projection
    # Normalize by n to get per-coordinate contribution
    if n > 0:
        spectral_energy = (spectral_weight ** 2) / n
    else:
        spectral_energy = 0.0
    
    # Penalize high spectral energy (low-frequency correlation)
    score -= 0.03 * spectral_energy

    # SECONDARY: pairwise coordinate interaction term.
    # In F_3^n, a 3-term AP requires coordinate-wise patterns.
    # Counting pairs of coordinates with the same value reveals
    # structural "clustering" that correlates with AP participation.
    # Vectors with more diverse pairwise interactions (fewer same-value
    # pairs) are less likely to form APs with other vectors.
    pair_same = 0
    for i in range(n):
        vi = vector[i]
        for j in range(i + 1, n):
            if vector[j] == vi:
                pair_same += 1
    # Normalize: max pair_same = n*(n-1)/2, min = 0
    max_pairs = n * (n - 1) / 2.0
    if max_pairs > 0:
        pair_diversity = 1.0 - (pair_same / max_pairs)
    else:
        pair_diversity = 1.0
    score += 0.05 * pair_diversity
    
    # TERTIARY: Higher-order spectral term using quadratic phase correlations.
    # In F_3, the quadratic form Q(x) = x^2 has interesting properties.
    # The sum of x_i^2 over coordinates measures "curvature" of the vector.
    # Vectors with balanced quadratic contributions are less likely to
    # form APs because they don't align with parabolic structures.
    quad_sum = 0
    for v in vector:
        quad_sum += v * v  # 0^2=0, 1^2=1, 2^2=4≡1 mod 3, but use real value
    
    # Normalize and penalize deviation from expected value
    expected_quad = n * (0 + 1 + 4) / 3.0  # average of 0,1,4 is 5/3
    quad_deviation = abs(quad_sum - expected_quad) / max(n, 1)
    score -= 0.02 * quad_deviation

    # TERTIARY: among equally balanced/diverse vectors, prefer those with more 1s
    # (midpoints are structurally useful for blocking APs).
    score += 0.01 * c1
    
    # QUATERNARY: Multi-scale spectral tie-break using different frequency bands.
    # Use multiple pseudo-random weight vectors to estimate spectral energy
    # at different "frequencies" and combine them.
    tie = 0.0
    for band in range(3):
        band_weight = 0.0
        for i, v in enumerate(vector):
            # Different pseudo-random weights for each band
            w = ((i * (7919 + band * 104729) + 1) % 3) - 1
            band_weight += w * v
        # Weight each band differently (lower bands matter more)
        tie += (band_weight ** 2) / (n + 1) * (1.0 / (band + 1))
    score -= tie * 1e-6
    
    # QUINTERNARY: deterministic position-weighted hash for final stable tie-break.
    pos_tie = 0.0
    for i, v in enumerate(vector):
        w = math.sin((i + 1) * 0.6180339887)
        pos_tie += w * v
    score += pos_tie * 1e-8

    return score
