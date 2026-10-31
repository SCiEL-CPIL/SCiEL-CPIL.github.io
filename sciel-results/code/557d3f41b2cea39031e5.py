"""Program for the cap set task using additive character theory prioritization."""
import numpy as np
from functools import lru_cache

# Precompute a fixed pseudo-random frequency vector in F_3^n for n up to a reasonable bound
# We'll generate a deterministic pseudo-random vector for each n using a simple PRNG seeded by n
def _get_frequency_vector(n):
    """Generate a fixed pseudo-random frequency vector in F_3^n."""
    # Use a simple LCG-based approach to generate deterministic values in {0, 1, 2}
    rng = np.random.RandomState(seed=n * 7919 + 13)
    # Generate n values in {0, 1, 2}
    freq = tuple(rng.randint(0, 3, size=n).tolist())
    return freq

# Cache frequency vectors
_freq_cache = {}

def _get_freq(n):
    if n not in _freq_cache:
        _freq_cache[n] = _get_frequency_vector(n)
    return _freq_cache[n]

# Precompute character sum for a vector given a frequency vector
# The additive character sum magnitude: |sum over coordinates of exp(2*pi*i*v_i*f_i/3)|
# But since we're in F_3, we can compute the "deviation" from uniformity.
# For a single vector v in {0,1,2}^n, the character evaluation at frequency f is:
#   chi_f(v) = prod_i omega^(v_i * f_i) where omega = exp(2*pi*i/3)
# The magnitude is always 1 for a single vector.
# 
# Instead, let's think about this differently. The key insight for cap sets:
# A set S is a cap set if it contains no three distinct points a, b, c with a+b=c (in F_3^n).
# This is equivalent to saying no three distinct points form an arithmetic progression.
#
# The Fourier-analytic approach: sets with small Fourier mass at non-zero frequencies
# tend to avoid arithmetic progressions. For a single vector, we can prioritize vectors
# that are "far" from having large character sums with common frequency patterns.
#
# A practical heuristic: prioritize vectors based on the sum of v_i * f_i mod 3,
# where f is a fixed pseudo-random frequency. Vectors with character values closer
# to "uniform" (i.e., the weighted sum mod 3 is balanced) are preferred.
#
# Actually, let's use a more direct approach: the priority should reflect how
# "random" the vector looks with respect to the frequency vector. We compute
# the additive character value and use its real part as a proxy for deviation.

def priority(vector, n):
    """
    Priority based on additive character theory.
    
    For a vector v in {0,1,2}^n and a fixed pseudo-random frequency f in F_3^n,
    compute the additive character chi_f(v) = omega^(sum v_i * f_i mod 3).
    The priority is based on how "balanced" this character value is.
    
    Since omega = exp(2*pi*i/3), the character values are {1, omega, omega^2}.
    We map these to real priorities:
      chi = 1   -> 0.0 (most "aligned" with the frequency, least preferred)
      chi = omega -> 1.0 (moderately preferred)
      chi = omega^2 -> 2.0 (most "orthogonal", most preferred)
    
    Additionally, we add a small tie-breaking term based on the weighted sum itself
    to break symmetries.
    """
    freq = _get_freq(n)
    
    # Compute the weighted sum mod 3
    weighted_sum = 0
    for i in range(n):
        weighted_sum += vector[i] * freq[i]
    weighted_sum %= 3
    
    # Map the character value to a priority
    # chi_0 = 1, chi_1 = omega, chi_2 = omega^2
    # We want to prefer vectors where the character is NOT 1 (i.e., not aligned)
    if weighted_sum == 0:
        base_priority = 0.0
    elif weighted_sum == 1:
        base_priority = 1.0
    else:  # weighted_sum == 2
        base_priority = 2.0
    
    # Add a small perturbation to break ties: use the actual weighted sum (before mod)
    # normalized to [0, 1)
    raw_sum = 0
    for i in range(n):
        raw_sum += vector[i] * freq[i]
    
    # Small tie-breaker: fractional part of raw_sum / (3*n) to spread priorities
    tie_breaker = (raw_sum % (3 * n + 1)) / (3 * n + 1)
    
    # Combine: base priority dominates, tie-breaker is small
    return base_priority + 0.001 * tie_breaker
