"""Cap set construction via spectral-aware greedy selection."""
import time
import math

# EVOLVE-BLOCK-START

# Global state for tracking accepted vectors
_accepted_vectors = set()
_accepted_list = []
_complement_map = {}  # maps vector -> (-v) mod 3
_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill

def _compute_spectral_score(vector, n):
    """
    Compute a spectral penalty based on the discrete Fourier transform
    over Z_3^n. High-frequency components correlate with AP formation.
    
    For a vector v in {0,1,2}^n, we compute the magnitude of its
    Fourier coefficients at non-trivial frequencies. The key insight is
    that vectors with significant high-frequency content are more likely
    to participate in 3-term arithmetic progressions.
    
    We approximate this by computing the sum of squares of Fourier
    coefficients at frequencies with non-zero "weight" (i.e., not the
    trivial frequency). This is equivalent to computing the variance
    of the vector's coordinate distribution in a frequency-aware way.
    
    For efficiency, we use a simplified spectral measure:
    - The total "energy" in non-DC frequencies
    - This is computed as: sum over all non-zero frequency vectors k
      of |sum_i v_i * omega^{k_i * v_i}|^2 where omega = e^{2pii/3}
    
    Since we can't do the full DFT efficiently for large n, we use
    a proxy: the variance of coordinate values, weighted by a
    frequency-dependent kernel.
    
    Actually, let's use a more direct approach: compute the sum of
    v_i^2 - (sum v_i / n)^2 which measures spread, and combine with
    a higher-order moment that captures non-uniformity.
    """
    # Compute basic statistics
    s = sum(vector)
    s2 = sum(x * x for x in vector)
    mean = s / n
    variance = s2 / n - mean * mean
    
    # Higher-order: measure of "clustering" vs "dispersion"
    # Vectors with many 0s and 2s (bimodal) are worse than
    # vectors with many 1s (central)
    count_0 = sum(1 for x in vector if x == 0)
    count_1 = sum(1 for x in vector if x == 1)
    count_2 = sum(1 for x in vector if x == 2)
    
    # Bimodality penalty: high when mass is at extremes
    bimodal_mass = (count_0 + count_2) / n
    # Spectral proxy: penalize bimodal distributions
    spectral_penalty = variance + 0.5 * bimodal_mass * bimodal_mass
    
    return spectral_penalty

def _center_score(vector, n):
    """
    Geometric center score: product of distances to all-zeros and all-twos.
    Vectors closer to the center of the hypercube are safer.
    """
    count_0 = sum(1 for x in vector if x == 0)
    count_1 = sum(1 for x in vector if x == 1)
    count_2 = sum(1 for x in vector if x == 2)
    
    dist_to_zero = count_1 + 2 * count_2
    dist_to_two = 2 * count_0 + count_1
    return dist_to_zero * dist_to_two

def _diversity_bonus(vector, n):
    """
    Bonus for having all three coordinate values present and well-distributed.
    """
    count_0 = sum(1 for x in vector if x == 0)
    count_1 = sum(1 for x in vector if x == 1)
    count_2 = sum(1 for x in vector if x == 2)
    
    if count_0 > 0 and count_1 > 0 and count_2 > 0:
        min_count = min(count_0, count_1, count_2)
        return min_count * 1.5
    elif count_1 > 0 and (count_0 > 0 or count_2 > 0):
        return 0.3 * min(count_0 + count_2, count_1)
    return 0.0

def _count_conflicts(v):
    """Count how many existing vectors a have (-a - v) mod 3 also in the set."""
    n = len(v)
    count = 0
    for a in _accepted_list:
        comp_a = _complement_map[a]
        c = tuple((comp_a[i] - v[i]) % 3 for i in range(n))
        if c in _accepted_vectors:
            count += 1
    return count

def _fnv1a_hash(vector, seed=0):
    """Fast non-linear hash for tie-breaking."""
    h = 1469598103934665603
    for v in vector:
        h ^= v
        h *= 1099511628211
    h ^= seed
    h ^= (h >> 33)
    h *= 0xff51afd7ed558ccd
    h ^= (h >> 33)
    h *= 0xc4ceb9fe1a85ec53
    h ^= (h >> 33)
    return (h & 0xFFFFFFFFFFFFFFFF) / (1 << 64)

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Combines spectral analysis, geometric centering, and conflict avoidance.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    v = tuple(vector)
    
    # Core: conflict count - dominant signal
    if _accepted_list:
        conflicts = _count_conflicts(v)
    else:
        conflicts = 0
    
    # Spectral penalty: penalize high-frequency content
    spectral = _compute_spectral_score(vector, n)
    
    # Geometric center score
    center = _center_score(vector, n)
    
    # Diversity bonus
    diversity = _diversity_bonus(vector, n)
    
    # Hash for deterministic tie-breaking
    h1 = _fnv1a_hash(vector, seed=1)
    
    # Combine: center and diversity are positive, spectral and conflicts are negative
    # Scale to make conflicts dominant
    total = (center * 0.1 + diversity - spectral * 2.0 - conflicts * 5.0 + h1 * 0.01)
    
    return total

def _accept_vector(v):
    """Record that vector v has been accepted into the cap set."""
    _accepted_vectors.add(v)
    _accepted_list.append(v)
    _complement_map[v] = tuple((-x) % 3 for x in v)

# EVOLVE-BLOCK-END
