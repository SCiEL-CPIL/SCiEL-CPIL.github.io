import time
import math

_START_TIME = time.time()
TIME_LIMIT = 310.0  # Safety margin before 330s kill

# Precompute a small set of fixed "probe" vectors that we use as anchors.
# For a new candidate v, we measure how "aligned" it is with each probe.
# If v is very aligned (small Hamming-ish distance in F3 sense) with a probe,
# it's more likely to form an AP with that probe and another vector.
# So we penalize high alignment.

# Probe vectors: we use a small set of "structured" vectors that are likely
# to appear in many cap sets or to be problematic.
# Actually, let's use random probes with a fixed seed for reproducibility.

import random
_random_state = random.Random(12345)
_NUM_PROBES = 32
_PROBE_DIM = 30  # Max dimension we'll precompute for
_PROBES = []
for _ in range(_NUM_PROBES):
    probe = tuple(_random_state.randint(0, 2) for _ in range(_PROBE_DIM))
    _PROBES.append(probe)

# Also, we'll use a few "anti-probes" that we want to be far from.
# Actually, let's just use the same set.

def _alignment_score(vector, n):
    """
    For each probe, compute the number of positions where vector matches the probe.
    High alignment = risky. We want LOW alignment on average.
    We return the negative of the average alignment, so higher is better.
    """
    if n == 0:
        return 0.0
    
    # Use only the first min(n, _PROBE_DIM) positions for alignment
    eff_n = min(n, _PROBE_DIM)
    if eff_n == 0:
        return 0.0
    
    total_matches = 0
    for probe in _PROBES:
        matches = 0
        for i in range(eff_n):
            if vector[i] == probe[i]:
                matches += 1
        total_matches += matches
    
    avg_matches = total_matches / _NUM_PROBES
    # Normalize: max matches is eff_n, so avg_matches is in [0, eff_n]
    # We want to penalize high matches. Return negative normalized matches.
    return -avg_matches / eff_n


def priority(vector, n):
    """
    Priority for a candidate vector in a greedy cap set construction.
    
    Uses:
    1. Entropy of coordinate distribution (prefer balanced vectors)
    2. Anti-alignment with fixed probe vectors (prefer vectors far from probes)
    3. A lightweight hash-based diversity term
    """
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT:
        # Emergency: return a neutral value to avoid timeout
        return 0.5
    
    if n == 0:
        return 1.0
    
    # Component 1: Entropy of coordinate distribution
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / n
            entropy -= p * math.log(p, 3)
    # entropy is in [0, 1], higher is better (more balanced)
    
    # Component 2: Anti-alignment with probes
    align_score = _alignment_score(vector, n)
    # align_score is negative (or zero), so -align_score is positive
    # Normalize to [0, 1] roughly
    anti_align = -align_score  # This is in [0, 1]
    
    # Component 3: Lightweight hash for diversity
    # Use a simple hash to break ties and add diversity
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 1)) & 0xFFFFFFFF
    hash_norm = h / 4294967295.0
    
    # Combine components
    # We want high entropy, high anti-alignment, and hash as a tie-breaker
    # The hash should have low weight
    score = 0.6 * entropy + 0.35 * anti_align + 0.05 * hash_norm
    
    return score
