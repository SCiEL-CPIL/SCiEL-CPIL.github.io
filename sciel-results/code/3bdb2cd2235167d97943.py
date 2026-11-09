import time
import math
import random
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 280

# Precompute Fourier basis vectors for the lowest non-trivial frequency
# For F_3^n, the lowest non-trivial character is chi(v) = omega^(sum v_j) where omega = e^(2*pi*i/3)
# The magnitude of the Fourier transform at this frequency is |sum_j v_j| mod 3 behavior
# Actually, for a single vector v, the "Fourier magnitude" at frequency k is related to
# how the vector projects onto the subspace orthogonal to the all-ones direction.
# We use the sum of coordinates mod 3 as a proxy for the lowest non-trivial frequency component.

# We'll maintain a lightweight LSH-style conflict density tracker
# The key insight: for cap sets in F_3^n, three points a, b, c form an AP
# iff a + c = 2b (mod 3), i.e., a + c = -b (mod 3) since 2 = -1 mod 3.
# So a new vector v conflicts with existing {a, b} if v = -a - b (mod 3) for some a, b in set.
# We can't check all pairs, but we can use bucket-based approximation.

# LSH configuration: multiple hash functions mapping vectors to small buckets
# A vector "collides" in a bucket if it shares that bucket with existing vectors.
# More collisions => higher "local density" => more likely to create APs => lower priority.

# We use a fixed set of hash functions. Each hash function maps a vector to a bucket index.
# We track counts per bucket. The conflict density of a new vector is the sum of counts
# in the buckets it maps to (minus 1 for itself if already counted, but we query before adding).

# Hash function design: for F_3^n, we can use linear projections.
# h_i(v) = sum_j v_j * w_{i,j} mod B, where w_{i,j} are random weights in {0, 1, 2}
# and B is a small bucket count.

# We'll use a fixed seed for reproducibility within a run, but the weights are fixed
# at module load time.

_NUM_HASHES = 16
_BUCKET_SIZE = 64  # Each hash function maps to one of _BUCKET_SIZE buckets

# Generate fixed random weights for the hash functions
_random_state = random.Random(42)
_HASH_WEIGHTS = []
for h in range(_NUM_HASHES):
    weights = [_random_state.randint(0, 2) for _ in range(64)]  # Support up to n=64
    _HASH_WEIGHTS.append(weights)

# Bucket counts: _NUM_HASHES lists, each of length _BUCKET_SIZE
_bucket_counts = [[0] * _BUCKET_SIZE for _ in range(_NUM_HASHES)]

# Track the total number of vectors added so far
_total_vectors = 0

# We need to be careful: the greedy algorithm calls priority(vector, n) for each candidate.
# We don't know when a vector is actually added to the set. But in a typical greedy cap set
# algorithm, the evaluator adds vectors in descending priority order, checking each one
# against the current set. The priority function is called for every candidate vector,
# but only those that don't create an AP are added.

# The problem: we don't have access to the current set from within priority().
# So we need a different approach.

# Alternative: Instead of tracking actual conflicts, we use the LSH signature as a
# static property of the vector that correlates with how "central" or "dense" it is
# in the vector space. Vectors that map to the same buckets as many other vectors
# (in a uniform distribution sense) are more likely to be in dense regions.

# Actually, let me reconsider. The greedy algorithm works like this:
# 1. Compute priority for all 3^n vectors
# 2. Sort by priority descending
# 3. Iterate through sorted list, adding each vector if it doesn't create an AP with existing vectors

# So priority() is called once per vector, in some order. The order is determined by the
# priority values. We can't know which vectors have been added when priority() is called
# for a given vector, because the sorting happens after all priorities are computed.

# Wait, that's not right either. In a typical implementation, the evaluator might:
# - For each vector, compute its priority
# - Sort all vectors by priority
# - Then greedily add them

# In that case, priority() is called for each vector exactly once, and we don't know
# the current set state. So LSH-based dynamic conflict tracking won't work directly.

# Let me think of a different approach within the same family (time-varying hash-based).

# The current approach uses time-varying hashes to create different orderings over time.
# But since all priorities are computed at once (before sorting), the "time" variation
# only matters if the evaluator calls priority() at different times for different vectors.

# Actually, looking at the current code, it uses time.time() inside priority(). If the
# evaluator computes all priorities in a loop, the time will vary slightly between calls.
# But the phase changes every 20 seconds, so within a single evaluation (which should
# take less than 300s), the phase might change a few times.

# Let me try a different strategy: use a more structured hash that has better
# properties for cap set exploration.

# Key insight for cap sets: vectors with more balanced symbol counts (roughly n/3 of each)
# tend to be "central" and more likely to be in cap sets. But the current code already
# has a balance penalty.

# Let me try: use a hash that emphasizes the "structure" of the vector in a way that
# correlates with cap set membership. One idea: use the value of the vector as a number
# in base 3, and use properties of that number (like digit sum, or modular properties).

# Actually, let me try a completely different approach within the hash family:
# Use multiple independent hash functions, and use the minimum (or a soft-min) of the
# hash values as the priority. This creates a "bottleneck" effect where a vector needs
# to be good across all hash functions, leading to more diverse exploration.

# Or: use the maximum instead, so that a vector only needs to be good in one hash function.

# Let me try: priority = -soft_min of hash values, where soft_min is log-sum-exp based.
# This gives a smooth approximation to the minimum, which is the "worst" hash value.
# Vectors that are bad in at least one hash function get lower priority.

# Actually, for a cap set, we want vectors that are "far" from each other. A good
# heuristic might be to use a hash that measures how "spread out" the vector is.

# Let me try a new approach: use the L2 norm of the vector (treating 0,1,2 as values)
# combined with a hash for diversity.

# Vectors with values close to 1 (the mean) have lower norm deviation, which might
# correlate with being in the "center" of the cube, which is where cap sets tend to be.

# Let me combine:
# 1. A balance score (existing)
# 2. A time-varying hash for diversity
# 3. A locality-sensitive component using fixed hash functions

# For the LSH component, since we can't track dynamic counts, we'll use a static
# property: the number of "nearby" vectors in a theoretical uniform distribution.
# This is just a constant, so it won't help.

# Let me try yet another idea: use the hash to create a "virtual neighborhood" score.
# For each hash function, compute the bucket. The "density" of that bucket in a
# uniform distribution is 1/_BUCKET_SIZE. But that's constant.

# I think the key improvement is to use a better combination of hash functions
# and to tune the balance penalty.

# Let me try: use 8 hash functions, and take the average. Add a stronger balance
# preference. Also, add a component that favors vectors with a specific pattern
# (e.g., vectors where the sum of coordinates mod 3 is 0, which is a known
# structure in cap sets).

# Actually, let me look at this from a different angle. The best known cap sets
# in F_3^n have size roughly (2.756)^n. The greedy algorithm with a good
# priority function can achieve close to this.

# A known good heuristic for cap sets is to prefer vectors that are "far" from
# the existing set. Since we can't track the existing set, we need a proxy.

# One proxy: vectors that are "generic" (not lying in any low-dimensional
# subspace) are more likely to be addable. This correlates with having a
# diverse set of coordinates.

# Let me try: use the entropy of the coordinate distribution as a priority
# component. High entropy means the coordinates are well-distributed.

def _coord_entropy(vector, n):
    """Compute the entropy of the coordinate value distribution."""
    if n == 0:
        return 0.0
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / n
            entropy -= p * math.log(p, 3)  # log base 3 for normalization
    return entropy

def _fourier_low_freq_score(vector, n):
    """
    Compute a score based on the negative magnitude of the Fourier transform
    at the lowest non-trivial frequency.
    
    For F_3^n, the lowest non-trivial character is omega^(sum v_j) where omega = e^(2*pi*i/3).
    The magnitude of this character is always 1, but the "alignment" with the additive
    identity (zero vector) is measured by how close sum(v_j) is to 0 mod 3.
    
    Vectors with sum(v_j) mod 3 == 0 are "aligned" with the identity and thus more
    likely to form APs with other vectors. We want to penalize these.
    
    We return a score that is high when sum(v_j) mod 3 is far from 0 (i.e., 1 or 2),
    which correlates with being less likely to complete a 3-term AP.
    """
    if n == 0:
        return 0.0
    s = sum(vector) % 3
    # s == 0: aligned with identity, bad (return low score)
    # s == 1 or s == 2: misaligned, good (return high score)
    # Use a smooth mapping: score = 1 - (1 if s == 0 else 0)
    # But to make it smoother and differentiate between 1 and 2, use:
    # score = 0 if s == 0, 1 if s == 1, 1 if s == 2
    # Actually, let's use a continuous proxy: the distance from 0 in the cyclic group
    # dist = min(s, 3-s) so dist is 0, 1, 1 for s=0,1,2
    # score = dist / 1.0 = dist
    dist = min(s, 3 - s)
    return float(dist)

def _zero_spread_score(vector, n):
    """
    Measure how spread out the zero positions are.
    Zeros that are isolated (not adjacent to other zeros in a structured way)
    are less likely to participate in APs.
    
    We compute the sum of squared distances between consecutive zero positions
    (cyclically). Higher spread = better.
    """
    if n == 0:
        return 0.0
    zero_pos = [i for i, v in enumerate(vector) if v == 0]
    if len(zero_pos) < 2:
        return 0.0
    # Sort zero positions
    zero_pos.sort()
    # Compute cyclic distances between consecutive zeros
    m = len(zero_pos)
    spread = 0.0
    for i in range(m):
        next_i = (i + 1) % m
        d = abs(zero_pos[next_i] - zero_pos[i])
        # Cyclic distance
        d = min(d, n - d)
        spread += d * d
    # Normalize by n^2 to get a value in [0, 1]
    return spread / (n * n + 1e-9)

def priority(vector, n):
    """
    Time-varying hash-based priority with entropy, balance, Fourier, and zero-spread components.
    
    Uses multiple hash functions combined via soft-min to create diverse
    exploration, with entropy, balance, Fourier low-frequency, and zero-spread
    as structural priors.
    """
    elapsed = time.time() - _START_TIME
    
    # Phase-based strategy
    phase = int(elapsed / 15)  # New strategy every 15 seconds
    
    # Multiple hash functions
    h1 = 0
    for v in vector:
        h1 = (h1 * 31 + v) & 0xFFFFFFFF
    
    h2 = 0
    for i, v in enumerate(vector):
        h2 ^= (v * (i + 1 + phase))
    h2 = h2 & 0xFFFFFFFF
    
    h3 = 0
    for i, v in enumerate(vector):
        h3 += v * (i + 1) * (phase + 1)
    h3 = h3 & 0xFFFFFFFF
    
    h4 = 0
    for i, v in enumerate(vector):
        h4 = (h4 * 17 + v * (i % 7 + 1)) & 0xFFFFFFFF
    
    # Normalize hashes to [0, 1]
    norm = 4294967295.0
    h1n = h1 / norm
    h2n = h2 / norm
    h3n = h3 / norm
    h4n = h4 / norm
    
    # Soft-min: log-sum-exp of negative values
    # soft_min(h1n, h2n, h3n, h4n) = -log(sum(exp(-h_i)))
    # This is in [0, 1] and approximates the minimum
    max_neg = max(-h1n, -h2n, -h3n, -h4n)
    sum_exp = math.exp(-h1n - max_neg) + math.exp(-h2n - max_neg) + math.exp(-h3n - max_neg) + math.exp(-h4n - max_neg)
    soft_min_val = -math.log(sum_exp) - max_neg
    
    # Balance component
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    balance_penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Entropy component
    entropy = _coord_entropy(vector, n)
    
    # Fourier low-frequency score
    fourier_score = _fourier_low_freq_score(vector, n)
    
    # Zero spread score
    zero_spread = _zero_spread_score(vector, n)
    
    # Combine: soft-min hash drives diversity, entropy and balance are structural priors,
    # Fourier score penalizes identity-aligned vectors, zero spread rewards isolated zeros
    # We want high soft_min, high entropy, low balance penalty, high fourier_score, high zero_spread
    score = 0.35 * soft_min_val + 0.2 * entropy - 0.15 * balance_penalty / (n + 1) + 0.2 * fourier_score + 0.1 * zero_spread
    
    return score
