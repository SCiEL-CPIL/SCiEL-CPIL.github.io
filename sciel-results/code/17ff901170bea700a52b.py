import hashlib
import itertools
import time

# EVOLVE-BLOCK-START

# Global deadline guard: 330s wall-clock limit, use 320s as safety margin
_DEADLINE = time.time() + 320

# Cache for computed priorities
_priority_cache = {}

# Precomputed line partners for small n (n <= 5)
_line_partner_cache = {}

def _compute_line_partners(n):
    """
    Precompute for each vector v, the set of vectors w such that v, w, and some x
    form an affine line. Actually, for cap sets, a line is three distinct points a, b, c
    such that a + b + c = 0 mod 3 (coordinate-wise), which is equivalent to
    c = -a - b mod 3.
    
    So for a given pair (a, b), the third point is c = (-a_i - b_i) mod 3 for each i.
    For a given vector v, the vectors that conflict with v are those b such that
    there exists a in the current set with a + b + v = 0.
    
    But since we're doing static priority (sort-first), we can't use current set.
    Instead, we compute for each vector v, how many other vectors w would form a line
    with v and some third vector. This is a structural measure of "conflict potential".
    
    For a fixed v, the number of pairs (a, b) such that a + b + v = 0 is:
    For each a, b is determined: b = -a - v mod 3.
    So for each a, there's exactly one b. The number of valid (a, b) pairs is
    3^n (all possible a), but we need a != b != v and a, b, v all distinct.
    
    Actually, this is getting complicated. Let me think differently.
    
    A simpler structural measure: for each vector v, count how many vectors w
    are "close" to v in the sense that they would frequently form lines with
    common third vectors.
    
    Let's use: for each v, compute the number of distinct vectors u such that
    u + v = u + v mod 3 has some property... 
    
    Actually, let's use a different approach. For static priority, the best
    known heuristic is based on the "weight" or "density" of the vector.
    
    Key insight from cap set theory: vectors with balanced coordinates (roughly
    n/3 of each symbol) tend to be in larger cap sets. This is because they
    have more "room" to avoid forming lines.
    
    Let's combine:
    1. Balance score (primary)
    2. A structural conflict estimate (secondary)
    3. Hash tiebreaker (tertiary)
    
    For the structural conflict estimate, I'll use the following:
    For a vector v, define its "conflict radius" as the number of vectors w
    such that the line through v and w (i.e., {v, w, -v-w}) has all three
    points distinct. This is always true unless w = v or w = -2v.
    
    Actually, every pair (v, w) with v != w determines a unique third point
    -v-w. The line is valid (all distinct) iff v != w and -v-w != v and -v-w != w.
    -v-w != v iff -w != 2v iff w != -2v
    -v-w != w iff -v != 2w iff v != -2w
    
    So the line is invalid only in specific cases. Most pairs form valid lines.
    
    Let's use a different structural measure: the "self-intersection" number.
    For vector v, count how many u satisfy u + u + v = 0, i.e., u = -v/2.
    Since we're in mod 3, this is u = 2v (since 2*2v = 4v = v mod 3... wait).
    2u + v = 0 mod 3 => 2u = -v => u = 2*(-v) = -2v = v mod 3 (since -2 = 1 mod 3).
    So u = v. This means the only "self" line is trivial.
    
    Let me just use the balance score as primary and add a good hash tiebreaker.
    The key finding from the knowledge is that polynomial hash tiebreaker
    worked better than CRC32.
    """
    pass

def _poly_hash(vector, n):
    """
    Polynomial hash for tiebreaking. This is the one that worked well (score 1.246).
    """
    h = 0
    for x in vector:
        h = (h * 31 + x + 1) & 0xFFFFFFFF
    # Convert to a float in [0, 1)
    return h / 0xFFFFFFFF

def _balance_score(vector, n):
    """
    Score based on how balanced the symbol counts are.
    Higher = more balanced = better.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    ideal = n / 3.0
    
    # Squared deviation from ideal
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Normalize: max deviation is when all symbols are the same.
    # If n is divisible by 3: max dev = 2*(2n/3)^2 + (n/3)^2 = n^2
    # If not, it's close to n^2.
    # So normalized balance is in [0, 1], where 1 = perfectly balanced.
    max_dev = n * n
    if max_dev == 0:
        return 1.0
    return 1.0 - (dev / max_dev)

def _structural_score(vector, n):
    """
    Additional structural score based on the sum of coordinates.
    Vectors with sum close to n (the "center") might be better.
    """
    s = sum(vector)
    # Ideal sum is n (since n/3 of each 0,1,2 gives sum = n/3*1 + n/3*2 = n)
    dev = abs(s - n)
    # Max deviation is n (if all 0s, sum=0, dev=n; if all 2s, sum=2n, dev=n)
    if n == 0:
        return 1.0
    return 1.0 - (dev / n)

def priority(vector, n):
    """
    Returns a priority score for the vector. Higher = should be considered earlier.
    
    Strategy:
    1. Primary: Balance of symbol counts (0, 1, 2). Weight: 10.0
    2. Secondary: Structural sum balance. Weight: 1.0
    3. Tertiary: Polynomial hash tiebreaker. Weight: 0.01
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a neutral priority if we're running out of time
        return 0.0
    
    # Use caching
    key = (vector, n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    # Compute scores
    bal = _balance_score(vector, n)
    struct = _structural_score(vector, n)
    tie = _poly_hash(vector, n)
    
    # Combine: Balance is dominant
    score = bal * 10.0 + struct * 1.0 + tie * 0.01
    
    _priority_cache[key] = score
    return score

# EVOLVE-BLOCK-END
