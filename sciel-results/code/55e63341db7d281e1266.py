import zlib
import time

_START = time.time()
_DEADLINE = _START + 280.0  # safety margin before 330s kill

def priority(vector, n):
    """
    Structural priority for cap set construction.
    
    Key insights:
    - A vector with many 1s is more likely to be the middle of a 3-AP
      (since for a,b,c with a+b+c≡0 mod 3, if b has coordinate 1,
      then a and c coordinates sum to 2 mod 3, i.e., {0,2} or {2,0})
    - Vectors that are "extremal" (mostly 0s and 2s, few 1s) are safer
    - Among safe vectors, prefer those with a specific pattern that
      minimizes pairwise conflict potential
    
    Strategy:
    1. Primary: count of 1s (fewer is better)
    2. Secondary: count of 0s (more 0s means more "room" for other vectors)
    3. Tertiary: deterministic hash for tie-breaking
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a low value to stop adding more vectors
        return 0.0
    
    # Count occurrences
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Primary signal: fewer 1s is better (max priority when c1=0)
    # Scale: c1=0 -> 1.0, c1=n -> 0.0
    one_score = 1.0 - (c1 / n) if n > 0 else 1.0
    
    # Secondary signal: prefer vectors with more 0s (they're "lighter")
    # but also consider that all-0 vector is special (it's the identity)
    # Actually, in cap sets, the all-zero vector is fine, but let's think...
    # Vectors with a mix of 0 and 2 (no 1s) form the "binary" cap set structure
    zero_score = c0 / n if n > 0 else 0.0
    two_score = c2 / n if n > 0 else 0.0
    
    # Tertiary: deterministic hash for stable tie-breaking
    data = bytes(vector)
    crc = zlib.crc32(data)
    hash_score = float(crc) / 0xFFFFFFFF
    
    # Combined score: weight primary heavily, then secondary, then hash
    # The all-1s vector is worst, all-0s or all-2s are good starting points
    # But we need diversity, so use a weighted combination
    score = 0.6 * one_score + 0.2 * (zero_score + two_score) + 0.2 * hash_score
    
    return score
