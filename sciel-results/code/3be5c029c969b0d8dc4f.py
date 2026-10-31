import time
import heapq
from itertools import product

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

# Precomputed primes for fast hashing
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
           73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
           157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
           239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
           331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
           421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503]

def _fast_hash(vector, n, seed=0):
    """Fast deterministic pseudo-random priority via linear prime-based hash."""
    h = 0
    for i, v in enumerate(vector):
        if i < len(_PRIMES):
            h += v * _PRIMES[i]
        else:
            h += v * (i * 31 + seed + 17)
    # Mix with seed and reduce to [0, 1)
    h = (h * 1000003 + seed * 999983) % (1 << 60)
    return h / (1 << 60)

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Geometric Centrality. Prioritize vectors close to the all-1s vector.
       This avoids the "boundary" regions of the hypercube which contain dense affine lines.
       Score is driven by the count of 1s, penalized by the presence of 0s and 2s.
    2. Secondary: Product penalty for mixed boundary symbols (0 and 2).
       Penalizes vectors that have both 0s and 2s, favoring "purer" deviations from the center.
    3. Tertiary: Fast structural hash for deterministic tie-breaking.
    
    This structural change shifts the search from "balance" (uniform distribution)
    to "centrality" (proximity to the geometric center of the hypercube).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # 1. Centrality Score:
    # Maximize c1 (distance from boundary in terms of 1s)
    # The term (c0 + c2) is the Hamming distance from the all-1s vector.
    # We want to minimize this distance.
    # Score component: c1 * 100 - (c0 + c2) * 10
    centrality_score = c1 * 100.0 - (c0 + c2) * 10.0
    
    # 2. Boundary Mixing Penalty:
    # The prompt suggests minimizing the product of 0s and 2s.
    # Vectors with both 0s and 2s are "mixed boundary" vectors.
    # Vectors with only 0s (or only 2s) are "pure boundary" vectors.
    # Pure boundary vectors might be safer than mixed ones in some contexts,
    # or mixed ones might create more APs. Let's penalize the product.
    mix_penalty = c0 * c2 * 0.5
    
    # 3. Distinct Symbol Bonus (retained from current for diversity):
    # Ensure we don't just pick all-1s (which is impossible as it's one vector) 
    # but we want vectors that utilize the space.
    # However, the centrality score already handles this.
    # A small bonus for having all 3 symbols can help break ties in a way that
    # explores the space more effectively than pure centrality.
    presence0 = 1.0 if c0 > 0 else 0.0
    presence1 = 1.0 if c1 > 0 else 0.0
    presence2 = 1.0 if c2 > 0 else 0.0
    num_distinct = presence0 + presence1 + presence2
    distinct_score = 1.0 * num_distinct  # Small weight, just for tie-breaking nuance
    
    # 4. Fast Hash for deterministic tie-breaking
    h = _fast_hash(vector, n, seed=1)
    hash_score = h * 0.1  # Very small weight, only for exact ties
    
    return float(centrality_score - mix_penalty + distinct_score + hash_score)
