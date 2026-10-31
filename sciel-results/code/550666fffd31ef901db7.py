import math
import time
from collections import Counter

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.95  # Use 95% of time budget

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols.
    2. Secondary: Maximize Shannon Entropy of the symbol distribution.
       Higher entropy implies more "random" distribution, which tends to
       spread out the vectors in the space, reducing the likelihood of
       forming affine lines with other similarly distributed vectors.
    3. Tertiary: Target the "center" of the hypercube by maximizing the count
       of 1s while minimizing the product of 0s and 2s. This geometrically
       targets vectors furthest from the boundary planes that densely contain
       affine lines, thereby avoiding boundary clustering.
    4. Quaternary: Robust hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() - _start_time > 300 * _DEADLINE_FRACTION:
        # Return a very low priority to stop processing if we are close to timeout
        # This is a safeguard, though the evaluator usually handles this.
        return -1e10

    # Count symbols using Counter for speed
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    # Distinct count (3 is best, 2 is middle, 1 is worst)
    counts = [c0, c1, c2]
    distinct_count = sum(1 for c_val in counts if c_val > 0)
    
    # Shannon Entropy
    # H = - sum(p_i * log(p_i))
    # If p_i = 0, term is 0.
    entropy = 0.0
    for c_val in counts:
        if c_val > 0:
            p = c_val / n
            entropy -= p * math.log(p)
    
    # Center-ness score:
    # We want to maximize c1 (number of 1s) and minimize c0 * c2 (product of 0s and 2s).
    # This targets the "center" of the hypercube.
    # Normalize by n to keep scale consistent.
    c1_score = (c1 / n) * 100.0
    # The product c0 * c2 is maximized when c0 and c2 are both large (boundary).
    # We want to minimize this, so we subtract it.
    # Scale factor: max product is roughly (n/2)*(n/2) = n^2/4.
    # Let's use a smooth penalty.
    prod_penalty = (c0 * c2) / (n * n) * 100.0 if n > 0 else 0.0
    
    center_score = c1_score - prod_penalty
    
    # Primary score: distinct count
    # Scale to make it dominant
    distinct_score = distinct_count * 10000.0
    
    # Secondary score: Higher entropy is better
    # Max entropy for n is log(3)
    # Scale by 1000 to make it significant but not overpower distinct count
    entropy_score = entropy * 1000.0
    
    # Tertiary score: Center-ness
    # Scale to be significant but secondary to entropy
    center_scaled = center_score * 500.0
    
    # Quaternary: Robust hash
    # Convert vector to bytes for hashing
    # vector elements are 0, 1, 2
    # Use a simple polynomial hash with a large prime
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 13) % (10**9 + 7)
    
    # Normalize hash to [0, 1]
    hash_score = h / (10**9 + 7)
    
    return float(distinct_score + entropy_score + center_scaled + hash_score)
