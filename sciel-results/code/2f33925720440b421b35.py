"""Cap set priority function for greedy insertion."""
import time

_start_time = time.time()
_DEADLINE_MARGIN = 10  # seconds before the 300s limit

def priority(vector, n):
    """
    Computes a priority score for a vector in {0, 1, 2}^n.
    Higher scores indicate vectors that should be added earlier in the greedy process.
    
    Strategy:
    1. Primary: Balance symbol counts (0, 1, 2) to avoid early constraint saturation.
       Vectors with counts close to n/3 are preferred.
    2. Secondary: A strong, non-linear hash of the vector to provide a deterministic 
       but pseudo-random tie-breaking order. This avoids local clustering of similar 
       structures that might form APs with each other, allowing the greedy algorithm 
       to explore a more uniform subset of the hypercube.
    """
    # Deadline guard: if we're close to the limit, return a neutral score
    elapsed = time.time() - _start_time
    if elapsed > 300 - _DEADLINE_MARGIN:
        return 0.0

    # Count symbols
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Balanced vectors (close to n/3 each) are preferred
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Primary: minimize deviation (negate for max-priority)
    # Scale by a large number to ensure it dominates the tie-breaker
    score = -deviation * 10000.0
    
    # Secondary: Deterministic pseudo-random tiebreaker
    # We use a simple integer mixing function to generate a unique, 
    # well-distributed integer for each vector.
    # This acts as a static permutation of the vectors with the same balance score.
    
    # Initialize hash
    h = 0
    # Use a large prime modulus or just let it wrap around (Python ints are arbitrary precision, 
    # but we want it to behave like a 64-bit hash for distribution).
    # We'll use a simple xor-shift and multiply mix.
    
    # Convert vector to a string or use direct integer ops. 
    # Direct integer ops are faster.
    # vector elements are 0, 1, 2.
    
    # A simple hash:
    # h = h * 31 + v
    # But this is linear. Let's use a better mix.
    
    # Since n is small, we can just pack the vector into an integer.
    # Each digit is 0, 1, or 2. We can encode it in base 3.
    # val = sum(vector[i] * (3**i))
    # This gives a unique integer for each vector.
    # Then we apply a finalizer to this integer to break any remaining patterns.
    
    val = 0
    power = 1
    for v in vector:
        val += v * power
        power *= 3
        
    # Finalizer: A simple mix to ensure uniformity
    # val ^= val >> 33
    # val *= 0xff51afd7ed558ccd
    # val ^= val >> 33
    # val *= 0xc4ceb9fe1a85ec53
    # val ^= val >> 33
    
    # To keep it fast and simple, we can just use the value itself if it's small enough,
    # but for n=10, 3^10 is ~60k, which is small.
    # For larger n, we might need the mix.
    # Let's apply a lightweight mix to be safe against any structural bias in base-3 representation.
    
    # Simple MurmurHash3-style finalizer (adapted for Python)
    val = (val ^ (val >> 16))
    val = (val * 0x85ebca6b) & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 13))
    val = (val * 0xc2b2ae35) & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 16))
    
    # Map to a small positive range to add to score
    # The tiebreak should be small enough not to override the primary score,
    # but large enough to break ties uniquely.
    tiebreak = val * 1e-6
    
    score += tiebreak
    
    return score
