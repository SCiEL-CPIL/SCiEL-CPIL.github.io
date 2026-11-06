import time
import zlib

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill

# Precomputed weights for a small "structural" signature
# We use a weighted sum of positions to create a deterministic, fast hash-like value
# that correlates with vector structure but is much faster than full CRC32 on bytes
# or complex autocorrelation loops.

def _fast_structural_score(vector, n):
    """
    A fast deterministic tie-breaker that captures some structural properties.
    Uses a weighted sum with prime coefficients to avoid simple pattern matching.
    """
    # Use a simple polynomial hash with a large prime modulus
    # This is faster than CRC32 for small n and avoids byte conversion overhead
    h = 0
    # Prime coefficients for positions to break symmetry
    # We only need a few terms to be discriminative
    for i, v in enumerate(vector):
        # Use a mix of position and value
        h = (h * 31 + v * (i + 1)) & 0x7FFFFFFF
    return h / 0x7FFFFFFF

def priority(vector, n):
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a neutral value if we're out of time
        # This shouldn't happen often, but ensures we don't crash
        return 0.0

    # 1. Global Balance Metric (Primary)
    # Count occurrences of 0, 1, 2
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    target = n / 3.0
    
    # Standard deviation from target balance
    dev = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    
    # 2. Structural Refinement (Secondary)
    # Instead of expensive autocorrelation, we use a "transition" metric
    # which is O(n) but very fast in Python if optimized, or we can use
    # a simpler structural hash.
    # Let's try a combination:
    # a) Sum mod 3 bonus (known to help in cap sets)
    s = sum(vector) % 3
    sum_bonus = {1: 2.0, 2: 1.0, 0: 0.0}[s]
    
    # b) Fast structural hash for tie-breaking
    h_norm = _fast_structural_score(vector, n)
    
    # 3. Combine
    # Primary: -dev (want balanced vectors)
    # Secondary: sum_bonus (prefer sum=1 mod 3)
    # Tertiary: h_norm (deterministic tie-breaker)
    
    # Scale factors:
    # dev is roughly O(n^2). For n=10, dev max ~ 200.
    # sum_bonus is 0, 1, or 2.
    # We want sum_bonus to be a small perturbation of the balance score.
    # If two vectors have very similar balance, sum_bonus breaks the tie.
    # Let's use a weight of 0.01 for sum_bonus relative to dev.
    # But dev is large. Let's normalize dev?
    # Actually, the existing programs use -dev directly.
    # Let's stick to the proven structure but replace the slow autocorrelation
    # with a faster, more structural tie-breaker.
    
    # The inspiration 1 (score 1.2383) used:
    # return -dev + sum_bonus * 0.01 + 1e-9 * h_norm
    # The current program (score 1.1680) added autocorrelation which might be
    # too heavy or not as effective.
    
    # Let's try a hybrid: keep the balance primary, but use a slightly
    # more sophisticated tie-breaker than just a simple hash.
    # We'll use a "pairwise interaction" sum that is cheaper than full autocorrelation.
    # Actually, the simple hash in inspiration 1 was quite good.
    # Let's refine the hash to be more sensitive to local structure.
    
    # New Structural Metric: Sum of adjacent differences squared?
    # No, that's similar to autocorrelation.
    
    # Let's go with the high-performing inspiration 1 structure but with
    # a slightly better tie-breaker.
    
    # Tie-breaker: CRC32 is very fast and deterministic.
    # Let's use it if n is small enough to convert to bytes quickly.
    # For n <= 64, bytes(vector) is fast.
    if n <= 128:
        try:
            b = bytes(vector)
            h = zlib.crc32(b)
            h_norm = h / 0xFFFFFFFF
        except:
            h_norm = _fast_structural_score(vector, n)
    else:
        h_norm = _fast_structural_score(vector, n)
    
    # Final combination
    # Weighting:
    # -dev is the main driver.
    # sum_bonus * 0.05 (slightly higher weight than 0.01 to make it more effective?)
    # Let's stick to 0.01 to be safe, as 1.2383 used it.
    
    score = -dev + sum_bonus * 0.01 + 1e-9 * h_norm
    
    return score
