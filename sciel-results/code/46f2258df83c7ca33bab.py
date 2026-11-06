"""Direct OneMax construction: return all-ones string of length N."""
import time

N = 100

# EVOLVE-BLOCK-START
# Pre-allocate the optimal string at module load time.
# The optimal OneMax solution is the all-ones string.
# We use string multiplication which is O(N) and highly optimized in CPython.
# The alternative big-integer approach (shifting/masking) involves bin() conversion
# which is also O(N) but with higher constant overhead due to base conversion.
# Thus, we stick to the direct string construction as the most efficient method.
_OPTIMAL = "1" * N

DEADLINE_SECONDS = 150.0
SAFETY_MARGIN_SECONDS = 5.0

def construct():
    """Return a 100-bit binary string maximizing the number of ones."""
    t0 = time.time()
    
    # The optimal solution for OneMax is trivially the all-ones string.
    # No search, optimization, or iterative refinement can improve upon this.
    # The precomputed constant is returned directly.
    result = _OPTIMAL
    
    # Deadline guard: ensure we return well before the 150 s kill limit.
    # Although this function is O(1) (just returning a reference), we maintain
    # the guard structure as per the task directive.
    elapsed = time.time() - t0
    if elapsed + SAFETY_MARGIN_SECONDS >= DEADLINE_SECONDS:
        return result
        
    return result
# EVOLVE-BLOCK-END
