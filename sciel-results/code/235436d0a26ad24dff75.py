"""
OneMax 100-bit direct construction.
Returns the global optimum: a 100-character string of '1's.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Return the optimal 100-bit string for OneMax: all bits set to 1.
    The global optimum for OneMax is trivially all ones.
    
    This is a direct, non-evolutionary approach based on the task structure.
    The operation is O(1) in logical complexity (string multiplication).
    """
    # Track start time for the explicit deadline guard required by the directive.
    deadline_start = time.monotonic()
    
    # Direct construction of the global optimum: N ones.
    # This is an O(1) operation in terms of logical complexity (string multiplication).
    best_candidate = "1" * N
    
    # Explicit deadline guard: check elapsed time against a safety margin.
    # The task limit is 150s. We use a 5s margin, so 145s.
    # This check is defensive; the operation above is instantaneous.
    if time.monotonic() - deadline_start > 145:
        return best_candidate
    
    return best_candidate
# EVOLVE-BLOCK-END
