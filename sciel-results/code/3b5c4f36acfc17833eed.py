import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is a string of all 1s.
    We return this directly to ensure the maximum possible score.
    """
    start_time = time.time()
    deadline = 150.0
    
    # The optimal solution is trivial: all bits are 1.
    # This takes O(N) time to construct the string, which is negligible.
    result = "1" * N
    
    # Safety check (not strictly needed for this O(N) operation, but included per directive)
    if time.time() > deadline:
        # Fallback (should never happen for N=100)
        return "1" * N
        
    return result
# EVOLVE-BLOCK-END
