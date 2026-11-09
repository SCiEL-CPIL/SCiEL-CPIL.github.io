"""Optimized OneMax solver: returns the global optimum (all 1s) directly."""
import time

N = 100
DEADLINE_LIMIT = 145  # Safety margin from 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    OneMax (100 bits): The global optimum is a string of 100 '1's.
    
    Model: b ∈ {0,1}^N, maximize J = Σ b_i.
    The separable linear objective achieves its maximum when every b_i = 1.
    
    This implementation returns the optimal solution directly.
    """
    start_time = time.time()
    
    # The global optimum for OneMax is all ones.
    result = '1' * N
    
    # Deadline guard (trivially satisfied for O(N) string construction)
    if time.time() - start_time > DEADLINE_LIMIT:
        return '0' * N
        
    return result
# EVOLVE-BLOCK-END
