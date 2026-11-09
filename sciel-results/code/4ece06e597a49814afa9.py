"""Program for OneMax direct binary construction (100 bits)."""
import time
import numpy as np

N = 100
DEADLINE = 150.0
SAFETY_MARGIN = 1.0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The direct, optimal construction is to set all bits to '1'.
    This achieves the maximum possible score of N (100 ones).
    
    We use a model-and-optimize framework:
    1. Define the objective: maximize sum(bits).
    2. Parameterize the solution: a bit vector of length N.
    3. Optimize: The gradient of the objective with respect to each bit is +1.
       Therefore, the unique global maximum is the vector of all 1s.
       
    Implementation detail:
    We represent the vector as a NumPy array of uint8 initialized to 1, 
    then cast it to a string using np.char.mod to leverage vectorized operations.
    """
    start_time = time.time()
    
    # Theoretical optimum for OneMax is all bits set to 1.
    # We construct this directly using NumPy as per directive.
    bits = np.ones(N, dtype=np.uint8)
    
    # Convert array to string of '1's
    # np.char.mod('1', arr) formats each element as a string '1'
    candidate = ''.join(np.char.mod('1', bits))
    
    # Verify objective to satisfy 'fit or optimize' directive structure
    # (Though for direct construction, the fit is trivially perfect)
    ones_count = candidate.count('1')
    
    # Deadline guard (not expected to trigger, but required)
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        # If we were in a loop, we would break here. 
        # Since we computed instantly, we just return.
        pass
        
    return candidate
# EVOLVE-BLOCK-END
