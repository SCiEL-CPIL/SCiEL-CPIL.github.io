"""OneMax direct binary construction (100 bits) - NUMPY PRE-ALLOCATION approach."""
import time
import numpy as np

N = 100
DEADLINE_SECONDS = 150.0

# EVOLVE-BLOCK-START
# Pre-allocate the optimal solution at module load time.
# This shifts the computational burden to import time.
_ONES_ARRAY = np.ones(N, dtype=np.uint8)

def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Strategy: NumPy Pre-allocation.
    The optimal solution for OneMax is a string of all '1's. 
    We pre-allocate a NumPy array of ones at module load time.
    Upon invocation, we convert this array to a string.
    A deadline guard is included to ensure the function returns 
    the best available result (the pre-allocated one) if the 
    time limit is approached, although the operation is O(N) 
    and instantaneous.
    """
    start_time = time.time()
    
    # Deadline guard: If we are already very close to the limit, 
    # we should return immediately. However, the conversion is so fast 
    # that this is mostly a structural requirement.
    if time.time() - start_time > DEADLINE_SECONDS:
        # Fallback, though practically unreachable
        return "1" * N

    # Convert the pre-allocated NumPy array to a string of '1's.
    # np.array2string with separator='' gives '1 1 1...', so we strip spaces.
    # Alternatively, we can just use the array to construct the string.
    # Actually, '1' * N is likely faster in CPython than numpy conversion.
    # But the directive asks for numpy pre-allocation. 
    # Let's stick to the directive's spirit: use the pre-allocated data.
    # We can map the array to characters.
    
    # Strategy: Model the binary string as a 128-bit integer.
    # Create a 128-bit integer with all bits set to 1.
    # x = (1 << 128) - 1
    # Extract the rightmost 100 bits using bin(x)[-100:].
    # This leverages Python's native big-integer arithmetic.
    
    # Pre-compute the integer at module level to minimize construct() overhead?
    # The directive says "generate ... in a single low-level operation".
    # Let's do it inside construct to strictly follow "upon invocation" if needed,
    # but pre-computing is safer for speed.
    # However, the prompt says "Model ... by shifting ... then extract".
    # Let's pre-compute the final string from the integer at module load time
    # to ensure construct() is as fast as possible, or compute it inside?
    # "bypassing string multiplication overhead entirely" suggests the generation method.
    # Let's pre-compute the string derived from the integer.
    
    # To adhere to the "upon invocation" spirit of the previous design while using the new method:
    # We can pre-compute the integer and the string derived from it.
    
    # Let's update the module-level pre-allocation to use this integer method.
    # But I can only edit the SEARCH block.
    # I will replace the return statement with the integer-based extraction.
    # To make it fast, I should probably pre-compute the string at module level
    # using this method, but I can't easily change the module-level code in this single edit
    # unless I expand the search.
    
    # Let's just do the calculation in construct. It's O(1) for small ints, but 128-bit is still very fast.
    # Actually, `bin()` creates a string of length 130. Slicing is O(100).
    # `"1" * 100` is also O(100).
    # The directive asks to try this specific strategy.
    
    # Let's implement it directly.
    mask = (1 << 128) - 1
    # bin(mask) is '0b' followed by 128 '1's
    # bin(mask)[-100:] gives the last 100 '1's
    return bin(mask)[-100:]
# EVOLVE-BLOCK-END
