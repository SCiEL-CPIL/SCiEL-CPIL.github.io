"""OneMax direct binary construction (100 bits)."""
import time
import numpy as np

N = 100

# Pre-allocate the optimal solution as a NumPy array of ones at module load time.
# This shifts the memory allocation and initialization cost to import time.
_OPTIMAL_ARRAY = np.ones(N, dtype=np.uint8)

# Convert to a list of characters '1' once, as NumPy tolist is efficient.
# Alternatively, we can just keep the string if conversion is the bottleneck, 
# but the directive asks for NumPy array representation.
# Let's stick to the directive: "Represent the solution as a pre-allocated NumPy array... converting it to a string only upon invocation".
# Actually, converting a NumPy array to a string every time might be slower than just returning a pre-allocated string.
# But the directive is specific: "convert it to a string only upon invocation".
# Let's follow the directive.

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: Track elapsed time. 
    # Since this is a direct construction, it should be extremely fast.
    # We include the guard as requested by the directive.
    start_time = time.time()
    deadline = 150.0 # Seconds
    
    # The operation is O(N) for string conversion, which is ~100 ops.
    # It will finish in microseconds.
    
    # Convert the pre-allocated NumPy array to a string.
    # np.array([1,1,1]) -> [1, 1, 1] -> "111"
    # A common way is ''.join(map(str, arr)) or arr.astype(str).join('')
    # For small N, simple list comprehension or join is fine.
    
    # Check deadline (formality, as this is instant)
    if time.time() > deadline:
        # Fallback, though practically unreachable
        return "1" * N
        
    return _OPTIMAL_ARRAY.astype(str).join('')
# EVOLVE-BLOCK-END
