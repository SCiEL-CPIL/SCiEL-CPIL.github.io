"""OneMax 100-bit direct construction - maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    t0 = time.monotonic()
    limit = 145.0

    # Direct construction: for OneMax the global optimum is the all-ones genotype.
    # Use bytearray for efficient mutable buffer allocation before decoding to string.
    # This leverages C-level memory operations for initialization.
    if time.monotonic() - t0 > limit:
        return '1' * N

    buffer = bytearray(b'1' * N)
    result = buffer.decode('ascii')
    
    return result
# EVOLVE-BLOCK-END
