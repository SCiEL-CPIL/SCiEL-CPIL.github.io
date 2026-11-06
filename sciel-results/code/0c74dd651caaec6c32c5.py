"""
Direct construction for OneMax (100-bit).
The optimal solution is a string of 100 '1' characters.
We build it explicitly and return it immediately.

Strategy: Analytical Constructive Optimization with bytes.translate refinement.
The OneMax objective is separable and linear. The unique global maximizer is
the all-ones string. We construct this directly in O(N) time using a bytes
translation operation, which bypasses Python-level string multiplication
overhead, while retaining an explicit deadline guard for safety.
"""
import time

N = 100

# Pre-compute the translation table at module level to avoid per-call overhead.
# bytes.maketrans(b'\x00', b'1') creates a 256-byte table where index 0 maps to ord('1')=49
# and all other indices map to themselves.
_TRANSLATE_TABLE = bytes.maketrans(b'\x00', b'1')

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    This is the direct, optimal solution for the OneMax problem.
    
    Uses bytes.translate for construction:
    1. Create a 100-byte zero buffer.
    2. Translate all bytes using the pre-computed table (0 -> '1').
    3. Decode to ASCII string.
    
    This is deterministic, O(N), and avoids string multiplication.
    """
    t0 = time.monotonic()
    wall_limit = 150.0
    margin = 5.0

    # Direct construction via bytes translation
    # bytes(100) creates a 100-byte sequence of zeros.
    # .translate(_TRANSLATE_TABLE) maps each 0x00 to 0x31 ('1').
    # .decode('ascii') converts to a Python str.
    result = bytes(N).translate(_TRANSLATE_TABLE).decode('ascii')

    # Explicit deadline guard
    elapsed = time.monotonic() - t0
    if elapsed + margin < wall_limit:
        return result
    else:
        # In the extremely unlikely event of timeout, return the best found (which is the same)
        return result
# EVOLVE-BLOCK-END
