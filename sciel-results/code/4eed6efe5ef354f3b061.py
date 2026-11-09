"""OneMax direct binary construction (100 bits) - Optimized NumPy Pathway"""
import time
import numpy as np

N = 100
START_TIME = time.time()
DEADLINE_LIMIT = 145.0  # 150s limit with 5s margin

# EVOLVE-BLOCK-START
def construct():
    # Check deadline explicitly to ensure we return before evaluator kills the process
    if time.time() - START_TIME > DEADLINE_LIMIT:
        return "1" * N

    try:
        # Strategy: Use NumPy to construct the byte sequence directly.
        # ASCII code for '1' is 49.
        # This avoids string formatting overhead by operating on raw bytes.
        byte_arr = np.full(N, 49, dtype=np.uint8)
        return byte_arr.tobytes().decode('ascii')
    except Exception:
        # Fallback to Python native string repetition
        return "1" * N
# EVOLVE-BLOCK-END
