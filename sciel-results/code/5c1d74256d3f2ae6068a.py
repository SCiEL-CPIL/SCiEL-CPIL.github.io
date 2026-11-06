"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution is all ones, which scores 100 (vs 50 for seed).
    """
    # Track deadline with safety margin
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill
    
    # The OneMax objective is trivially maximized by setting all bits to 1.
    # This is the global optimum with score = N = 100.
    # No search needed; direct construction is optimal.
    result = "1" * N
    
    # Verify length is exactly N
    if len(result) != N:
        result = result[:N].ljust(N, "1")
    
    return result
# EVOLVE-BLOCK-END
