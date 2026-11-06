"""
Task: OneMax direct binary construction (100 bits)
Objective: Maximize the number of ones in a 100-bit binary string.
Solution: Return a string of 100 ones.
"""

import time

N = 100
# The optimal solution is a string of N ones.
OPTIMAL_RESULT = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string with all bits set to 1.
    This maximizes the OneMax objective function.
    Includes an explicit deadline guard to ensure compliance with compute budget.
    """
    start_time = time.time()
    
    # The computation is O(N) and instantaneous.
    # We construct the result immediately.
    result = OPTIMAL_RESULT
    
    # Explicit deadline check mechanism.
    # The evaluator kills after 150s. We set a conservative internal deadline of 145s.
    # If for any reason the system is stalled or time has passed (e.g., due to GIL issues
    # in a multi-threaded environment, though unlikely for this simple function),
    # we ensure we return the best result found (which is the optimal one) before the limit.
    
    # In this specific task, the result is deterministic and pre-computed.
    # The guard is primarily to satisfy the directive structure and ensure safety.
    if time.time() - start_time > 145.0:
        # This branch is effectively unreachable for this O(N) task,
        # but ensures we don't hang if the environment is pathological.
        return result

    return result
# EVOLVE-BLOCK-END
