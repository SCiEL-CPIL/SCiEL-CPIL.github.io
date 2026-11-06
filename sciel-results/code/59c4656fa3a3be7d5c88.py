import time

N = 100
# The evaluator kills the program after 150s. 
# We set a deadline slightly below this to ensure we return a result in time.
DEADLINE_SECONDS = 145.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string with all bits set to 1.
    This is the optimal solution for the OneMax objective.
    """
    # Explicit deadline guard as required by the contract.
    # Since the operation is O(1) effectively (string multiplication of fixed length),
    # this guard will never trigger, but it satisfies the structural requirement.
    if time.time() - START_TIME > DEADLINE_SECONDS:
        return "1" * N

    # Direct construction of the optimal solution.
    # "1" * 100 is the fastest and most reliable way to generate the result.
    return "1" * N
# EVOLVE-BLOCK-END
