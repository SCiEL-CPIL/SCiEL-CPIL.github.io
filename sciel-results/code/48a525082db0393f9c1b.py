"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # OneMax with 100 bits: the global optimum is the string of N ones.
    # Return it directly; add a minimal deadline guard per directive.
    start = time.perf_counter()
    budget = 145.0  # safety margin before the 150s wall-clock kill
    # Direct construction of the known optimum
    result = "1" * N
    # The task is unconstrained beyond length and binary alphabet,
    # so no search is needed. Guard is included to satisfy the
    # explicit-deadline requirement without adding overhead.
    if time.perf_counter() - start > budget:
        # Extremely unlikely to trigger for this constant-time construction,
        # but ensures we never exceed the wall-clock budget.
        return result
    return result
# EVOLVE-BLOCK-END
