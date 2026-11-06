"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction of the optimal OneMax solution: a 100-bit string of all ones.
    This achieves the theoretical maximum of 100 ones.
    Includes a strict deadline guard to ensure safe termination.
    """
    t0 = time.perf_counter()
    deadline = t0 + 145.0  # 5-second safety margin against the 150s limit

    # The global optimum for OneMax with N=100 is all ones.
    result = "1" * N

    # Defensive check: if somehow the construction took too long (impossible), return anyway.
    # This ensures we never exceed the wall-clock limit.
    if time.perf_counter() >= deadline:
        return result

    return result
# EVOLVE-BLOCK-END
