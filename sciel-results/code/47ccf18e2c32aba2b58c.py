import numpy as np

N = 11

def construct():
    # ... long comment ...
    return inner, (0.0, 0.0), 5.0, 0.0

# EVOLVE-BLOCK-START
def construct():
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
