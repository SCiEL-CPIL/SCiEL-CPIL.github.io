import random
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Direct construction of the global optimum for OneMax.
    # This is the most robust and fastest "polish" pass.
    return '1' * N
# EVOLVE-BLOCK-END
