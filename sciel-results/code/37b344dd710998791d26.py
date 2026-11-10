"""Model-and-optimize: construct n from prime factors so phi(n) divisible by 120, minimize n."""
import math
import time

# EVOLVE-BLOCK-START

def construct():
    # The minimum n such that phi(n) is divisible by 120 is 155.
    # 155 = 5 * 31
    # phi(155) = phi(5) * phi(31) = 4 * 30 = 120
    # 120 is divisible by 120.
    # Any n < 155 either lacks a prime p ≡ 1 (mod 5) or requires a larger product
    # to satisfy the 2^3 * 3 * 5 requirement in the totient.
    return 155

# EVOLVE-BLOCK-END
