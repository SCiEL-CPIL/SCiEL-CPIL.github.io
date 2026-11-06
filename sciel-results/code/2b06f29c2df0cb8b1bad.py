"""Program to find the smallest n in [1, 1000000] with phi(n) divisible by 120."""
# EVOLVE-BLOCK-START

def _phi(n):
    """Compute Euler's totient function phi(n) using trial division."""
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

def construct():
    # Iterate n from 1 upwards.
    # Since the smallest n is expected to be small, trial division is efficient.
    # We stop as soon as we find a valid n.
    for n in range(1, 1000001):
        if _phi(n) % 120 == 0:
            return n
    return 14400  # Fallback

# EVOLVE-BLOCK-END
