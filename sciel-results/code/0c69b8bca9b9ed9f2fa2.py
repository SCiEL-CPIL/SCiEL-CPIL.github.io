"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    # We need the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    # We can compute phi(n) for n from 1 upwards and return the first one satisfying the condition.
    # To be efficient, we can use a sieve-like method to compute phi(n) for all n up to some limit,
    # or we can check n one by one using the formula.
    # Since we need the MINIMUM n, we can just iterate n from 1 upwards and check phi(n) % 120 == 0.
    # The answer is known to be 143 (phi(143) = phi(11)*phi(13) = 10*12 = 120).
    # Let's verify: 143 = 11 * 13. phi(143) = 143 * (1 - 1/11) * (1 - 1/13) = 143 * (10/11) * (12/13) = 13 * 10 * 12 / 11 = 120.
    # Is there a smaller n?
    # Let's check a few small values to be sure.
    # phi(1) = 1
    # phi(2) = 1
    # ...
    # We'll just compute it to be safe, but we can also hardcode if we're confident.
    # However, the task says "Minimize n", so we should find the true minimum.
    # Let's write a helper to compute phi(n).
    
    def phi(n):
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
    
    # We can optimize by noting that phi(n) is even for n > 2.
    # Also, 120 = 2^3 * 3 * 5.
    # So phi(n) must be divisible by 8, 3, and 5.
    
    # Let's just iterate and check. The search space is small enough for the first few thousand.
    # Actually, we can be smarter. 
    # But for safety and correctness, let's just do a linear search with an efficient phi computation.
    # Given the time limit is 60s, even a linear search up to 1000000 with a fast phi should be fine,
    # but we expect to find the answer very early (around 143).
    
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    for n in range(1, 1000001):
        if time.time() > deadline:
            # If we somehow didn't find it, return a known valid answer (though we should have found it long ago)
            return 143
        if phi(n) % 120 == 0:
            return n
    
    # Fallback, should not be reached
    return 143
# EVOLVE-BLOCK-END
