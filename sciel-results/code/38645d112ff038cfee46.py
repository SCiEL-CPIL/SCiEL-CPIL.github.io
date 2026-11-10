"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 512
    
    # Strategy: Find the integer M <= N with the maximum number of divisors.
    # The set of all divisors of M is a valid clique because for any a, b | M,
    # lcm(a, b) | M, so lcm(a, b) <= M <= N.
    # This provides a strong lower bound. Given the constraints and time limits,
    # this is likely optimal or very close to it, and runs instantly.
    
    max_divisors = 0
    best_M = 1
    
    # Helper to count divisors
    def count_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    
    for m in range(1, N + 1):
        d = count_divisors(m)
        if d > max_divisors:
            max_divisors = d
            best_M = m
    
    # Generate divisors of best_M
    divisors = []
    i = 1
    while i * i <= best_M:
        if best_M % i == 0:
            divisors.append(i)
            if i != best_M // i:
                divisors.append(best_M // i)
        i += 1
    
    divisors.sort()
    
    # Optional: Try to improve by adding elements that are compatible with all existing ones?
    # Given the structure, adding an element x not dividing M would require lcm(x, d) <= N for all d | M.
    # This implies x must be "compatible" with the "maximal" divisors of M.
    # However, the divisor set is a clique. Any superset must also be a clique.
    # If we try to add x, it must connect to ALL divisors of M.
    # In particular, it must connect to M itself (if M is in the set).
    # lcm(x, M) <= N.
    # Also x must connect to all other divisors.
    # This is a strict condition. For M=480, divisors include 480.
    # lcm(x, 480) <= 512.
    # Since 480 | lcm(x, 480), lcm(x, 480) must be a multiple of 480.
    # The only multiple of 480 <= 512 is 480 itself.
    # So lcm(x, 480) = 480, which implies x | 480.
    # Thus, no element outside the divisor set of M can be added if M is in the set.
    # Since the divisor set includes M, the set of divisors of M is a maximal clique
    # with respect to adding elements that are compatible with M.
    # Could there be a larger clique that does NOT include M?
    # Possibly, but finding it requires search.
    # Given the time limit, returning the divisor set of the most composite number is the safest
    # and fastest robust solution.
    
    # Verify the solution just in case
    for i in range(len(divisors)):
        for j in range(i + 1, len(divisors)):
            a, b = divisors[i], divisors[j]
            g = gcd(a, b)
            if a * b // g > N:
                # Fallback to a smaller safe set
                return [1, 2, 4, 8, 16, 32, 64, 128, 256]
                
    return divisors
# EVOLVE-BLOCK-END
